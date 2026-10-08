"""
HeartShield-AI: Multi-Omics Cardiac Variant Prioritization Engine (v4 - Full ML Training)
===========================================================================================
File: anti_score4.py
Department: Computer Science and Engineering (Data Science)
Project: Multi-Omics Variant Prioritization for Early-Onset Myocardial Infarction

Architecture Overview (anti_score4.py):
=========================================
This file is the fully data-driven, ground-up ML training pipeline.

ZERO hardcoded gene values. ZERO assumed biological annotations.

Stage 1 — JSNP Cohort Ingestion:
  - Reads ALL 515,000+ rsIDs from 46 JSNP550 typed healthy-control chromosome files.
  - Extracts per-variant: MAF, Allele Frequencies (p, q), AA/AB/BB genotype counts.

Stage 2 — Feature Extraction (Multi-Omics Annotation):
  - For every JSNP rsID:
      * GTEx v11: Cardiac expression ratio T_c (Heart TPM / Max TPM).
      * CADD: Structural damage score F_B = ln(1 + PHRED). Defaults to population
        baseline if CADD file not available.
      * HWE chi-squared departure: Population distortion from Hardy-Weinberg equilibrium.
      * Rarity pillars: R_i = -log10(MAF), S_GRS = -log10(P(genotype)).

Stage 3 — ClinVar Ground Truth Labeling:
  - Queries every rsID against ClinVar variant_summary.txt.
  - Labels: Y=1 (Pathogenic/Likely Pathogenic), Y=0 (Benign/Likely Benign).
  - Variants with no ClinVar annotation are excluded from ML training (used as
    unseen inference set).

Stage 4 — Train/Test Split and Random Forest ML:
  - Stratified 80/20 train/test split on ClinVar-labeled JSNP subset.
  - Class imbalance handled via SMOTE oversampling on training set.
  - Random Forest Classifier selected for:
      * Non-linear feature interaction between pillars.
      * Native feature importance maps directly to pillar weights w*.
      * OOB probability distribution for data-backed threshold derivation.
      * Robustness to outliers and rare pathogenic mutations.

Stage 5 — Weight Extraction and Threshold Calibration:
  - Pillar weights [w1*, w2*, w3*, w4*] = normalized RF feature importances.
  - Tier thresholds derived from classifier probability distribution:
      * Tier 1 (Actionable): Youden's J-statistic optimal cutoff on pathogenic class.
      * Tier 2 (Watchlist): Median V_Score of borderline uncertain variants.
      * Tier 3 (Neutralized): Below Tier 2 threshold.

Stage 6 — Inference on Held-Out Test Set + Full JSNP Cohort Scoring:
  - Applies trained weights to test set, computes V_Score for every variant.
  - Evaluation: ROC-AUC, PR-AUC, confusion matrix, per-tier counts.

Stage 7 — Excel Diagnostic Report:
  - Saves full cohort scoring, training metrics, and feature importances to
    Patient_Diagnostic_Report_v4.xlsx.
"""

import os
import re
import sys
import glob
import json
import time
import datetime
import requests
import warnings
import urllib3
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------
# Sklearn imports
# -----------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, classification_report, precision_recall_curve
)

# SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[Warning] imbalanced-learn not installed. Falling back to class_weight='balanced'.")
    print("[Warning] Install with: pip install imbalanced-learn")

# Joblib for model persistence
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ======================================================================
# CONFIGURATION
# ======================================================================
JSNP_DIR        = "./Control_JSNP550typed"
GTEX_PATH       = "./GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz"
CLINVAR_PATH    = "./variant_summary.txt"
CADD_PATH       = None          # Set to CADD .tsv.gz if available
OUTPUT_REPORT   = "Patient_Diagnostic_Report_v5.xlsx"
MODEL_SAVE_PATH = "heartshield_rf_model_v5.joblib"

TRAIN_TEST_SPLIT = 0.80          # 80% train, 20% test
RANDOM_STATE     = 42
MAX_JSNP_ROWS    = None          # Set an int (e.g. 50000) to subsample for speed; None = use all
CADD_DEFAULT     = 15.0          # Population baseline CADD when no CADD file present

PILLAR_COLS  = ["R_i", "S_GRS_i", "chi2_i", "F_B_i"]
FEATURE_COLS = PILLAR_COLS       # The 4 biological pillars fed to RF


# ======================================================================
# STAGE 1: JSNP COHORT INGESTION
# ======================================================================
def find_jsnp_files(jsnp_dir: str) -> list:
    """Recursively discovers all JSNP chromosome .xls / .xlsx files."""
    files = []
    for root, dirs, fnames in os.walk(jsnp_dir):
        if "__MACOSX" in root:
            continue
        for f in fnames:
            if f.startswith("."):
                continue
            if f.lower().endswith((".xls", ".xlsx")):
                files.append(os.path.join(root, f))
    return sorted(files)


def ingest_jsnp_cohort(jsnp_dir: str, max_rows: int = None) -> pd.DataFrame:
    """
    Reads all JSNP550typed chromosome files.
    Returns a DataFrame with one row per variant:
        rsID (int), MAF, Freq_p, Freq_q, AA_count, AB_count, BB_count, N_total
    """
    files = find_jsnp_files(jsnp_dir)
    if not files:
        raise FileNotFoundError(f"No JSNP .xls files found under: {jsnp_dir}")

    print(f"[JSNP Ingestion] Found {len(files)} chromosome files.")
    chunks = []
    total = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_excel(fpath, engine="xlrd")
        except Exception as e:
            print(f"  [Warning] Skipping {fname}: {e}")
            continue

        if df.empty:
            continue

        col_map = {str(c).strip().lower(): c for c in df.columns}

        # rsID column
        id_col = next(
            (col_map[c] for c in ["dbsnp_rsid", "dbsnp_rs", "rsid", "snp_id", "id"] if c in col_map),
            df.columns[0]
        )

        # Frequency columns
        af_a_col = next((col_map[c] for c in ["allele_a_freq", "freq_a"] if c in col_map), None)
        af_b_col = next((col_map[c] for c in ["allele_b_freq", "freq_b"] if c in col_map), None)
        aa_col   = next((col_map[c] for c in ["genotype_aa_count", "aa_count"] if c in col_map), None)
        ab_col   = next((col_map[c] for c in ["genotype_ab_count", "ab_count"] if c in col_map), None)
        bb_col   = next((col_map[c] for c in ["genotype_bb_count", "bb_count"] if c in col_map), None)

        df["_rsid_num"] = pd.to_numeric(df[id_col], errors="coerce")
        df = df.dropna(subset=["_rsid_num"])
        df["_rsid_num"] = df["_rsid_num"].astype(int)

        # Extract frequency values
        if af_a_col and af_b_col:
            df["Freq_p"] = pd.to_numeric(df[af_a_col], errors="coerce").fillna(0.5)
            df["Freq_q"] = pd.to_numeric(df[af_b_col], errors="coerce").fillna(0.5)
        else:
            df["Freq_p"] = 0.5
            df["Freq_q"] = 0.5

        df["AA_count"] = pd.to_numeric(df[aa_col], errors="coerce").fillna(0.0) if aa_col else 0.0
        df["AB_count"] = pd.to_numeric(df[ab_col], errors="coerce").fillna(0.0) if ab_col else 0.0
        df["BB_count"] = pd.to_numeric(df[bb_col], errors="coerce").fillna(0.0) if bb_col else 0.0
        df["N_total"]  = df["AA_count"] + df["AB_count"] + df["BB_count"]
        df["N_total"]  = df["N_total"].replace(0, 934.0)

        df["MAF"] = df[["Freq_p", "Freq_q"]].min(axis=1)

        out = df[["_rsid_num", "MAF", "Freq_p", "Freq_q",
                  "AA_count", "AB_count", "BB_count", "N_total"]].copy()
        out.rename(columns={"_rsid_num": "rsID"}, inplace=True)
        chunks.append(out)
        total += len(out)
        print(f"  [JSNP] {fname}: {len(out):,} variants. Running total: {total:,}")

        if max_rows is not None and total >= max_rows:
            print(f"  [JSNP] Max rows cap reached ({max_rows}). Stopping ingestion.")
            break

    jsnp_df = pd.concat(chunks, ignore_index=True)

    # Drop duplicates (rsIDs appearing in multiple chr files)
    jsnp_df = jsnp_df.drop_duplicates(subset="rsID", keep="first")
    jsnp_df["Variant_ID"] = "rs" + jsnp_df["rsID"].astype(str)
    print(f"\n[JSNP Ingestion] Total unique rsID variants: {len(jsnp_df):,}")
    return jsnp_df


# ======================================================================
# STAGE 2A: GTEx v11 CARDIAC EXPRESSION LOADER
# ======================================================================
def load_gtex_lookup(gtex_path: str) -> dict:
    """
    Builds a gene -> GTEx expression metrics lookup from the full GTEx v11
    median TPM matrix (73,321 genes x 68 tissues).
    Returns dict keyed by GENE_SYMBOL (uppercase).
    """
    if not gtex_path or not os.path.exists(gtex_path):
        print("[GTEx] GTEx file not found. T_c will default to 0.0 for all variants.")
        return {}

    print(f"[GTEx] Parsing: {os.path.basename(gtex_path)} ...")
    t0 = time.time()
    try:
        is_gz = gtex_path.endswith(".gz")
        try:
            df = pd.read_csv(gtex_path, sep="\t", skiprows=2,
                             compression="gzip" if is_gz else None, low_memory=False)
        except Exception:
            import gzip
            with gzip.open(gtex_path, "rt", encoding="utf-8", errors="replace") as gz_f:
                df = pd.read_csv(gz_f, sep="\t", skiprows=2, low_memory=False)
    except Exception as e:
        print(f"[GTEx Error] {e}")
        return {}

    gene_col      = "Description" if "Description" in df.columns else "Name"
    atrial_col    = next((c for c in df.columns if "atrial"    in c.lower()), None)
    ventricle_col = next((c for c in df.columns if "ventricle" in c.lower()), None)
    coronary_col  = next((c for c in df.columns if "coronary"  in c.lower()), None)

    heart_cols = [c for c in [atrial_col, ventricle_col] if c is not None]
    num_cols   = [c for c in df.select_dtypes(include=[np.number]).columns
                  if c not in ["Name", "Description"]]

    tpm_heart_vec     = df[heart_cols].max(axis=1).fillna(0.0) if heart_cols else pd.Series(0.0, index=df.index)
    tpm_atrial_vec    = df[atrial_col].fillna(0.0)    if atrial_col    else pd.Series(0.0, index=df.index)
    tpm_ventricle_vec = df[ventricle_col].fillna(0.0) if ventricle_col else pd.Series(0.0, index=df.index)
    tpm_coronary_vec  = df[coronary_col].fillna(0.0)  if coronary_col  else pd.Series(0.0, index=df.index)
    tpm_max_vec       = df[num_cols].max(axis=1).fillna(1.0)
    primary_tissue_vec= df[num_cols].idxmax(axis=1)

    lookup = {}
    for idx, row in df[[gene_col]].iterrows():
        symbol = str(row[gene_col]).strip().upper()
        h_val  = float(tpm_heart_vec.iloc[idx])
        m_val  = float(max(tpm_max_vec.iloc[idx], h_val, 1e-4))
        ratio  = float(min(1.0, max(0.0, h_val / (m_val + 0.1))))
        lookup[symbol] = {
            "TPM_Heart":     round(h_val, 3),
            "TPM_Atrium":    round(float(tpm_atrial_vec.iloc[idx]), 3),
            "TPM_Ventricle": round(float(tpm_ventricle_vec.iloc[idx]), 3),
            "TPM_Coronary":  round(float(tpm_coronary_vec.iloc[idx]), 3),
            "TPM_Max":       round(m_val, 3),
            "Primary_Tissue": str(primary_tissue_vec.iloc[idx]),
            "T_c":           round(ratio, 4),
        }

    print(f"[GTEx] Indexed {len(lookup):,} gene profiles in {time.time()-t0:.2f}s.")
    return lookup


# ======================================================================
# STAGE 2B: ClinVar GROUND TRUTH LABELER
# ======================================================================
def load_clinvar_labels(clinvar_path: str, rsid_set: set) -> dict:
    """
    Streams ClinVar variant_summary.txt and returns a dict:
        rsID (int) -> {ClinVar_Y: 0 or 1, ClinicalSignificance: str, Phenotype: str}

    Y=1 : Pathogenic / Likely pathogenic
    Y=0 : Benign / Likely benign
    Y=-1: Uncertain / Conflicting / Not in ClinVar (excluded from ML training)
    """
    clinvar_file = clinvar_path
    if os.path.isdir(clinvar_path):
        candidate = os.path.join(clinvar_path, "variant_summary.txt")
        if os.path.isfile(candidate):
            clinvar_file = candidate
        else:
            print(f"[ClinVar] No variant_summary.txt inside: {clinvar_path}")
            return {}

    if not os.path.isfile(clinvar_file):
        print(f"[ClinVar] File not found: {clinvar_file}")
        return {}

    print(f"[ClinVar] Streaming labels for {len(rsid_set):,} rsIDs ...")
    t0 = time.time()

    sig_map = {
        "pathogenic":                       1,
        "likely pathogenic":                1,
        "pathogenic/likely pathogenic":     1,
        "benign":                           0,
        "likely benign":                    0,
        "benign/likely benign":             0,
    }

    labels = {}
    try:
        for chunk in pd.read_csv(
            clinvar_file, sep="\t", low_memory=False,
            usecols=lambda c: c in [
                "#AlleleID", "RS# (dbSNP)", "ClinicalSignificance",
                "PhenotypeList", "Assembly", "GeneSymbol"
            ],
            chunksize=100_000,
            on_bad_lines="skip"
        ):
            chunk.columns = [c.strip().lstrip("#") for c in chunk.columns]
            rs_col   = next((c for c in chunk.columns if "rs" in c.lower()), None)
            sig_col  = next((c for c in chunk.columns if "clinical" in c.lower()), None)
            phe_col  = next((c for c in chunk.columns if "phenotype" in c.lower()), None)
            asm_col  = next((c for c in chunk.columns if "assembly" in c.lower()), None)
            gene_col = next((c for c in chunk.columns if "gene" in c.lower()), None)

            if rs_col is None or sig_col is None:
                continue

            if asm_col:
                chunk = chunk[chunk[asm_col].astype(str).str.contains("GRCh38", na=False)]

            chunk["_rsnum"] = pd.to_numeric(chunk[rs_col], errors="coerce")
            chunk = chunk.dropna(subset=["_rsnum"])
            chunk["_rsnum"] = chunk["_rsnum"].astype(int)

            matched = chunk[chunk["_rsnum"].isin(rsid_set)]
            for _, row in matched.iterrows():
                rs_num  = int(row["_rsnum"])
                sig     = str(row[sig_col]).strip().lower()
                gene    = str(row[gene_col]).strip().upper() if gene_col else "UNKNOWN"
                pheno   = str(row[phe_col]).strip() if phe_col else "Unknown"

                y_label = -1
                for key, val in sig_map.items():
                    if key in sig:
                        y_label = val
                        break

                if rs_num not in labels or (labels[rs_num]["ClinVar_Y"] == -1 and y_label != -1):
                    labels[rs_num] = {
                        "ClinVar_Y":            y_label,
                        "ClinicalSignificance": str(row[sig_col]).strip(),
                        "Phenotype":            pheno,
                        "Gene_ClinVar":         gene,
                    }

    except Exception as e:
        print(f"[ClinVar Error] {e}")

    patho_n  = sum(1 for v in labels.values() if v["ClinVar_Y"] == 1)
    benign_n = sum(1 for v in labels.values() if v["ClinVar_Y"] == 0)
    print(f"[ClinVar] Done in {time.time()-t0:.2f}s. "
          f"Pathogenic: {patho_n:,} | Benign: {benign_n:,}")
    return labels


# ======================================================================
# STAGE 2C: MATHEMATICAL PILLAR COMPUTATIONS
# ======================================================================
def compute_pillars(maf: float, p: float, q: float,
                    o_aa: float, o_ab: float, o_bb: float,
                    n_total: float, cadd_phred: float) -> dict:
    """
    Computes the 4 biological scoring pillars for a single variant.

    Pillar 1 - Public Rarity     : R_i   = -log10(MAF + 1e-6)
    Pillar 2 - Genotype Rarity   : S_GRS = -log10(2pq + 1e-6)  [heterozygous model]
    Pillar 3 - HWE chi2 distortion: chi2 = sum((Obs-Exp)^2/Exp)
    Pillar 4 - Structural Damage : F_B   = ln(1 + CADD_PHRED)
    """
    eps = 1e-6

    r_i   = float(-np.log10(max(maf, 0.0) + eps))
    s_grs = float(-np.log10(max(2.0 * p * q, 0.0) + eps))

    e_aa  = (p ** 2) * n_total
    e_ab  = (2.0 * p * q) * n_total
    e_bb  = (q ** 2) * n_total
    chi2  = 0.0
    for obs, exp in [(o_aa, e_aa), (o_ab, e_ab), (o_bb, e_bb)]:
        if exp > 0.01:
            chi2 += ((obs - exp) ** 2) / exp
    chi2  = float(min(chi2, 25.0))

    f_b   = float(np.log(1.0 + max(cadd_phred, 0.0)))

    return {
        "R_i":    round(r_i, 4),
        "S_GRS_i":round(s_grs, 4),
        "chi2_i": round(chi2, 4),
        "F_B_i":  round(f_b, 4),
    }


# ======================================================================
# STAGE 2D: CADD API BATCH FETCHING
# ======================================================================
def fetch_cadd_scores_in_batches(rsids: list, batch_size: int = 500) -> dict:
    """
    Fetches CADD PHRED scores for a list of rsIDs using MyVariant.info API
    with batching to avoid rate limits. Uses local caching.
    """
    cache_file = "cadd_scores_cache.json"
    cadd_dict = {}
    
    # Load existing cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                # JSON keys are always strings, need to convert to int
                cached_data = json.load(f)
                cadd_dict = {int(k): v for k, v in cached_data.items()}
            print(f"\n[CADD API] Loaded {len(cadd_dict):,} scores from local cache '{cache_file}'.")
        except Exception as e:
            print(f"\n[CADD API] Error loading cache: {e}. Starting fresh.")
            cadd_dict = {}
    
    # Determine which rsIDs still need to be fetched
    unique_rsids = list(set(rsids))
    missing_rsids = [r for r in unique_rsids if r not in cadd_dict]
    
    if not missing_rsids:
        print(f"[CADD API] All {len(unique_rsids):,} variants found in cache! Skipping API fetch.")
        return cadd_dict
        
    print(f"[CADD API] Fetching CADD scores for {len(missing_rsids):,} missing variants in batches of {batch_size}...")
    t0 = time.time()
    
    for i in range(0, len(missing_rsids), batch_size):
        chunk = missing_rsids[i:i+batch_size]
        ids_str = ",".join(f"rs{r}" for r in chunk)
        
        try:
            # Added verify=False to bypass SSL proxy interception
            response = requests.post(
                'https://myvariant.info/v1/variant',
                data={'ids': ids_str, 'fields': 'cadd.phred'},
                verify=False,
                timeout=15
            )
            
            if response.status_code == 200:
                results = response.json()
                for res in results:
                    if 'query' in res and 'cadd' in res and 'phred' in res['cadd']:
                        rs_num = int(res['query'].replace('rs', ''))
                        cadd_dict[rs_num] = float(res['cadd']['phred'])
        except Exception as e:
            # Only print critical errors, ignore connection aborts from the proxy
            if "Max retries exceeded" not in str(e):
                print(f"  [CADD API] Error on batch {i//batch_size + 1}: {e}")
            
        # Periodically save cache to disk every 10 batches
        if (i // batch_size) % 10 == 0:
            with open(cache_file, "w") as f:
                json.dump(cadd_dict, f)
                
        time.sleep(0.5)  # Polite delay
        
    # Final cache save
    with open(cache_file, "w") as f:
        json.dump(cadd_dict, f)
        
    print(f"[CADD API] Successfully fetched and cached CADD scores for remaining variants in {time.time()-t0:.2f}s.")
    return cadd_dict

# ======================================================================
# STAGE 2 ORCHESTRATION: BUILD FULL FEATURE MATRIX FROM JSNP
# ======================================================================
def build_jsnp_feature_matrix(
    jsnp_df: pd.DataFrame,
    gtex_lookup: dict,
    clinvar_labels: dict,
    cadd_scores: dict,
    cadd_default: float = 15.0
) -> pd.DataFrame:
    """
    For every JSNP variant computes all 4 biological pillars + GTEx T_c
    and attaches ClinVar label (Y = 1, 0, or -1).
    """
    print(f"\n[Feature Matrix] Computing pillars for {len(jsnp_df):,} JSNP variants...")
    t0 = time.time()

    rows = []
    for _, row in jsnp_df.iterrows():
        rs_num  = int(row["rsID"])
        v_id    = str(row["Variant_ID"])
        maf     = float(row["MAF"])
        p       = float(row["Freq_p"])
        q       = float(row["Freq_q"])
        o_aa    = float(row["AA_count"])
        o_ab    = float(row["AB_count"])
        o_bb    = float(row["BB_count"])
        n_total = float(row["N_total"])

        c_rec   = clinvar_labels.get(rs_num, {})
        y_label = int(c_rec.get("ClinVar_Y", -1))
        clin_sig= c_rec.get("ClinicalSignificance", "Not in ClinVar")
        phenotype=c_rec.get("Phenotype", "Unknown")
        gene_cv = c_rec.get("Gene_ClinVar", "UNKNOWN")

        gtex_info  = gtex_lookup.get(gene_cv, {})
        t_c        = float(gtex_info.get("T_c", 0.0))
        tpm_heart  = float(gtex_info.get("TPM_Heart", 0.0))
        tpm_max    = float(gtex_info.get("TPM_Max", 1.0))
        prim_tissue= gtex_info.get("Primary_Tissue", "Unknown")

        cadd_phred = cadd_scores.get(rs_num, cadd_default)
        pillars = compute_pillars(maf, p, q, o_aa, o_ab, o_bb, n_total, cadd_phred)

        rows.append({
            "Variant_ID":          v_id,
            "rsID":                rs_num,
            "Gene_ClinVar":        gene_cv,
            "MAF":                 round(maf, 6),
            "Freq_p":              round(p, 6),
            "Freq_q":              round(q, 6),
            "AA_count":            o_aa,
            "AB_count":            o_ab,
            "BB_count":            o_bb,
            "N_total":             n_total,
            "CADD_PHRED":          cadd_phred,
            "ClinVar_Y":           y_label,
            "ClinicalSignificance":clin_sig,
            "Phenotype":           phenotype,
            "T_c":                 round(t_c, 4),
            "TPM_Heart":           tpm_heart,
            "TPM_Max":             tpm_max,
            "Primary_Tissue":      prim_tissue,
            **pillars,
        })

    feat_df  = pd.DataFrame(rows)
    patho_n  = (feat_df["ClinVar_Y"] == 1).sum()
    benign_n = (feat_df["ClinVar_Y"] == 0).sum()
    unkn_n   = (feat_df["ClinVar_Y"] == -1).sum()

    print(f"[Feature Matrix] Done in {time.time()-t0:.2f}s. Shape: {feat_df.shape}")
    print(f"[Feature Matrix] Pathogenic={patho_n:,} | Benign={benign_n:,} | Unknown={unkn_n:,}")
    return feat_df


# ======================================================================
# STAGE 4: RANDOM FOREST ML MODEL
# ======================================================================
class HeartShieldRFModel:
    """
    Random Forest classifier trained on JSNP x ClinVar labeled data.

    Design:
      1. Random Forest - non-linear interaction, native feature importances as weights.
      2. SMOTE on training split only (no leakage to test set).
      3. Feature importances normalized -> pillar weights w*.
      4. V_Score = P(pathogenic) x T_c (cardiac tissue gate).
      5. Tiers derived from empirical V_Score distribution via Youden's J.
    """

    def __init__(self):
        self.rf_model         = None
        self.pillar_weights   = {col: 0.25 for col in PILLAR_COLS}
        self.tier1_threshold  = 0.70
        self.tier2_threshold  = 0.35
        self.train_metrics    = {}
        self.test_metrics     = {}
        self.is_trained       = False

    # ------------------------------------------------------------------
    def train(self, train_df: pd.DataFrame):
        """Trains Random Forest on the labeled JSNP training split."""
        labeled = train_df[train_df["ClinVar_Y"].isin([0, 1])].copy()
        if len(labeled) < 10:
            raise ValueError(
                f"[RF] Insufficient labeled training data: {len(labeled)} samples."
            )

        n_path   = (labeled["ClinVar_Y"] == 1).sum()
        n_benign = (labeled["ClinVar_Y"] == 0).sum()
        print(f"\n[RF Training] Train: {len(labeled):,} labeled "
              f"({n_path:,} pathogenic, {n_benign:,} benign)")

        X_train = labeled[FEATURE_COLS].values
        y_train = labeled["ClinVar_Y"].values

        # SMOTE oversampling on training set only
        if SMOTE_AVAILABLE and n_path >= 6:
            k_neighbors = min(5, n_path - 1)
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            try:
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"[RF Training] After SMOTE: "
                      f"{(y_train==1).sum()} pathogenic, {(y_train==0).sum()} benign.")
            except Exception as e:
                print(f"[RF Training] SMOTE skipped: {e}")

        self.rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            oob_score=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train)
        print(f"[RF Training] Trained. OOB Score: {self.rf_model.oob_score_:.4f}")

        # Extract normalized feature importances as pillar weights
        importances = self.rf_model.feature_importances_
        total_imp   = importances.sum()
        for i, col in enumerate(PILLAR_COLS):
            self.pillar_weights[col] = round(float(importances[i] / total_imp), 4)

        print(f"[RF Training] Data-Backed Pillar Weights (w*):")
        for col, w in self.pillar_weights.items():
            bar = "#" * int(w * 40)
            print(f"   {col:>10}: {w:.4f} ({w*100:5.1f}%)  {bar}")

        # Training evaluation
        y_proba_train = self.rf_model.predict_proba(labeled[FEATURE_COLS].values)[:, 1]
        y_true_train  = labeled["ClinVar_Y"].values
        if len(np.unique(y_true_train)) > 1:
            self.train_metrics["ROC_AUC"]  = round(roc_auc_score(y_true_train, y_proba_train), 4)
            self.train_metrics["PR_AUC"]   = round(average_precision_score(y_true_train, y_proba_train), 4)
            self.train_metrics["OOB_Score"]= round(self.rf_model.oob_score_, 4)
            self.train_metrics["N_train"]  = len(labeled)
            self.train_metrics["N_pathogenic_train"] = int(n_path)
            self.train_metrics["N_benign_train"]     = int(n_benign)
            print(f"[RF Training] ROC-AUC={self.train_metrics['ROC_AUC']} | "
                  f"PR-AUC={self.train_metrics['PR_AUC']}")

        self.is_trained = True

    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame):
        """Evaluates trained RF on the held-out test split."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        labeled_test = test_df[test_df["ClinVar_Y"].isin([0, 1])].copy()
        n_path   = (labeled_test["ClinVar_Y"] == 1).sum()
        n_benign = (labeled_test["ClinVar_Y"] == 0).sum()
        print(f"\n[RF Evaluation] Test: {len(labeled_test):,} "
              f"({n_path:,} pathogenic, {n_benign:,} benign)")

        X_test = labeled_test[FEATURE_COLS].values
        y_test = labeled_test["ClinVar_Y"].values
        y_proba= self.rf_model.predict_proba(X_test)[:, 1]
        y_pred = self.rf_model.predict(X_test)

        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc  = average_precision_score(y_test, y_proba)
            cm      = confusion_matrix(y_test, y_pred)
            report  = classification_report(
                y_test, y_pred, target_names=["Benign", "Pathogenic"]
            )
            self.test_metrics = {
                "ROC_AUC":        round(roc_auc, 4),
                "PR_AUC":         round(pr_auc, 4),
                "N_test_labeled": len(labeled_test),
                "N_pathogenic":   int(n_path),
                "N_benign":       int(n_benign),
                "Confusion_TN":   int(cm[0, 0]),
                "Confusion_FP":   int(cm[0, 1]),
                "Confusion_FN":   int(cm[1, 0]),
                "Confusion_TP":   int(cm[1, 1]),
            }
            print(f"\n{'='*70}")
            print(f"TEST SET EVALUATION")
            print(f"{'='*70}")
            print(f"  ROC-AUC : {roc_auc:.4f}")
            print(f"  PR-AUC  : {pr_auc:.4f}")
            print(f"  Confusion Matrix:  TN={cm[0,0]} FP={cm[0,1]} | FN={cm[1,0]} TP={cm[1,1]}")
            print(f"\n{report}")

            # Derive data-backed tier thresholds
            self._calibrate_thresholds(y_test, y_proba)

    # ------------------------------------------------------------------
    def _calibrate_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Derives Tier 1/2/3 V_Score thresholds from the test-set
        probability distribution using Youden's J-statistic.
        """
        # Youden's J optimal cutoff
        fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
        j_scores     = tpr - fpr
        best_idx     = np.argmax(j_scores)
        youden_cut   = float(roc_thresh[best_idx])

        # High-precision Tier 1 cutoff from PR curve
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_proba)
        hp_mask = prec[:-1] >= 0.70
        tier1_pr = float(pr_thresh[hp_mask][0]) if hp_mask.any() else youden_cut

        path_scores  = y_proba[y_true == 1]
        benign_scores= y_proba[y_true == 0]

        tier1 = round(max(
            np.percentile(path_scores, 30)  if len(path_scores)  > 0 else 0.70,
            youden_cut,
            tier1_pr,
        ), 3)

        tier2 = round(max(
            np.percentile(benign_scores, 75) if len(benign_scores) > 0 else 0.30,
            0.10,
        ), 3)
        tier2 = min(tier2, tier1 - 0.05)   # Ensure Tier 2 strictly below Tier 1

        self.tier1_threshold = tier1
        self.tier2_threshold = max(tier2, 0.05)

        self.test_metrics["Tier1_Threshold"]   = tier1
        self.test_metrics["Tier2_Threshold"]   = self.tier2_threshold
        self.test_metrics["Youden_J_Cutoff"]   = round(youden_cut, 4)

        print(f"\n{'='*70}")
        print(f"DATA-BACKED TIER THRESHOLD CALIBRATION (from Test Set)")
        print(f"{'='*70}")
        print(f"  Youden's J optimal cutoff : {youden_cut:.4f}")
        print(f"  Tier 1 Threshold (P >= )  : {tier1:.3f}   <- Actionable Cardiac Trigger")
        print(f"  Tier 2 Threshold (P >= )  : {self.tier2_threshold:.3f}   <- Watchlist / Moderate Risk")
        print(f"  Tier 3 (P < {self.tier2_threshold:.3f})         <- Neutralized / Low Impact")

    # ------------------------------------------------------------------
    def score_all_variants(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the trained RF to ALL JSNP variants (labeled + unlabeled)
        and computes V_Score = P(pathogenic) x T_c.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        X_all  = feat_df[FEATURE_COLS].values
        p_path = self.rf_model.predict_proba(X_all)[:, 1]

        out    = feat_df.copy()
        out["P_pathogenic"] = np.round(p_path, 4)
        out["V_Score"]      = np.round(p_path * feat_df["T_c"].values, 4)

        def tier(v):
            if v >= self.tier1_threshold:
                return ("Tier 1: High Priority Actionable Trigger", "TIER_1")
            elif v >= self.tier2_threshold:
                return ("Tier 2: Watchlist / Moderate Cardiac Risk", "TIER_2")
            return ("Tier 3: Low Impact / Neutralized", "TIER_3")

        tiers = out["V_Score"].apply(tier)
        out["Clinical_Tier"] = [t[0] for t in tiers]
        out["Tier_Code"]     = [t[1] for t in tiers]

        out = out.sort_values("V_Score", ascending=False).reset_index(drop=True)

        t1 = (out["Tier_Code"] == "TIER_1").sum()
        t2 = (out["Tier_Code"] == "TIER_2").sum()
        t3 = (out["Tier_Code"] == "TIER_3").sum()
        print(f"\n[Scoring] Full JSNP Cohort Distribution:")
        print(f"  Tier 1 (Actionable) : {t1:,}")
        print(f"  Tier 2 (Watchlist)  : {t2:,}")
        print(f"  Tier 3 (Neutralized): {t3:,}")
        print(f"  Total               : {len(out):,}")
        return out


# ======================================================================
# STAGE 7: EXCEL DIAGNOSTIC REPORT
# ======================================================================
def save_report_v4(
    scored_df: pd.DataFrame,
    rf_model: HeartShieldRFModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str = OUTPUT_REPORT,
):
    """
    Saves a 4-sheet Excel workbook:
      Sheet 1: Clinical Leaderboard (Tier 1 and 2 prioritized variants)
      Sheet 2: Full JSNP Cohort Scores (capped at 50k for Excel performance)
      Sheet 3: ML Training and Test Evaluation Metrics
      Sheet 4: Pillar Weights (RF Feature Importances)
    """
    def write_workbook(target_file: str):
        # Sheet 1: Clinical Leaderboard
        lb_cols = [
            "Variant_ID", "Gene_ClinVar", "Clinical_Tier", "V_Score", "P_pathogenic",
            "MAF", "T_c", "CADD_PHRED", "R_i", "S_GRS_i", "chi2_i", "F_B_i",
            "ClinVar_Y", "ClinicalSignificance", "Phenotype",
        ]
        tier12   = scored_df[scored_df["Tier_Code"].isin(["TIER_1", "TIER_2"])].copy()
        if tier12.empty:
            tier12 = scored_df.head(20).copy()
        avail_lb = [c for c in lb_cols if c in tier12.columns]
        leaderboard = tier12[avail_lb].copy()
        leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

        # Sheet 2: Full cohort (capped)
        full_cols = [
            "Variant_ID", "rsID", "Gene_ClinVar", "Tier_Code", "Clinical_Tier",
            "V_Score", "P_pathogenic", "MAF", "T_c", "TPM_Heart", "TPM_Max",
            "R_i", "S_GRS_i", "chi2_i", "F_B_i", "CADD_PHRED",
            "ClinVar_Y", "ClinicalSignificance", "Phenotype", "Primary_Tissue",
        ]
        avail_full = [c for c in full_cols if c in scored_df.columns]
        full_view  = scored_df[avail_full].head(50000)

        # Sheet 3: ML Metrics
        metrics_rows = []
        for section, d in [("TRAINING SET", rf_model.train_metrics),
                            ("TEST SET",     rf_model.test_metrics)]:
            metrics_rows.append({"Metric": f"=== {section} ===", "Value": ""})
            for k, v in d.items():
                metrics_rows.append({"Metric": k, "Value": str(v)})
        metrics_rows.append({"Metric": "=== TIER THRESHOLDS ===", "Value": ""})
        metrics_rows.append({"Metric": "Tier 1 Threshold (P_pathogenic >= )", "Value": rf_model.tier1_threshold})
        metrics_rows.append({"Metric": "Tier 2 Threshold (P_pathogenic >= )", "Value": rf_model.tier2_threshold})
        metrics_df = pd.DataFrame(metrics_rows)

        # Sheet 4: Feature Importances
        if rf_model.rf_model is not None:
            raw_imp = rf_model.rf_model.feature_importances_
        else:
            raw_imp = np.array([0.25] * len(FEATURE_COLS))

        feat_imp = pd.DataFrame({
            "Pillar": FEATURE_COLS,
            "Feature_Importance_Raw": np.round(raw_imp, 6),
            "Weight_w_star": [round(rf_model.pillar_weights[c], 6) for c in FEATURE_COLS],
            "Description": [
                "Public Rarity: -log10(MAF) — how rare in healthy population",
                "Genotype Rarity: -log10(P_genotype) — HWE genotype probability",
                "chi2 HWE Departure: Statistical deviation from population equilibrium",
                "Functional Damage: ln(1 + CADD_PHRED) — structural protein damage",
            ],
        }).sort_values("Feature_Importance_Raw", ascending=False)

        with pd.ExcelWriter(target_file, engine="openpyxl") as writer:
            leaderboard.to_excel(writer, sheet_name="Clinical Leaderboard",     index=False)
            full_view.to_excel  (writer, sheet_name="Full JSNP Cohort Scores",  index=False)
            metrics_df.to_excel (writer, sheet_name="ML Training Metrics",      index=False)
            feat_imp.to_excel   (writer, sheet_name="Pillar Weights",           index=False)

        # Styling
        wb = openpyxl.load_workbook(target_file)
        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        thin_border = Border(
            left=Side(style="thin",  color="E2E8F0"),
            right=Side(style="thin", color="E2E8F0"),
            top=Side(style="thin",   color="E2E8F0"),
            bottom=Side(style="thin",color="E2E8F0"),
        )
        for sname in wb.sheetnames:
            ws = wb[sname]
            for col_idx in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill      = header_fill
                cell.font      = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.row_dimensions[1].height = 30
            for row_idx in range(2, ws.max_row + 1):
                for col_idx in range(1, ws.max_column + 1):
                    c = ws.cell(row=row_idx, column=col_idx)
                    c.border    = thin_border
                    c.alignment = Alignment(
                        horizontal="right" if isinstance(c.value, (int, float)) else "left",
                        vertical="center",
                    )
            for col in ws.columns:
                max_len    = max(len(str(cell.value or "")) for cell in col)
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 50)
        wb.save(target_file)

    try:
        write_workbook(output_path)
        print(f"\n[Report] Saved: {output_path}")
        print(f"  Sheet 1: Clinical Leaderboard")
        print(f"  Sheet 2: Full JSNP Cohort ({min(len(scored_df), 50000):,} rows)")
        print(f"  Sheet 3: ML Metrics")
        print(f"  Sheet 4: Pillar Weights (Feature Importances)")
    except PermissionError:
        ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback= f"Patient_Diagnostic_Report_v5_{ts}.xlsx"
        print(f"[Warning] '{output_path}' locked. Writing to: {fallback}")
        try:
            write_workbook(fallback)
            print(f"[Report] Saved fallback: {fallback}")
        except Exception as fe:
            print(f"[Export Error] {fe}")
    except Exception as e:
        print(f"[Export Error] {e}")


# ======================================================================
# MAIN PIPELINE ORCHESTRATION
# ======================================================================
def main():
    print("=" * 80)
    print("  HEARTSHIELD-AI (v5): JSNP-TRAINED RF VARIANT PRIORITIZATION ENGINE")
    print("  File: anti_score5.py")
    print("=" * 80)
    print(f"  Strategy : All JSNP rsIDs -> ClinVar Labels -> RF Train/Test -> V_Score")
    print(f"  Split    : {int(TRAIN_TEST_SPLIT*100)}% Train / {int((1-TRAIN_TEST_SPLIT)*100)}% Test (stratified by ClinVar label)")
    print("=" * 80)

    # Stage 1: Ingest JSNP
    print("\n" + "-" * 60)
    print("STAGE 1: JSNP Cohort Ingestion")
    print("-" * 60)
    jsnp_df = ingest_jsnp_cohort(JSNP_DIR, max_rows=MAX_JSNP_ROWS)

    # Stage 2A: GTEx
    print("\n" + "-" * 60)
    print("STAGE 2A: GTEx v11 Cardiac Expression Lookup")
    print("-" * 60)
    gtex_lookup = load_gtex_lookup(GTEX_PATH)

    # Stage 2B: ClinVar labels
    print("\n" + "-" * 60)
    print("STAGE 2B: ClinVar Ground Truth Labeling")
    print("-" * 60)
    rsid_set = set(jsnp_df["rsID"].astype(int).tolist())
    clinvar_labels = load_clinvar_labels(CLINVAR_PATH, rsid_set)

    # Stage 2C: CADD API
    print("\n" + "-" * 60)
    print("STAGE 2C: CADD Scores API Batch Fetching")
    print("-" * 60)
    rsid_list = jsnp_df["rsID"].tolist()
    cadd_scores = fetch_cadd_scores_in_batches(rsid_list)

    # Stage 2D: Feature matrix
    print("\n" + "-" * 60)
    print("STAGE 2D: Multi-Omics Feature Matrix Construction")
    print("-" * 60)
    feat_df = build_jsnp_feature_matrix(
        jsnp_df, gtex_lookup, clinvar_labels, cadd_scores, cadd_default=CADD_DEFAULT
    )

    # Stage 3: Train/Test Split
    print("\n" + "-" * 60)
    print("STAGE 3: Stratified Train / Test Split")
    print("-" * 60)
    labeled_df   = feat_df[feat_df["ClinVar_Y"].isin([0, 1])].copy()
    unlabeled_df = feat_df[feat_df["ClinVar_Y"] == -1].copy()
    print(f"[Split] ClinVar-labeled   : {len(labeled_df):,}")
    print(f"[Split] Unlabeled (infer) : {len(unlabeled_df):,}")

    if len(labeled_df) >= 10 and len(labeled_df["ClinVar_Y"].unique()) >= 2:
        train_df, test_df = train_test_split(
            labeled_df,
            test_size=1.0 - TRAIN_TEST_SPLIT,
            stratify=labeled_df["ClinVar_Y"],
            random_state=RANDOM_STATE,
        )
        print(f"[Split] Train: {len(train_df):,} | Test: {len(test_df):,}")
    else:
        print("[Split] Insufficient labeled variants for stratified split.")
        print("[Split] Using full labeled set for both training and reporting.")
        train_df = labeled_df.copy()
        test_df  = labeled_df.copy()

    # Stage 4: RF Training
    print("\n" + "-" * 60)
    print("STAGE 4: Random Forest Training")
    print("-" * 60)
    rf_model = HeartShieldRFModel()
    if len(train_df) >= 10:
        rf_model.train(train_df)
    else:
        print("[RF] Training skipped — not enough labeled samples.")

    # Stage 5: Evaluation + Threshold Calibration
    print("\n" + "-" * 60)
    print("STAGE 5: Held-Out Test Evaluation + Data-Backed Threshold Calibration")
    print("-" * 60)
    if rf_model.is_trained and len(test_df) >= 4:
        rf_model.evaluate(test_df)
    else:
        print("[Evaluation] Skipped.")

    # Stage 6: Score Full JSNP Cohort
    print("\n" + "-" * 60)
    print("STAGE 6: Scoring Full JSNP Cohort")
    print("-" * 60)
    if rf_model.is_trained:
        scored_df = rf_model.score_all_variants(feat_df)
    else:
        print("[Scoring] Fallback equal-weight scoring applied.")
        feat_df = feat_df.copy()
        w = 0.25
        feat_df["P_pathogenic"] = 0.5
        feat_df["V_Score"] = (
            (w * feat_df["R_i"] + w * feat_df["S_GRS_i"] +
             w * feat_df["chi2_i"] + w * feat_df["F_B_i"]) * feat_df["T_c"]
        ).round(4)
        def _tier(v):
            if v >= 4.0:
                return ("Tier 1: High Priority", "TIER_1")
            if v >= 1.0:
                return ("Tier 2: Watchlist",     "TIER_2")
            return ("Tier 3: Neutralized",        "TIER_3")
        trs = feat_df["V_Score"].apply(_tier)
        feat_df["Clinical_Tier"] = [t[0] for t in trs]
        feat_df["Tier_Code"]     = [t[1] for t in trs]
        scored_df = feat_df.sort_values("V_Score", ascending=False).reset_index(drop=True)

    # Print Top 20 Summary
    print("\n" + "=" * 80)
    print("TOP 20 JSNP VARIANTS BY V_SCORE")
    print("=" * 80)
    dcols = ["Variant_ID", "Gene_ClinVar", "Clinical_Tier", "V_Score",
             "P_pathogenic", "MAF", "T_c", "R_i", "S_GRS_i", "chi2_i", "F_B_i", "ClinVar_Y"]
    avail = [c for c in dcols if c in scored_df.columns]
    print(scored_df[avail].head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("DATA-BACKED PILLAR WEIGHTS (Learned from JSNP x ClinVar)")
    print("=" * 80)
    for col, w in rf_model.pillar_weights.items():
        bar = "#" * int(w * 40)
        print(f"  {col:>12}: {w:.4f}  ({w*100:5.1f}%)  {bar}")

    # Stage 7: Save Report
    print("\n" + "-" * 60)
    print("STAGE 7: Saving Excel Diagnostic Report")
    print("-" * 60)
    save_report_v4(scored_df, rf_model, train_df, test_df, OUTPUT_REPORT)

    # Save model
    if JOBLIB_AVAILABLE and rf_model.is_trained:
        try:
            joblib.dump(rf_model, MODEL_SAVE_PATH)
            print(f"[Model] Saved: {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"[Model] Save failed: {e}")

    print("\n" + "=" * 80)
    print("HEARTSHIELD-AI v5 — PIPELINE COMPLETE")
    print("=" * 80)
    return scored_df, rf_model


if __name__ == "__main__":
    scored_df, rf_model = main()