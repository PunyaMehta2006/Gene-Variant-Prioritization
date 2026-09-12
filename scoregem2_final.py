import os
import glob
import math
import numpy as np
import pandas as pd


class FullyIntegratedVariantEngine:
    def __init__(
        self,
        jsnp_dir_path: str,
        gtex_tpm_path: str = None,
        cadd_tsv_path: str = None,
        clinvar_path: str = None,
        maf_threshold: float = 0.01,
    ):
        self.jsnp_dir_path = os.path.normpath(jsnp_dir_path) if jsnp_dir_path else None
        self.maf_threshold = maf_threshold

        self.jsnp_lookup_table = {}
        self.gtex_lookup_table = {}
        self.cadd_lookup_table = {}
        self.clinvar_lookup_table = set()

        # Parse datasets using robust multi-encoding and correct parameters
        if self.jsnp_dir_path and os.path.exists(self.jsnp_dir_path):
            self.load_all_jsnp_files()

        if gtex_tpm_path and os.path.exists(gtex_tpm_path):
            self.load_gtex_dataset(gtex_tpm_path)

        if cadd_tsv_path and os.path.exists(cadd_tsv_path):
            self.load_cadd_dataset(cadd_tsv_path)

        if clinvar_path and os.path.exists(clinvar_path):
            self.load_clinvar_dataset(clinvar_path)

    def load_all_jsnp_files(self):
        """Parse JSNP files supporting legacy Excel binary (.xls) and raw text/TSV with encoding fallbacks."""
        if os.path.isdir(self.jsnp_dir_path):
            file_list = sorted([
                os.path.join(self.jsnp_dir_path, f)
                for f in os.listdir(self.jsnp_dir_path)
                if not f.startswith(".")
            ])
        else:
            file_list = [self.jsnp_dir_path]

        record_count = 0
        parsed_files = 0

        for file_path in file_list:
            df = None
            # Fix utf-8 decode errors by handling binary/excel vs text formats with encodings
            if file_path.lower().endswith((".xls", ".xlsx")):
                try:
                    df = pd.read_excel(file_path)
                except Exception:
                    pass

            if df is None:
                for enc in ["latin1", "cp1252", "shift_jis", "utf-8"]:
                    try:
                        df = pd.read_csv(
                            file_path,
                            sep=r"\s+|,|\t",
                            engine="python",
                            encoding=enc,
                            on_bad_lines="skip",
                        )
                        break
                    except Exception:
                        continue

            if df is None or df.empty:
                continue

            df.columns = [str(c).strip().lower() for c in df.columns]

            key_col = next((c for c in ["snp_id", "rsid", "id", "snp", "variant_id", "pos"] if c in df.columns), df.columns[0])

            for _, row in df.iterrows():
                v_key = str(row[key_col]).strip()

                aa_cnt = float(row.get("aa_count", row.get("aa", 0.0)))
                ab_cnt = float(row.get("ab_count", row.get("ab", 0.0)))
                bb_cnt = float(row.get("bb_count", row.get("bb", 0.0)))

                maf_val = None
                for maf_col in ["maf", "freq", "allele_freq", "maf_pop"]:
                    if maf_col in df.columns and pd.notnull(row[maf_col]):
                        try:
                            maf_val = float(row[maf_col])
                            break
                        except ValueError:
                            pass

                if maf_val is None:
                    total_alleles = 2.0 * (aa_cnt + ab_cnt + bb_cnt)
                    maf_val = ((2.0 * bb_cnt) + ab_cnt) / total_alleles if total_alleles > 0 else 0.0

                self.jsnp_lookup_table[v_key] = {
                    "MAF": maf_val,
                    "AA_count": aa_cnt,
                    "AB_count": ab_cnt,
                    "BB_count": bb_cnt,
                    "source_file": os.path.basename(file_path),
                }
                record_count += 1
            parsed_files += 1

        print(f"[JSNP] Parsed {parsed_files}/{len(file_list)} file(s). Indexed {record_count} entries.")

    def load_gtex_dataset(self, gtex_tpm_path: str):
        """Parse GTEx TPM matrix."""
        try:
            is_gz = gtex_tpm_path.endswith(".gz")
            df_gtex = pd.read_csv(
                gtex_tpm_path,
                sep="\t",
                skiprows=2,
                compression="gzip" if is_gz else None,
                low_memory=False,
            )

            gene_col = "Description" if "Description" in df_gtex.columns else "Name"
            heart_cols = [c for c in df_gtex.columns if "Heart" in c or "heart" in c.lower()]

            for _, row in df_gtex.iterrows():
                gene_symbol = str(row[gene_col]).strip().upper()

                tpm_heart = max([float(row[c]) for c in heart_cols if pd.notnull(row[c])] + [0.0]) if heart_cols else 0.0
                numeric_cols = df_gtex.select_dtypes(include=[np.number]).columns
                tpm_max = max([float(row[c]) for c in numeric_cols if pd.notnull(row[c])] + [tpm_heart, 0.0])

                self.gtex_lookup_table[gene_symbol] = {
                    "TPM_Heart": tpm_heart,
                    "TPM_Max": tpm_max,
                }
            print(f"[GTEx] Indexed {len(self.gtex_lookup_table)} gene expression profiles.")
        except Exception as e:
            print(f"[GTEx Error] {e}")

    def load_cadd_dataset(self, cadd_path: str):
        """Parse CADD PHRED score TSV dataset."""
        try:
            df_cadd = pd.read_csv(cadd_path, sep="\t", comment="#", low_memory=False)
            id_col = next((c for c in df_cadd.columns if c.lower() in ["rsid", "variant_id", "id", "snp"]), df_cadd.columns[0])
            phred_col = next((c for c in df_cadd.columns if "phred" in c.lower() or "cadd" in c.lower()), None)

            if phred_col:
                for _, row in df_cadd.iterrows():
                    self.cadd_lookup_table[str(row[id_col]).strip()] = float(row[phred_col])
                print(f"[CADD] Indexed {len(self.cadd_lookup_table)} CADD scores.")
        except Exception as e:
            print(f"[CADD Error] {e}")

    def load_clinvar_dataset(self, clinvar_path: str):
        """Parse ClinVar summary dataset using on_bad_lines='skip' to avoid argument errors."""
        try:
            df_clinvar = pd.read_csv(
                clinvar_path,
                sep="\t",
                low_memory=False,
                on_bad_lines="skip",  # Fixes TypeError: unexpected keyword argument 'errors'
            )
            df_clinvar.columns = [c.lower() for c in df_clinvar.columns]

            id_col = next((c for c in df_clinvar.columns if c in ["rsid", "variationid", "rsid (dbSNP)"]), None)
            sig_col = next((c for c in df_clinvar.columns if "significance" in c or "clinicalsignificance" in c), None)

            if id_col and sig_col:
                for _, row in df_clinvar.iterrows():
                    sig_val = str(row[sig_col]).lower()
                    if "pathogenic" in sig_val and "benign" not in sig_val:
                        self.clinvar_lookup_table.add(str(row[id_col]).strip())
                print(f"[ClinVar] Indexed {len(self.clinvar_lookup_table)} pathogenic variants.")
        except Exception as e:
            print(f"[ClinVar Error] {e}")

    # ==================== MATHEMATICAL FORMULATIONS ====================
    def compute_public_rarity(self, maf: float) -> float:
        """Formula: R_i = -log10(MAF_i + 10^-6)"""
        return -np.log10(maf + 1e-6)

    def compute_genotype_rarity(self, maf: float) -> float:
        """Formula: S_GRS,i = -log10(2 * (1 - MAF_i) * MAF_i)"""
        carrier_freq = 2.0 * (1.0 - maf) * maf
        return -np.log10(carrier_freq + 1e-6)

    def compute_functional_damage(self, b_p_i: float) -> float:
        """Formula: F_B,i = ln(1 + B_p,i) where B_p,i is the CADD PHRED score"""
        return np.log(1.0 + b_p_i)

    def compute_chi2_distortion(self, observed: float, expected: float) -> float:
        """Formula: chi^2_i = (Observed - Expected)^2 / Expected"""
        if expected <= 0:
            return 0.0
        return ((observed - expected) ** 2) / expected

    def compute_cardiac_tissue_gate(self, tpm_heart: float, tpm_max: float) -> float:
        """Formula: T_c,i = TPM_heart / TPM_max"""
        if tpm_max <= 0:
            return 0.0
        return min(1.0, max(0.0, tpm_heart / tpm_max))

    def compute_pedigree_multiplier(self, is_de_novo: bool) -> float:
        """Formula: D_i = 2.5 if de novo else 1.0"""
        return 2.5 if is_de_novo else 1.0

    # ==================== MAIN EVALUATION ENGINE ====================
    def process_dataset(self, variant_query_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        for _, row in variant_query_df.iterrows():
            v_id = str(row["Variant_ID"]).strip()
            gene = str(row.get("Gene", "UNKNOWN")).strip().upper()

            # 1. Populate JSNP parameters
            if v_id in self.jsnp_lookup_table:
                jsnp_rec = self.jsnp_lookup_table[v_id]
                maf = jsnp_rec["MAF"]
                obs_ab = jsnp_rec["AB_count"]
                total_samples = jsnp_rec["AA_count"] + jsnp_rec["AB_count"] + jsnp_rec["BB_count"]
            else:
                maf = float(row.get("MAF", 0.0))
                obs_ab = float(row.get("AB_count", 0.0))
                total_samples = float(row.get("Total_Samples", 1000.0))

            # Hard Gatekeeper Filter (MAF >= 1%)
            if maf >= self.maf_threshold:
                results.append({
                    "Variant_ID": v_id,
                    "Gene": gene,
                    "MAF": round(maf, 5),
                    "CADD_PHRED": row.get("CADD_PHRED", 0.0),
                    "Base_Sum": 0.0,
                    "T_c": 0.0,
                    "V_Score": 0.0,
                    "Clinical_Tier": "Tier 3: Neutralized / Dropped (MAF >= 1%)",
                })
                continue

            # 2. Extract CADD PHRED score (B_p,i)
            b_p_i = self.cadd_lookup_table.get(v_id, float(row.get("CADD_PHRED", 0.0)))

            # 3. Extract Tissue Expression (GTEx)
            if gene in self.gtex_lookup_table:
                tpm_heart = self.gtex_lookup_table[gene]["TPM_Heart"]
                tpm_max = self.gtex_lookup_table[gene]["TPM_Max"]
            else:
                tpm_heart = float(row.get("TPM_Heart", 0.0))
                tpm_max = float(row.get("TPM_Max", 0.0))

            is_de_novo = bool(row.get("is_de_novo", False))

            # Compute formula components directly from slide
            R_i = self.compute_public_rarity(maf)
            S_GRS_i = self.compute_genotype_rarity(maf)
            F_B_i = self.compute_functional_damage(b_p_i)

            # Expected Heterozygotes: 2 * (1 - MAF) * MAF * N
            exp_ab = 2.0 * (1.0 - maf) * maf * total_samples
            exp_ab = max(exp_ab, 1e-4)  # Prevent zero-division
            chi2_i = self.compute_chi2_distortion(obs_ab, exp_ab)

            T_c_i = self.compute_cardiac_tissue_gate(tpm_heart, tpm_max)
            D_i = self.compute_pedigree_multiplier(is_de_novo)

            # Base Component Combination
            base_sum = 0.25 * (R_i + S_GRS_i + F_B_i + chi2_i)

            # Composite Formulation: V_Score = f(R_i, S_GRS,i, F_B,i, chi^2_i, T_c,i, D_i)
            v_score = base_sum * T_c_i * D_i

            # Classification Hierarchy
            if v_score >= 4.0:
                tier = "Tier 1: High Priority Candidate (V_Score >= 4.0)"
            elif 1.5 <= v_score < 4.0:
                tier = "Tier 2: Moderate Risk Candidate"
            else:
                tier = "Tier 3: Low Impact / Benign (V_Score < 1.5)"

            results.append({
                "Variant_ID": v_id,
                "Gene": gene,
                "MAF": round(maf, 6),
                "CADD_PHRED": b_p_i,
                "Base_Sum": round(base_sum, 2),
                "T_c": round(T_c_i, 3),
                "V_Score": round(v_score, 2),
                "Clinical_Tier": tier,
            })

        return pd.DataFrame(results).sort_values(by="V_Score", ascending=False).reset_index(drop=True)

