"""
HeartShield-AI: Multi-Omics Cardiac Variant Prioritization Platform (v3 - ML Enhanced)
========================================================================================
Interactive Clinical Streamlit Dashboard (NO SLIDERS)
Department: Computer Science and Engineering (Data Science)
Project: Multi-Omics Variant Prioritization for Early-Onset Myocardial Infarction
"""

import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import the ML Engine from anti_score3
from anti_score3 import (
    HeartShieldEngineV3,
    HeartShieldMLCalibrator,
    process_patient_genomics,
    save_diagnostic_report,
    load_vcf_stream,
)

# Set page configuration
st.set_page_config(
    page_title="HeartShield-AI | Multi-Omics Prioritization",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .reportview-container {
        background: #0B0F19;
    }
    .main-header {
        font-size: 28px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 14px;
        color: #94A3B8;
        margin-bottom: 20px;
    }
    .tier-card-1 {
        background: linear-gradient(135deg, rgba(255, 42, 109, 0.15), rgba(15, 23, 42, 0.85));
        border: 2px solid #FF2A6D;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .tier-card-2 {
        background: linear-gradient(135deg, rgba(255, 179, 0, 0.15), rgba(15, 23, 42, 0.85));
        border: 2px solid #FFB300;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .tier-card-3 {
        background: linear-gradient(135deg, rgba(100, 116, 139, 0.15), rgba(15, 23, 42, 0.85));
        border: 2px solid #64748B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .tier-card-dropped {
        background: rgba(15, 23, 42, 0.7);
        border: 1px dashed #475569;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.5);
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# CACHED ENGINE INITIALIZATION
# ======================================================================
@st.cache_resource(show_spinner="Loading GTEx v11 RNA-Seq & JSNP genomic databases...")
def get_initialized_engine():
    engine = HeartShieldEngineV3(
        jsnp_dir_path="./Control_JSNP550typed",
        gtex_tpm_path="./GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
        clinvar_path="./variant_summary.txt",
    )
    return engine

engine = get_initialized_engine()

# Initialize session state for persistent cohort results
if "scored_results" not in st.session_state:
    st.session_state.scored_results = None
if "calibrator" not in st.session_state:
    st.session_state.calibrator = None


# ======================================================================
# SIDEBAR NAVIGATION
# ======================================================================
with st.sidebar:
    st.markdown("### 🫀 HEARTSHIELD-AI")
    st.caption("Multi-Omics Variant Prioritization (v3 ML)")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        [
            "📁 Ingest Patient VCF & Prioritize",
            "⚡ Interactive Variant Calculator",
            "🤖 ML Weight & Threshold Calibration",
            "🫀 GTEx v11 Cardiac Explorer",
            "📊 Clinical Leaderboard & Reports",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("**System Specifications:**")
    st.caption("• Reference Assembly: **GRCh38**")
    st.caption(f"• GTEx v11 Genes: **{len(engine.gtex_lookup_table):,}**")
    st.caption("• Target Leakage: **Strictly Zero**")
    st.caption("• Architecture: **Stages 1 - 6**")


# ======================================================================
# SECTION 1: INGEST PATIENT VCF & PRIORITIZE
# ======================================================================
if nav_choice == "📁 Ingest Patient VCF & Prioritize":
    st.markdown('<div class="main-header">Patient Genomic Ingestion & Prioritization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload patient Whole Exome/Genome Sequencing (WES/WGS) VCF files for multi-omics biological prioritization and ML triage.</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Patient Genomic VCF File (.vcf)",
            type=["vcf", "txt"],
            help="Upload a standard VCFv4.2 file containing coordinates, REF, ALT, and optional INFO annotations."
        )

        cohort_options = ["patient_clinical_sample.vcf (Multi-Tier Verification Set)", "sample_random.vcf (Original Sample)"]
        selected_sample_cohort = st.selectbox("Or choose an existing verification cohort:", cohort_options, index=0)

        use_sample = st.checkbox("Run with selected cohort above", value=(uploaded_file is None))

        raw_vcf_text = st.text_area(
            "Or Paste Raw VCF Content Directly",
            placeholder="#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n11\t47364097\trs72552763\tG\tA\t99\tPASS\tGENE=MYBPC3;DE_NOVO",
            height=100
        )

        btn_run = st.button("🚀 Execute Multi-Omics Prioritization Pipeline", type="primary")

    with col_info:
        st.markdown("#### Ingestion Gatekeepers")
        st.info("""
        **1. GRCh38 Gatekeeper:**
        Harmonizes variant coordinates to primary assembly key `chr_pos_ref_alt`.
        
        **2. Public Noise Gate:**
        Filters common benign variants with $\\text{MAF} \\ge 1\\%$.
        
        **3. Zero Target Leakage:**
        Scores variants purely on biophysics and tissue expression without ClinVar training bias.
        """)

    # Pipeline Trigger
    vcf_source_to_run = None
    if btn_run or st.session_state.scored_results is None:
        if uploaded_file is not None:
            vcf_source_to_run = uploaded_file
        elif raw_vcf_text.strip():
            vcf_source_to_run = raw_vcf_text.strip()
        elif use_sample:
            cohort_filename = selected_sample_cohort.split(" ")[0]
            if os.path.exists(f"./{cohort_filename}"):
                vcf_source_to_run = f"./{cohort_filename}"
            elif os.path.exists("./patient_clinical_sample.vcf"):
                vcf_source_to_run = "./patient_clinical_sample.vcf"
            else:
                vcf_source_to_run = "./sample_random.vcf"

        if vcf_source_to_run is not None:
            with st.spinner("Executing Stage 1-6 Multi-Omics Prioritization & ML Calibration..."):
                t_start = time.time()
                scored_df, cal = process_patient_genomics(vcf_source_to_run, engine)
                st.session_state.scored_results = scored_df
                st.session_state.calibrator = cal
                save_diagnostic_report(scored_df, "Patient_Diagnostic_Report.xlsx")
                st.success(f"Prioritization Complete in {time.time()-t_start:.2f}s! Evaluated {len(scored_df)} variants.")

    # Results Display
    if st.session_state.scored_results is not None:
        df = st.session_state.scored_results
        cal = st.session_state.calibrator

        st.markdown("---")
        st.markdown("### Cohort Prioritization Summary")

        # KPI Metrics Row
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        total_v = len(df)
        t1_count = len(df[df["Tier_Code"] == "TIER_1"])
        t2_count = len(df[df["Tier_Code"] == "TIER_2"])
        t3_count = len(df[df["Tier_Code"] == "TIER_3"])
        dropped_count = len(df[df["Tier_Code"] == "DROPPED"])

        kpi1.metric("Total Ingested", total_v)
        kpi2.metric("Tier 1 (Actionable)", t1_count, delta=f"{t1_count/total_v*100:.0f}%" if total_v else None)
        kpi3.metric("Tier 2 (Moderate)", t2_count)
        kpi4.metric("Tier 3 (Neutralized)", t3_count)
        kpi5.metric("Common Noise Dropped", dropped_count, delta="Suppressed" if dropped_count else None)

        st.markdown("### Prioritized Clinical Leaderboard (Sheet 1: Physician View)")
        display_cols = [
            "Variant_ID", "Gene", "Clinical_Tier", "V_Score", "MAF",
            "CADD_PHRED", "Primary_Tissue", "T_c", "D_i", "Clinical_Significance", "Phenotype"
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, height=280)

        # Download Report Button
        report_file = "Patient_Diagnostic_Report.xlsx"
        if os.path.exists(report_file):
            with open(report_file, "rb") as f:
                excel_bytes = f.read()
            st.download_button(
                label="📥 Download Clinical Diagnostic Workbook (Patient_Diagnostic_Report.xlsx)",
                data=excel_bytes,
                file_name="Patient_Diagnostic_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ======================================================================
# SECTION 2: INTERACTIVE VARIANT CALCULATOR (NO SLIDERS)
# ======================================================================
elif nav_choice == "⚡ Interactive Variant Calculator":
    st.markdown('<div class="main-header">Interactive Multi-Omics Variant Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter precise variant parameters using numeric and text controls (zero sliders) for instant mathematical prioritization and tier flagging.</div>', unsafe_allow_html=True)

    # Preset Loader without sliders
    col_pre, col_btn = st.columns([3, 1])
    with col_pre:
        preset_choice = st.selectbox(
            "Load Verified Clinical Preset",
            [
                "Custom Input",
                "MYBPC3 rs72552763 (De Novo Acute Cardiomyopathy)",
                "MYH7 rs11549465 (Sarcomeric Hypertrophic Trigger)",
                "LDLR rs121964858 (Familial Hypercholesterolemia)",
                "HFE rs1800562 (Hemochromatosis - Moderate Cardiac)",
                "HFE rs1800758 (Benign Common Allele - MAF 22%)",
                "ALB Albumin (Liver-Specific Negative Control)",
            ]
        )

    # Preset values
    p_id = "rs72552763"
    p_gene = "MYBPC3"
    p_maf = 0.000010
    p_genotype = "Heterozygous (0/1 or AB)"
    p_cadd = 24.1
    p_pedigree = "De Novo Trio Mutation (2.5x Multiplier)"
    p_heart_tpm = 1396.55
    p_max_tpm = 1396.55

    if "MYBPC3" in preset_choice:
        p_id, p_gene, p_maf, p_cadd = "rs72552763", "MYBPC3", 0.000010, 24.1
        p_heart_tpm, p_max_tpm = 1396.55, 1396.55
        p_pedigree = "De Novo Trio Mutation (2.5x Multiplier)"
    elif "MYH7" in preset_choice:
        p_id, p_gene, p_maf, p_cadd = "rs11549465", "MYH7", 0.000001, 26.8
        p_heart_tpm, p_max_tpm = 1845.20, 1845.20
        p_pedigree = "De Novo Trio Mutation (2.5x Multiplier)"
    elif "LDLR" in preset_choice:
        p_id, p_gene, p_maf, p_cadd = "rs121964858", "LDLR", 0.000010, 18.2
        p_heart_tpm, p_max_tpm = 12.908, 154.717
        p_pedigree = "Inherited Carrier (1.0x Standard)"
    elif "rs1800562" in preset_choice:
        p_id, p_gene, p_maf, p_cadd = "rs1800562", "HFE", 0.000001, 28.5
        p_heart_tpm, p_max_tpm = 2.099, 14.473
        p_pedigree = "Inherited Carrier (1.0x Standard)"
    elif "rs1800758" in preset_choice:
        p_id, p_gene, p_maf, p_cadd = "rs1800758", "HFE", 0.220000, 0.8
        p_heart_tpm, p_max_tpm = 2.099, 14.473
        p_pedigree = "Inherited Carrier (1.0x Standard)"
    elif "ALB" in preset_choice:
        p_id, p_gene, p_maf, p_cadd = "ALB:Syn:01", "ALB", 0.000010, 20.0
        p_heart_tpm, p_max_tpm = 0.12, 12450.0
        p_pedigree = "Inherited Carrier (1.0x Standard)"

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.markdown("#### 1. Identification & Population Rarity")
        calc_id = st.text_input("Variant ID (dbSNP rsID or Coordinate)", value=p_id)
        calc_gene = st.text_input("Canonical Gene Symbol", value=p_gene).strip().upper()

        # Minor Allele Frequency: NUMERIC INPUT (NO SLIDERS)
        calc_maf = st.number_input(
            "Minor Allele Frequency (MAF / allele_B_freq)",
            min_value=0.000000,
            max_value=1.000000,
            value=float(p_maf),
            step=0.000001,
            format="%.6f",
            help="Public population frequency from JSNP / gnomAD."
        )

        calc_genotype = st.selectbox(
            "Patient Diploid Genotype",
            [
                "Heterozygous (0/1 or AB)",
                "Homozygous Variant (1/1 or BB) [Double Mutant Penalty]",
                "Homozygous Reference (0/0 or AA)",
            ],
            index=0 if "Heterozygous" in p_genotype else 1
        )

    with col_in2:
        st.markdown("#### 2. Functional Damage & Tissue Expression")
        # CADD PHRED: NUMERIC INPUT (NO SLIDERS)
        calc_cadd = st.number_input(
            "CADD v1.6 PHRED Structural Damage Score",
            min_value=0.0,
            max_value=99.0,
            value=float(p_cadd),
            step=0.1,
            format="%.1f",
            help="Genome-wide in-silico structural deleteriousness score."
        )

        calc_pedigree = st.radio(
            "Family Trio Pedigree Transmission",
            [
                "De Novo Trio Mutation (2.5x Multiplier)",
                "Inherited Carrier (1.0x Standard)",
            ],
            index=0 if "De Novo" in p_pedigree else 1
        )

        # Dynamic GTEx Query
        gtex_info = engine.get_gene_expression(calc_gene)
        st.caption(f"GTEx v11 Database: Peak Organ = **{gtex_info['Primary_Tissue']}**")

        c_tpm1, c_tpm2 = st.columns(2)
        with c_tpm1:
            calc_heart_tpm = st.number_input(
                "Heart Peak TPM",
                min_value=0.0,
                max_value=50000.0,
                value=float(gtex_info["TPM_Heart"] if gtex_info["TPM_Heart"] > 0 else p_heart_tpm),
                step=0.1,
                format="%.2f"
            )
        with c_tpm2:
            calc_max_tpm = st.number_input(
                "Max Body Organ TPM",
                min_value=0.1,
                max_value=50000.0,
                value=float(gtex_info["TPM_Max"] if gtex_info["TPM_Max"] > 1.0 else p_max_tpm),
                step=0.1,
                format="%.2f"
            )

    # Calculation logic
    st.markdown("---")
    st.markdown("### Clinical Prioritization Output & Tier Flagging")

    # Get active calibrator weights or default
    cal = st.session_state.calibrator
    w1, w2, w3, w4 = cal.weights if cal else (0.25, 0.25, 0.25, 0.25)
    t1_thresh = cal.tier1_threshold if cal else 4.0
    t2_thresh = cal.tier2_threshold if cal else 1.0

    is_noise = calc_maf >= 0.01
    q = calc_maf
    p = max(0.0, 1.0 - q)

    gt_code = "AB" if "Heterozygous" in calc_genotype else ("BB" if "Homozygous Variant" in calc_genotype else "AA")
    is_dn = "De Novo" in calc_pedigree

    r_i = engine.compute_public_rarity(calc_maf)
    s_grs = engine.compute_genotype_rarity(gt_code, p, q)
    chi2 = 0.0
    f_b = engine.compute_functional_damage(calc_cadd)
    t_c = engine.compute_cardiac_tissue_gate(calc_heart_tpm, calc_max_tpm)
    d_i = engine.compute_pedigree_multiplier(is_dn)

    base_sum = (w1 * r_i) + (w2 * s_grs) + (w3 * chi2) + (w4 * f_b)
    v_score = round(base_sum * t_c * d_i, 2)

    if is_noise:
        st.markdown(f"""
        <div class="tier-card-dropped">
            <h2 style="color: #94A3B8; margin: 0;">DROPPED: COMMON POPULATION NOISE</h2>
            <p style="font-size: 14px; margin-top: 6px;">
                <strong>Gatekeeper Triggered:</strong> MAF = {calc_maf:.6f} &ge; 0.01 (1%). This mutation is too frequent in healthy humans to cause early-onset monogenic heart attacks. Computated V_Score = <strong>0.00</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif v_score >= t1_thresh:
        st.markdown(f"""
        <div class="tier-card-1">
            <span style="background: #FF2A6D; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">TIER 1 ACTIONABLE TRIGGER</span>
            <h1 style="color: #FFFFFF; margin: 8px 0 4px;">Composite V_Score: {v_score:.2f}</h1>
            <p style="font-size: 14px; color: #F1F5F9; line-height: 1.5;">
                <strong>Clinical Action Protocol:</strong> Acute high-impact mutation in primary cardiac structure (<strong>{calc_gene}</strong>) with peak left ventricle expression ($T_c = {t_c:.3f}$) and severe biophysical damage. Immediate cardiology review, echocardiography, and family cascade pedigree screening indicated.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif v_score >= t2_thresh:
        st.markdown(f"""
        <div class="tier-card-2">
            <span style="background: #FFB300; color: black; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">TIER 2 MODERATE RISK CANDIDATE</span>
            <h1 style="color: #FFFFFF; margin: 8px 0 4px;">Composite V_Score: {v_score:.2f}</h1>
            <p style="font-size: 14px; color: #F1F5F9; line-height: 1.5;">
                <strong>Clinical Action Protocol:</strong> Significant multi-omics damage in cardiovascular-relevant gene (<strong>{calc_gene}</strong>). Potential recessive carrier or secondary risk driver; continuous lipid/cardiac monitoring indicated.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="tier-card-3">
            <span style="background: #64748B; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">TIER 3 LOW IMPACT / NEUTRALIZED</span>
            <h1 style="color: #FFFFFF; margin: 8px 0 4px;">Composite V_Score: {v_score:.2f}</h1>
            <p style="font-size: 14px; color: #F1F5F9; line-height: 1.5;">
                <strong>Clinical Action Protocol:</strong> Score neutralized due to low cardiac tissue expression (<strong>{calc_gene}</strong> has off-target expression in {gtex_info['Primary_Tissue']}, $T_c = {t_c:.3f}$) or mild computational impact. Excluded from primary coronary intervention.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 4-Pillar Derivation Breakdown
    st.markdown("#### Mathematical Pillar Derivation (Using ML Weights):")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric(f"Pillar 1: R_i ({w1*100:.0f}%)", f"{r_i:.3f}", help="Public Allele Rarity: -log10(MAF + 1e-6)")
    p2.metric(f"Pillar 2: S_GRS ({w2*100:.0f}%)", f"{s_grs:.3f}", help="Genotype Severity: -log10(P(genotype) + 1e-6)")
    p3.metric(f"Pillar 3: chi2 ({w3*100:.0f}%)", f"{chi2:.3f}", help="Cohort Departure Test")
    p4.metric(f"Pillar 4: F_B ({w4*100:.0f}%)", f"{f_b:.3f}", help="Protein Damage: ln(1 + PHRED)")
    p5.metric("Cardiac Gate (T_c)", f"{t_c:.3f}", help="Heart TPM / (Max Body TPM + 0.1)")
    p6.metric("Pedigree Multiplier", f"{d_i:.1f}x", help="2.5x if De Novo Trio else 1.0x")


# ======================================================================
# SECTION 3: ML WEIGHT & THRESHOLD CALIBRATION
# ======================================================================
elif nav_choice == "🤖 ML Weight & Threshold Calibration":
    st.markdown('<div class="main-header">Machine Learning Weight & Dynamic Threshold Calibration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-driven optimization replaces arbitrary static parameters with trained biological weights and ROC-derived decision thresholds.</div>', unsafe_allow_html=True)

    cal = st.session_state.calibrator
    if cal is None:
        st.warning("Please execute the prioritization pipeline in Tab 1 first to calibrate the ML weights on your cohort.")
    else:
        col_w, col_t = st.columns(2)

        with col_w:
            st.markdown("### 1. Learned Pillar Feature Weights ($\mathbf{w}^*$)")
            st.caption("Derived via constrained optimization on ClinVar ground truth:")

            weights_df = pd.DataFrame({
                "Biological Evidence Pillar": [
                    "Pillar 1: Public Allele Rarity (R_i)",
                    "Pillar 2: Genotype Severity (S_GRS)",
                    "Pillar 3: Cohort Distortion (chi2)",
                    "Pillar 4: Protein Physical Damage (F_B)"
                ],
                "Static Baseline Weight": [0.25, 0.25, 0.25, 0.25],
                "ML Optimized Weight": cal.weights,
                "Relative Influence": [f"{w*100:.1f}%" for w in cal.weights]
            })
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

            # Chart comparison
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#0E1322')
            ax.set_facecolor('#0E1322')
            x = np.arange(4)
            ax.bar(x - 0.2, [0.25, 0.25, 0.25, 0.25], 0.35, label='Static Baseline (25%)', color='#64748B')
            ax.bar(x + 0.2, cal.weights, 0.35, label='ML Calibrated Weight', color='#00F0FF')
            ax.set_xticks(x)
            ax.set_xticklabels(['R_i (Rarity)', 'S_GRS (Genotype)', 'chi2 (HWE)', 'F_B (Damage)'], color='#94A3B8')
            ax.tick_params(colors='#94A3B8')
            ax.legend(facecolor='#1E293B', labelcolor='#FFFFFF')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            st.pyplot(fig)

        with col_t:
            st.markdown("### 2. Data-Driven Dynamic Thresholds")
            st.caption("Derived empirically from Youden's J-statistic on the variant score distribution:")

            st.metric("Dynamic Tier 1 Cutoff (θ₁)", f"V_Score ≥ {cal.tier1_threshold:.2f}", help="Separates high-confidence actionable triggers")
            st.metric("Dynamic Tier 2 Cutoff (θ₂)", f"V_Score ≥ {cal.tier2_threshold:.2f}", help="Separates severe inherited carriers from benign background")

            st.markdown("#### Model Performance Metrics (Zero Target Leakage)")
            st.write(f"• **Optimized AUROC**: `{cal.calibration_metrics.get('Optimized_AUROC', 'N/A')}`")
            st.write(f"• **Optimized PR-AUC**: `{cal.calibration_metrics.get('Optimized_PRAUC', 'N/A')}`")
            st.write(f"• **Ground-Truth Sample Size**: `{cal.calibration_metrics.get('Sample_Size', 'N/A')} variants`")


# ======================================================================
# SECTION 4: GTEx v11 CARDIAC EXPLORER
# ======================================================================
elif nav_choice == "🫀 GTEx v11 Cardiac Explorer":
    st.markdown('<div class="main-header">GTEx v11 RNA-Seq Multi-Tissue Cardiac Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Direct expression quantification across 73,321 human genes to verify tissue specificity.</div>', unsafe_allow_html=True)

    search_gene = st.text_input("Query Any Gene Symbol in GTEx v11", value="MYBPC3").strip().upper()
    prof = engine.get_gene_expression(search_gene)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gene Analyzed", search_gene)
    m2.metric("Heart Peak TPM", f"{prof['TPM_Heart']:.2f}")
    m3.metric("Body Max TPM", f"{prof['TPM_Max']:.2f}", help=f"Peak organ: {prof['Primary_Tissue']}")
    m4.metric("Cardiac Gate (T_c)", f"{prof['T_c']:.3f}")

    st.markdown(f"**Peak Expression Organ:** `{prof['Primary_Tissue']}`")

    # Tissue breakdown bar chart
    tissues = ["Heart_Left_Ventricle", "Heart_Atrial_Appendage", "Artery_Coronary", "Body_Peak_Max"]
    tpm_vals = [prof["TPM_Ventricle"], prof["TPM_Atrium"], prof["TPM_Coronary"], prof["TPM_Max"]]

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0E1322')
    ax.set_facecolor('#0E1322')
    bars = ax.bar(tissues, tpm_vals, color=['#FF2A6D', '#FFB300', '#00E676', '#00F0FF'], width=0.5)
    ax.set_ylabel("Median RNA-Seq TPM", color='#94A3B8')
    ax.tick_params(colors='#94A3B8')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='#FFFFFF', fontsize=9)
    st.pyplot(fig)


# ======================================================================
# SECTION 5: CLINICAL LEADERBOARD & REPORTS
# ======================================================================
elif nav_choice == "📊 Clinical Leaderboard & Reports":
    st.markdown('<div class="main-header">Clinical Leaderboard & Trace Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection views matching Sheet 1 & Sheet 2 of the Patient Diagnostic Workbook.</div>', unsafe_allow_html=True)

    df = st.session_state.scored_results
    if df is None:
        st.warning("Please execute the pipeline in Tab 1 first.")
    else:
        view_tab = st.radio("Select Inspection View", ["Sheet 1: Clinical Leaderboard (Physician View)", "Sheet 2: Extraction Trace Matrix (25+ Raw Columns)"], horizontal=True)

        if "Sheet 1" in view_tab:
            st.dataframe(df[[
                "Variant_ID", "Gene", "Clinical_Tier", "V_Score", "MAF", "CADD_PHRED",
                "Primary_Tissue", "T_c", "D_i", "Clinical_Significance", "Phenotype"
            ]], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

        report_file = "Patient_Diagnostic_Report.xlsx"
        if os.path.exists(report_file):
            with open(report_file, "rb") as f:
                excel_bytes = f.read()
            st.download_button(
                label="📥 Download Diagnostic Workbook (.xlsx)",
                data=excel_bytes,
                file_name="Patient_Diagnostic_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
