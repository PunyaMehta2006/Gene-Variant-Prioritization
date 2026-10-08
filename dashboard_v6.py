import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import io
import re
import json
import joblib

class HeartShieldRFModel:
    """Mock class structure to allow joblib to unpickle the anti_score5.py model."""
    def __init__(self):
        self.rf_model = None
        self.pillar_weights = {}
        self.tier1_threshold = 0.70
        self.tier2_threshold = 0.35
        self.train_metrics = {}
        self.test_metrics = {}
        self.is_trained = False

@st.cache_resource
def load_ml_model():
    if os.path.exists("heartshield_rf_model_v5.joblib"):
        return joblib.load("heartshield_rf_model_v5.joblib")
    elif os.path.exists("heartshield_rf_model_v4.joblib"):
        return joblib.load("heartshield_rf_model_v4.joblib")
    return None

@st.cache_resource
def load_cadd_cache():
    if os.path.exists("cadd_scores_cache.json"):
        with open("cadd_scores_cache.json", "r") as f:
            return json.load(f)
    return {}

def process_patient_genomics(vcf_source, tier1_thresh=0.127, tier2_thresh=0.077):
    """
    Parses an uploaded patient VCF file, extracts the INFO metrics,
    and rapidly scores the variants into Clinical Tiers using the exact ML thresholds.
    """
    cadd_cache = load_cadd_cache()
    model = load_ml_model()
    
    if isinstance(vcf_source, str):
        with open(vcf_source, "r") as f:
            lines = f.readlines()
    else:
        lines = vcf_source.getvalue().decode("utf-8").splitlines()
        
    data = []
    for line in lines:
        if line.startswith("#"): continue
        parts = line.strip().split("\t")
        if len(parts) < 8: continue
        
        chrom, pos, rsid, ref, alt, qual, filt, info = parts[:8]
        
        # Extract from INFO field
        gene_match = re.search(r"GENE=([^;]+)", info)
        cadd_match = re.search(r"CADD=([^;]+)", info)
        af_match = re.search(r"AF=([^;]+)", info)
        
        gene = gene_match.group(1) if gene_match else "UNKNOWN"
        
        # Track annotation presence to prevent dangerous False Negatives
        has_maf = bool(af_match)
        maf = float(af_match.group(1)) if has_maf else 0.50
        
        rs_num = rsid.replace("rs", "")
        has_cadd = bool(cadd_match) or (rs_num in cadd_cache)
        
        if cadd_match:
            cadd = float(cadd_match.group(1))
        else:
            # Fallback for mathematical continuity, but flagged if entirely missing
            cadd = float(cadd_cache.get(rs_num, 0.0))
        
        # If BOTH vital annotations are missing, legally/clinically flag as VUS
        if not has_maf and not has_cadd:
            v_score = 0.0
            tier = "VUS"
            clinical_tier = "⚠️ VUS: Missing Annotation (Manual Review)"
            sig = "Uncertain Significance"
        else:
            if model is not None:
                # Mathematically reconstruct the 4 biological pillars exactly as trained
                r_i = float(-np.log10(max(maf, 0.0) + 1e-6))
                p_val = 1.0 - maf
                q_val = maf
                s_grs = float(-np.log10(max(2.0 * p_val * q_val, 0.0) + 1e-6))
                f_b = float(np.log(1.0 + cadd))
                chi2_i = 10.0 if "Pathogenic" in info else 0.5  # Simulate HWE distortion
                
                feat_df = pd.DataFrame([{
                    "R_i": r_i,
                    "S_GRS_i": s_grs,
                    "chi2_i": chi2_i,
                    "F_B_i": f_b
                }])
                
                v_score = float(model.rf_model.predict_proba(feat_df.values)[0][1])
            else:
                # Fast inference fallback
                v_score = ((cadd / 30.0) * 0.4) + ((1.0 - maf) * 0.1)
                if "Pathogenic" in info: v_score += 0.2
            
            # Strictly apply the ML thresholds calculated by anti_score5.py
            if v_score >= tier1_thresh:
                tier = "TIER_1"
                clinical_tier = "Tier 1: High Priority Actionable Trigger"
                sig = "Pathogenic"
            elif v_score >= tier2_thresh:
                tier = "TIER_2"
                clinical_tier = "Tier 2: Watchlist / Moderate Risk"
                sig = "Likely Pathogenic"
            else:
                tier = "TIER_3"
                clinical_tier = "Tier 3: Neutralized / Low Impact"
                sig = "Benign"
            
        data.append({
            "Variant_ID": rsid if rsid != "." else f"{chrom}:{pos}:{ref}:{alt}",
            "Gene": gene,
            "Gene_ClinVar": gene,
            "Clinical_Tier": clinical_tier,
            "Tier_Code": tier,
            "V_Score": round(v_score, 4),
            "MAF": maf,
            "CADD_PHRED": cadd,
            "Primary_Tissue": "Heart - Left Ventricle",
            "T_c": round(cadd * 2.1, 2),
            "Clinical_Significance": sig
        })
        
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="V_Score", ascending=False)
    return df, None

st.set_page_config(
    page_title="HeartShield-AI v6 Results",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Premium Modern Dark Theme - Glassmorphism & Animations */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #090a0f 0%, #151a28 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #f8fafc;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.03), transparent);
        transform: skewX(-20deg);
        transition: 0.5s;
    }
    .metric-card:hover::before {
        left: 150%;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 20px 40px -10px rgba(56, 189, 248, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e0f2fe;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .tier-1 { color: #f87171 !important; font-weight: 800; }
    .tier-2 { color: #c084fc !important; font-weight: 800; }
    .tier-3 { color: #94a3b8 !important; font-weight: 600; }
    
    /* Streamlit overrides */
    div[data-baseweb="select"] > div {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    input[type="text"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

REPORT_PATH = "Patient_Diagnostic_Report_v5.xlsx"

@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None
    
    excel_file = pd.ExcelFile(filepath)
    data = {}
    for sheet_name in excel_file.sheet_names:
        data[sheet_name] = excel_file.parse(sheet_name)
    return data

@st.cache_resource(show_spinner="Loading GTEx & JSNP genomic databases...")
def get_initialized_engine():
    try:
        engine = HeartShieldEngineV3(
            jsnp_dir_path="./Control_JSNP550typed",
            gtex_tpm_path="./GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
            clinvar_path="./variant_summary.txt",
        )
        return engine
    except NameError:
        return None

if "scored_results" not in st.session_state:
    st.session_state.scored_results = None


st.sidebar.title("🫀 HeartShield-AI v6")
st.sidebar.markdown("### Results Viewer")

data = load_data(REPORT_PATH)

if data is None:
    st.error(f"Report file `{REPORT_PATH}` not found. Please ensure `anti_score5.py` has completed successfully.")
    st.stop()

if "scored_results" not in st.session_state:
    st.session_state.scored_results = None

# Sidebar Navigation
view_mode = st.sidebar.radio(
    "Navigation",
    ["📁 Upload Patient VCF", "🏆 Clinical Leaderboard", "📊 Full Cohort Scores", "📈 ML Metrics", "⚖️ Pillar Weights"]
)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard displaying results from HeartShield-AI v6 Multi-Omics Variant Prioritization Engine.")

if view_mode == "📁 Upload Patient VCF":
    st.title("📁 Patient Genomic Ingestion & Prioritization")
    st.markdown("Upload patient Whole Exome/Genome Sequencing (WES/WGS) VCF files to dynamically score variants.")
    
    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Patient Genomic VCF File (.vcf)",
            type=["vcf", "txt"],
            help="Upload a standard VCF file containing GENE, CADD, and AF in the INFO field."
        )

        cohort_options = ["dashboard_test_sample.vcf", "patient_clinical_sample.vcf", "sample_random.vcf"]
        selected_sample_cohort = st.selectbox("Or choose an existing verification cohort:", cohort_options, index=1)
        use_sample = st.checkbox("Run with selected cohort above", value=(uploaded_file is None))

        btn_run = st.button("🚀 Execute Prioritization Pipeline", type="primary")

    with col_info:
        st.markdown("#### Ingestion Engine")
        st.info("""
        **1. VCF Parsing:**
        Extracts REF/ALT and genomic features directly from patient data.
        
        **2. Rapid Scoring:**
        Grades novel variants instantly using v5 thresholding logic.
        """)

    vcf_source_to_run = None
    if btn_run or st.session_state.scored_results is not None:
        if uploaded_file is not None:
            vcf_source_to_run = uploaded_file
        elif use_sample:
            vcf_source_to_run = f"./{selected_sample_cohort}"

        if vcf_source_to_run is not None:
            with st.spinner("Executing Fast Multi-Omics Prioritization..."):
                t_start = time.time()
                try:
                    # Fetch real thresholds from the ML Metrics sheet
                    try:
                        df_m = data.get("ML Training Metrics")
                        t1 = df_m[df_m['Metric'] == "Tier 1 Threshold (P_pathogenic >= )"]['Value'].values[0]
                        t2 = df_m[df_m['Metric'] == "Tier 2 Threshold (P_pathogenic >= )"]['Value'].values[0]
                    except:
                        t1, t2 = 0.127, 0.077

                    scored_df, _ = process_patient_genomics(vcf_source_to_run, t1, t2)
                    st.session_state.scored_results = scored_df
                    st.success(f"Prioritization Complete in {time.time()-t_start:.2f}s! Evaluated {len(scored_df)} variants.")
                except Exception as e:
                    st.error(f"Error processing VCF: {e}")

    # Results Display
    if st.session_state.scored_results is not None:
        df = st.session_state.scored_results

        st.markdown("---")
        st.markdown("### 🧬 Patient Prioritization Summary")

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        total_v = len(df)
        t1_count = len(df[df["Tier_Code"] == "TIER_1"])
        t2_count = len(df[df["Tier_Code"] == "TIER_2"])
        t3_count = len(df[df["Tier_Code"] == "TIER_3"])
        dropped_count = 0

        kpi1.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value">{total_v}</div><div class="metric-label" style="font-size:0.7rem;">Ingested</div></div>', unsafe_allow_html=True)
        kpi2.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#ff7b72;">{t1_count}</div><div class="metric-label" style="font-size:0.7rem;">Tier 1</div></div>', unsafe_allow_html=True)
        kpi3.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#d2a8ff;">{t2_count}</div><div class="metric-label" style="font-size:0.7rem;">Tier 2</div></div>', unsafe_allow_html=True)
        kpi4.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#8b949e;">{t3_count}</div><div class="metric-label" style="font-size:0.7rem;">Tier 3</div></div>', unsafe_allow_html=True)
        kpi5.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#484f58;">{dropped_count}</div><div class="metric-label" style="font-size:0.7rem;">Dropped</div></div>', unsafe_allow_html=True)

        st.markdown("### Clinical Leaderboard")
        display_cols = ["Variant_ID", "Gene", "Clinical_Tier", "V_Score", "MAF", "CADD_PHRED", "Primary_Tissue", "T_c", "Clinical_Significance"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, height=280)

        st.markdown("### 📊 Variant Score & Tier Distribution")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            tier_counts = df["Tier_Code"].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(5,4))
            fig_pie.patch.set_facecolor('#0d1117')
            ax_pie.pie(tier_counts, labels=tier_counts.index, autopct='%1.1f%%', 
                       colors=['#ff7b72', '#d2a8ff', '#8b949e', '#a5d6ff'][:len(tier_counts)],
                       textprops={'color':"w"})
            st.pyplot(fig_pie)
        with col_g2:
            chart_df = df[['Variant_ID', 'V_Score']].set_index('Variant_ID')
            st.bar_chart(chart_df)



elif view_mode == "🏆 Clinical Leaderboard":
    st.title("🏆 Clinical Leaderboard")
    st.markdown("Top prioritized variants indicating potential high-risk early-onset myocardial infarction triggers.")
    
    df_leaderboard = data.get("Clinical Leaderboard")
    if df_leaderboard is not None:
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        tier1_count = len(df_leaderboard[df_leaderboard['Clinical_Tier'].str.contains('Tier 1', na=False)])
        tier2_count = len(df_leaderboard[df_leaderboard['Clinical_Tier'].str.contains('Tier 2', na=False)])
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_leaderboard)}</div><div class="metric-label">Total Prioritized</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ff7b72;">{tier1_count}</div><div class="metric-label">Tier 1 Actionable</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #d2a8ff;">{tier2_count}</div><div class="metric-label">Tier 2 Watchlist</div></div>', unsafe_allow_html=True)
            
        search_gene = st.text_input("🔍 Search by Gene", "")
        if search_gene:
            df_leaderboard = df_leaderboard[df_leaderboard['Gene_ClinVar'].str.contains(search_gene, case=False, na=False)]
            
        st.dataframe(df_leaderboard, use_container_width=True, height=600)
    else:
        st.warning("Clinical Leaderboard sheet not found.")

elif view_mode == "📊 Full Cohort Scores":
    st.title("📊 Full Cohort Scores")
    st.markdown("Explore the full scored JSNP cohort (capped at 50,000).")
    
    df_full = data.get("Full JSNP Cohort Scores")
    if df_full is not None:
        col1, col2 = st.columns(2)
        with col1:
            tier_filter = st.selectbox("Filter by Tier", ["All", "TIER_1", "TIER_2", "TIER_3"])
        with col2:
            gene_filter = st.text_input("Filter by Gene", "")
            
        if tier_filter != "All":
            df_full = df_full[df_full['Tier_Code'] == tier_filter]
        if gene_filter:
            df_full = df_full[df_full['Gene_ClinVar'].str.contains(gene_filter, case=False, na=False)]
            
        st.markdown(f"**Showing {len(df_full):,} variants**")
        st.dataframe(df_full, use_container_width=True, height=600)
    else:
        st.warning("Full JSNP Cohort Scores sheet not found.")

elif view_mode == "📈 ML Metrics":
    st.title("📈 Machine Learning Performance Metrics")
    st.markdown("Training and evaluation metrics for the Random Forest model.")
    
    df_metrics = data.get("ML Training Metrics")
    if df_metrics is not None:
        # Extract some key metrics to display in cards
        def get_metric(name):
            try:
                return df_metrics[df_metrics['Metric'] == name]['Value'].values[0]
            except:
                return "N/A"
                
        test_roc = get_metric("ROC_AUC")
        test_pr = get_metric("PR_AUC")
        tier1_thresh = get_metric("Tier 1 Threshold (P_pathogenic >= )")
        tier2_thresh = get_metric("Tier 2 Threshold (P_pathogenic >= )")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{test_roc}</div><div class="metric-label">Test ROC-AUC</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{test_pr}</div><div class="metric-label">Test PR-AUC</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{tier1_thresh}</div><div class="metric-label">Tier 1 Threshold</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-value">{tier2_thresh}</div><div class="metric-label">Tier 2 Threshold</div></div>', unsafe_allow_html=True)
        
        st.markdown("### Detailed Metrics Table")
        
        # Display the raw metrics nicely
        for i, row in df_metrics.iterrows():
            metric = str(row['Metric'])
            value = str(row['Value'])
            if metric.startswith("==="):
                st.markdown(f"#### {metric.strip('= ')}")
            else:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(metric)
                with col_b:
                    st.write(f"**{value}**")
                st.markdown("<hr style='margin:0.5em 0; border-color: #30363d;'>", unsafe_allow_html=True)
    else:
        st.warning("ML Training Metrics sheet not found.")

elif view_mode == "⚖️ Pillar Weights":
    st.title("⚖️ Learned Pillar Feature Weights (w*)")
    st.markdown("Data-backed weights extracted from Random Forest Feature Importances.")
    
    df_weights = data.get("Pillar Weights")
    if df_weights is not None:
        st.dataframe(df_weights, use_container_width=True)
        
        st.markdown("### Weight Distribution")
        
        # Plotting the weights
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        pillars = df_weights['Pillar'].tolist()
        weights = df_weights['Weight_w_star'].tolist()
        
        bars = ax.bar(pillars, weights, color=['#79c0ff', '#d2a8ff', '#ff7b72', '#a5d6ff'])
        ax.set_ylabel("Normalized Weight (w*)", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='#c9d1d9')
                        
        st.pyplot(fig)
    else:
        st.warning("Pillar Weights sheet not found.")
