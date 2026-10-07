import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

try:
    from anti_score3 import HeartShieldEngineV3, process_patient_genomics, save_diagnostic_report
except ImportError:
    st.warning("anti_score3 module not found. VCF ingestion will be disabled.")


st.set_page_config(
    page_title="HeartShield-AI v4 Results",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Custom Modern Dark Theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    .metric-card {
        background: rgba(48, 54, 61, 0.5);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #79c0ff;
    }
    .metric-label {
        font-size: 1rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .tier-1 { color: #ff7b72; font-weight: bold; }
    .tier-2 { color: #d2a8ff; font-weight: bold; }
    .tier-3 { color: #8b949e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

REPORT_PATH = "Patient_Diagnostic_Report_v4.xlsx"

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


st.sidebar.title("🫀 HeartShield-AI v4")
st.sidebar.markdown("### Results Viewer")

data = load_data(REPORT_PATH)

if data is None:
    st.error(f"Report file `{REPORT_PATH}` not found. Please ensure `anti_score4.py` has completed successfully.")
    st.stop()

# Sidebar Navigation
view_mode = st.sidebar.radio(
    "Navigation",
    ["📁 Upload Patient VCF", "🏆 Clinical Leaderboard", "📊 Full Cohort Scores", "📈 ML Metrics", "⚖️ Pillar Weights"]
)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard displaying results from HeartShield-AI v4 Multi-Omics Variant Prioritization Engine.")

if view_mode == "📁 Upload Patient VCF":
    st.title("📁 Patient Genomic Ingestion & Prioritization")
    st.markdown("Upload patient Whole Exome/Genome Sequencing (WES/WGS) VCF files to score individual clinical variants.")
    
    engine = get_initialized_engine()
    if engine is None:
        st.error("Engine failed to load. Please ensure `anti_score3.py` is present.")
    else:
        col_up, col_info = st.columns([2, 1])

        with col_up:
            uploaded_file = st.file_uploader(
                "Upload Patient Genomic VCF File (.vcf)",
                type=["vcf", "txt"],
                help="Upload a standard VCFv4.2 file containing coordinates, REF, ALT, and optional INFO annotations."
            )

            cohort_options = ["dashboard_test_sample.vcf", "patient_clinical_sample.vcf", "sample_random.vcf"]
            selected_sample_cohort = st.selectbox("Or choose an existing verification cohort:", cohort_options, index=0)
            use_sample = st.checkbox("Run with selected cohort above", value=(uploaded_file is None))

            btn_run = st.button("🚀 Execute Prioritization Pipeline", type="primary")

        with col_info:
            st.markdown("#### Ingestion Gatekeepers")
            st.info("""
            **1. GRCh38 Gatekeeper:**
            Harmonizes variant coordinates to primary assembly.
            
            **2. Public Noise Gate:**
            Filters common benign variants with MAF ≥ 1%.
            """)

        # Pipeline Trigger
        vcf_source_to_run = None
        if btn_run or st.session_state.scored_results is not None:
            if uploaded_file is not None:
                vcf_source_to_run = uploaded_file
            elif use_sample:
                vcf_source_to_run = f"./{selected_sample_cohort}"

            if vcf_source_to_run is not None:
                with st.spinner("Executing Multi-Omics Prioritization..."):
                    t_start = time.time()
                    try:
                        scored_df, cal = process_patient_genomics(vcf_source_to_run, engine)
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
            dropped_count = len(df[df["Tier_Code"] == "DROPPED"])

            kpi1.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value">{total_v}</div><div class="metric-label" style="font-size:0.7rem;">Ingested</div></div>', unsafe_allow_html=True)
            kpi2.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#ff7b72;">{t1_count}</div><div class="metric-label" style="font-size:0.7rem;">Tier 1</div></div>', unsafe_allow_html=True)
            kpi3.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#d2a8ff;">{t2_count}</div><div class="metric-label" style="font-size:0.7rem;">Tier 2</div></div>', unsafe_allow_html=True)
            kpi4.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#8b949e;">{t3_count}</div><div class="metric-label" style="font-size:0.7rem;">Tier 3</div></div>', unsafe_allow_html=True)
            kpi5.markdown(f'<div class="metric-card" style="padding:10px;"><div class="metric-value" style="color:#484f58;">{dropped_count}</div><div class="metric-label" style="font-size:0.7rem;">Dropped</div></div>', unsafe_allow_html=True)

            st.markdown("### Clinical Leaderboard")
            display_cols = ["Variant_ID", "Gene", "Clinical_Tier", "V_Score", "MAF", "CADD_PHRED", "Primary_Tissue", "T_c", "Clinical_Significance"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols], use_container_width=True, height=280)

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
