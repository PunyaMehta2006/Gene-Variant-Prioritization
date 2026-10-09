# HeartShield-AI v6 🫀

**Multi-Omics Cardiac Variant Prioritization Engine**

HeartShield-AI is a premium, clinical-grade bioinformatics pipeline and diagnostic dashboard designed to dynamically ingest, score, and prioritize human genomic mutations based on their risk of triggering Myocardial Infarction (MI).

By integrating population rarity, structural damage, and tissue-specific gene expression into a single Random Forest machine learning architecture, HeartShield-AI systematically separates deadly actionable triggers (Tier 1) from harmless biological noise (Tier 3).

---

## 🧬 Core Architecture

The system strictly adheres to an elegant two-file architecture:

### 1. The Backend Engine (`anti_score5.py`)
- **Data Ingestion**: Parses over 515,000 genomic variants from the JSNP Japanese Population cohort.
- **Dynamic API Caching**: Automatically queries the MyVariant.info API in batches of 500 to pull CADD (Combined Annotation Dependent Depletion) structural damage scores, utilizing a local `.json` cache to completely bypass network latency on future runs.
- **Biological Pillar Math**: Mathematically distills variant data into 4 core Machine Learning features:
  - `R_i` (Public Rarity based on Minor Allele Frequency)
  - `S_GRS` (Genotype Rarity using Hardy-Weinberg Equilibrium)
  - `chi2_i` (HWE Distortion)
  - `F_B_i` (Structural Damage via CADD PHRED)
- **Machine Learning**: Trains a Scikit-Learn `RandomForestClassifier` on ClinVar ground-truth data, utilizing SMOTE to balance pathogenic/benign labels.
- **Output**: Saves the highly-optimized `heartshield_rf_model_v5.joblib` binary model and an automated clinical Excel report.

### 2. The Clinical Dashboard (`dashboard_v6.py`)
- **UI/UX**: Built entirely in Streamlit with custom CSS to achieve a premium, medical-grade "Glassmorphism" aesthetic with micro-animations.
- **VCF Ingestion**: Allows clinicians to upload raw `.vcf` (Variant Call Format) files directly from a sequencing machine.
- **Smart Fallbacks**: Actively tracks missing `MAF` and `CADD` annotations. If both are missing, it surgically intercepts the variant and flags it as a `VUS` (Variant of Uncertain Significance) to prevent lethal False Negatives.
- **Live Scoring**: Uses the 70MB Random Forest `.joblib` model to mathematically score novel VCF variants on the fly, outputting a dynamic "Clinical Leaderboard".

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9+
- Windows/Linux/MacOS

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn streamlit joblib requests
```

### 2. Train the Model
If you want to train the Random Forest from scratch using your local JSNP dataset:
```bash
python anti_score5.py
```
*(Note: Requires the JSNP and GTEx `.txt`/`.gct` files in the root directory).*

### 3. Launch the Dashboard
To start the clinical inference UI:
```bash
streamlit run dashboard_v6.py
```

---

## 📊 The 3-Tier Clinical System

HeartShield-AI classifies variants into three actionable tiers based on Youden's J-statistic cutoffs:

- 🟥 **Tier 1 (High Priority Actionable Trigger):** Statistically rare, highly damaging variants that strongly overlap with ClinVar pathogenic markers. 
- 🟨 **Tier 2 (Watchlist / Moderate Risk):** Variants exhibiting moderate structural damage or elevated rarity requiring continued clinical observation.
- 🟩 **Tier 3 (Neutralized / Low Impact):** Common population noise with low structural damage. 
- ⚠️ **VUS (Manual Review Required):** Variants completely lacking external annotation data (MAF/CADD), bypassing the ML engine to ensure patient safety.