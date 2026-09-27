# Project Setup & Dataset Access

This repository contains the source code for the genomic evaluation engine (`scoregem2_final.py`). Due to file size limitations, raw datasets are not stored directly in this Git repository.

---

## 📥 Dataset Setup Instructions

### 1. Automated Dataset Download
Run the automated dataset downloader to pull the latest versions of **GTEx** and **ClinVar**:

**Using Python (Cross-platform):**
```bash
python download_data.py
```

**Using Bash (Linux/macOS):**
```bash
bash download_data.sh
```

---

### 2. Manual Downloads & Restricted Datasets

Place all downloaded files into the root project directory alongside `scoregem2_final.py`:

| Dataset | Source / Link | Local Destination |
| :--- | :--- | :--- |
| **GTEx Expression** | [GTEx Portal](https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression) | `./GTEx_Analysis_...gene_median_tpm.gct.gz` |
| **ClinVar Summary** | [NCBI FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/) | `./variant_summary.txt.gz` |
| **CADD PHRED Scores** | [CADD Portal](https://cadd.gs.washington.edu/score) | `./cadd_scores.tsv` |
| **JSNP Data** | *Shared Team Drive / Shared Storage* | `./Control_JSNP550typed/` |

---

## 🚀 Running the Pipeline

Once datasets are in place, run:
```bash
python scoregem2_final.py
```