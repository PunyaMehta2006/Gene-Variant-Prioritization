import os
import urllib.request

DATASETS = {
    "GTEx TPM Matrix": {
        "url": "https://storage.googleapis.com/gtex_analysis_v11/single_tissue_gene_expression/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
        "filename": "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz"
    },
    "ClinVar Summary": {
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
        "filename": "variant_summary.txt.gz"
    }
}

def download_file(url, target_path):
    print(f"Downloading {os.path.basename(target_path)}...")
    urllib.request.urlretrieve(url, target_path)
    print(f"Saved: {target_path}")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    
    for name, info in DATASETS.items():
        output_path = os.path.join(".", info["filename"])
        if not os.path.exists(output_path):
            print(f"Fetching {name}...")
            download_file(info["url"], output_path)
        else:
            print(f"Skipping {name} (File already exists).")

    print("\nDataset download setup complete!")