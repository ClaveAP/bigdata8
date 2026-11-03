"""
Modul Ingestion Dataset E-Commerce Shopper Behavior
Mengunduh dataset dari Kaggle dan memindahkannya ke direktori data/raw/
"""

import os
import shutil
import sys
from pathlib import Path

DATASET_SLUG = "dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def download_dataset():
    try:
        import kagglehub
    except ImportError:
        print("[ERROR] Library 'kagglehub' belum terpasang. Jalankan: pip install kagglehub")
        sys.exit(1)

    print(f"[INFO] Memulai unduhan dataset: {DATASET_SLUG}...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Unduh dataset via kagglehub (disimpan sementara di cache kagglehub)
    download_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"[INFO] Dataset berhasil diunduh ke cache: {download_path}")

    # Salin file dataset ke direktori data/raw/
    download_dir = Path(download_path)
    csv_files = list(download_dir.glob("*.csv"))

    if not csv_files:
        print(f"[WARN] Tidak ditemukan file .csv di {download_path}")
        return

    for csv_file in csv_files:
        dest_file = RAW_DATA_DIR / csv_file.name
        print(f"[INFO] Menyalin {csv_file.name} -> {dest_file}...")
        shutil.copy2(csv_file, dest_file)
        file_size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] File tersimpan: {dest_file.name} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    download_dataset()
