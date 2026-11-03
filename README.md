# E-Commerce Shopper Behavior & Checkout Abandonment Analysis (Big Data)

Repositori ini memuat implementasi proyek **Teknologi Big Data (Semester 5)** oleh **Kelompok 8 (Anggrek Mekar Cilibende)** untuk menganalisis faktor-faktor pendorong pembatalan transaksi di tahap akhir (*cart/checkout abandonment*) pada platform e-commerce berskala **1.000.000 data pengguna** menggunakan **Apache Spark (PySpark)** dan **Decision Tree Classifier**.

---

## 👥 Anggota Kelompok (Anggrek Mekar Cilibende)
- **Raden Mas Galih Pradityo** (J0403241030)
- **Trias Aldi Prasetia** (J0403241002)
- **Risha Mufliha Safira** (J0403241094)
- **Raihan Alma Putra** (J0403241075)

---

## 📌 Alur Pipeline & Reproducibility (Panduan Menjalankan Proyek)

Karena ukuran dataset asli adalah **233,6 MB** (melebihi batas upload GitHub 100 MB), dataset mentah, file Parquet, dan bobot model sengaja tidak di-commit ke Git. 

Setiap anggota tim dapat mereplikasi seluruh data, model, dan laporan dari awal di laptop masing-masing dengan mengikuti 5 langkah mudah berikut:

### 1. Setup Environment
Pastikan laptop sudah terinstall **Java (OpenJDK 11 atau 17)** untuk menjalankan Apache Spark:
```bash
# Clone repositori
git clone https://github.com/ClaveAP/bigdata8.git
cd bigdata8

# Buat virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Unduh Dataset 1 Juta Baris (Otomatis ke `data/raw/`)
```bash
python src/download_data.py
```
*Script ini otomatis mengunduh dataset asli dari Kaggle (`e_commerce_shopper_behaviour_and_lifestyle.csv` - 233.6 MB) dan menyimpannya di folder `data/raw/`.*

### 3. Pembersihan Data & Konversi ke Apache Parquet (Hemat 84.3%)
```bash
python src/preprocessing.py
```
*Memvalidasi missing value, membuat target biner `is_high_abandonment` (ambang batas >= 5), dan menyimpan data bersih ke `data/processed/ecommerce_processed_parquet/`.*

### 4. Feature Engineering & Split Train/Test (80:20)
```bash
python src/feature_engineering.py
```
*Melakukan StringIndexer pada 14 kolom kategorikal, menggabungkan 53 dimensi fitur via VectorAssembler, dan membagi dataset menjadi Train (799.723 baris) serta Test (200.277 baris).*

### 5. Pelatihan Model Decision Tree & Ekstraksi Aturan Bisnis
```bash
# Melatih baseline model
python src/train_classifier.py

# Hyperparameter Grid Tuning & Tree Pruning (Champion Model: depth=7, minInstances=20)
python src/tune_tree.py

# Ekstraksi Feature Importance & IF-THEN Rules
python src/extract_rules_insights.py
```

### 6. Ekspor Data Mart untuk Dashboard Looker Studio
```bash
python src/export_looker_data.py
```
*Menghasilkan file agregasi ringkas (< 10 KB) di `reports/looker_data/` yang siap di-upload ke Google Sheets / Looker Studio.*

---

## 📂 Struktur Direktori Repositori
- `docs/`: Ruang kendali dokumentasi proyek:
  - `00_PROJECT_CHARTER.md`: Rumusan masalah & hipotesis bisnis.
  - `01_DATA_DICTIONARY.md`: Kamus 60 kolom fitur dataset e-commerce.
  - `02_DATA_PIPELINE_LOG.md`: Log teknis preprocessing Parquet & temuan statistik EDA.
  - `03_MODEL_EXPERIMENTS.md`: Log metrik evaluasi model & benchmark tuning.
  - `05_TIMELINE_TRACKER.md`: Pelacak progres 18 aktivitas proyek (sinkron dengan Google Sheets).
  - `06_LOOKER_STUDIO_GUIDE.md`: Blueprint panduan visualisasi dashboard.
- `src/`: Modul kode produksi PySpark modular.
- `reports/`: Hasil visualisasi grafik feature importance, diagram EDA, teks aturan IF-THEN, dan dashboard interaktif (`dashboard_visualisasi.html`).
- `data/`: Folder penyimpanan lokal data mentah dan data Parquet (dilindungi `.gitkeep` dan `.gitignore`).
- `models/`: Folder penyimpanan lokal artefak model Decision Tree PySpark MLlib.
