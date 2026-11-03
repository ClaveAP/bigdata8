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

---

## 📋 Task List & Status Pengerjaan (Tracking Delegasi Tim)

Berikut adalah status aktivitas proyek mengacu pada timeline resmi perkuliahan. Bagian yang berstatus **[BELUM SELESAI]** dapat didelegasikan untuk pembagian tugas anggota tim:

### ✅ Tugas yang Sudah Selesai (Completed - 55,6%)
- [x] **Aktivitas 1**: Penyusunan Desain Teknis & Alur Data (Worksheet 4)
- [x] **Aktivitas 2**: Setup Lingkungan & Ingestion Dataset E-Commerce (1M baris, PySpark 4.3)
- [x] **Aktivitas 3**: Eksplorasi Data Awal (EDA) Sesi vs Abandonment & Visualisasi Korelasi
- [x] **Aktivitas 4**: Pembersihan Data Skala Besar & Ekspor Apache Parquet (Kompresi 84.3%)
- [x] **Aktivitas 5**: Feature Engineering: Binerisasi Label Risiko Abandonment
- [x] **Aktivitas 6**: Encoding Variabel Kategorikal (StringIndexer 14 kolom & VectorAssembler)
- [x] **Aktivitas 7**: Pembagian Dataset Terdistribusi (Train 80% / Test 20%)
- [x] **Aktivitas 8**: Pelatihan Baseline Decision Tree Classifier
- [x] **Aktivitas 9**: Hyperparameter Tuning & Pruning Decision Tree (Champion Model: Depth=7)
- [x] **Aktivitas 10**: Evaluasi Kinerja Model (Akurasi, F1-Score, Model Comparison)

---

### ⏳ Tugas yang Belum Selesai & Next Steps (Siap Didelegasikan)

#### 1. Perumusan Rekomendasi Strategis Bisnis & UI/UX (Aktivitas 12)
- **Status**: Siap Dikerjakan
- **Fokus Pekerjaan**:
  - Menyusun dokumen rekomendasi intervensi bisnis di `docs/04_BUSINESS_RECOMMENDATIONS.md`.
  - Menerjemahkan temuan *Top-10 Feature Importance* dan aturan *IF-THEN Rules* (dari file `reports/decision_tree_rules.txt`) menjadi solusi nyata untuk tim UI/UX, pemasaran, dan manajemen checkout e-commerce.

#### 2. Finalisasi Dashboard Visualisasi Interaktif Looker Studio (Aktivitas 13)
- **Status**: Siap Dikerjakan
- **Fokus Pekerjaan**:
  - Mengunggah 4 file CSV data mart dari folder `reports/looker_data/` ke Looker Studio / Google Sheets.
  - Merangkai grafik: Scorecard KPI, Diagram Donut (Perangkat), Diagram Batang (Negara & Metode Pembayaran), serta Grafik Garis (Stres Finansial).
  - Mengikuti blueprint tata letak yang sudah disusun di `docs/06_LOOKER_STUDIO_GUIDE.md`.

#### 3. Penulisan Draft Jurnal Ilmiah / Manuskrip Publikasi (Aktivitas 14 & 16)
- **Status**: Siap Dikerjakan (Target W8 - W12)
- **Fokus Pekerjaan**:
  - Menyusun struktur artikel ilmiah: Abstrak, Pendahuluan, Tinjauan Pustaka, Metodologi Big Data (PySpark & Decision Tree), Hasil & Pembahasan, serta Kesimpulan.
  - Memasukkan tabel perbandingan model dan grafik feature importance ke dalam manuskrip.

#### 4. Penyusunan Laporan Akhir & Dokumentasi Sistem (Aktivitas 15)
- **Status**: Siap Dikerjakan (Target W10 - W12)
- **Fokus Pekerjaan**:
  - Mengompilasi seluruh dokumen teknis di folder `docs/` (`00_PROJECT_CHARTER`, `01_DATA_DICTIONARY`, `02_DATA_PIPELINE_LOG`, `03_MODEL_EXPERIMENTS`) menjadi satu dokumen Laporan Akhir Praktikum/Proyek.

#### 5. Persiapan Slide Presentasi & Skenario Demo Showcase (Aktivitas 17 & 18)
- **Status**: Menjelang Minggu 11 - 12
- **Fokus Pekerjaan**:
  - Membuat slide presentasi showcase (latar belakang, metodologi Big Data, demo dashboard Looker Studio, dan temuan IF-THEN rules).
  - Menyusun pembagian sesi bicara untuk demo akhir produk di depan dosen penguji.
