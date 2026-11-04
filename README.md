# E-Commerce Shopper Behavior & Checkout Abandonment Analysis (Big Data)

Repositori ini memuat implementasi proyek **Teknologi Big Data (Semester 5)** oleh **Kelompok 8 (Anggrek Mekar Cilibende)** untuk menganalisis faktor-faktor pendorong pembatalan transaksi di tahap akhir (*cart/checkout abandonment*) pada platform e-commerce berskala **1.000.000 data pengguna** menggunakan **Apache Spark (PySpark)** dan **Decision Tree Classifier**.

---

## 👥 Anggota Kelompok (Anggrek Mekar Cilibende)
- **Raden Mas Galih Pradityo** (J0403241030)
- **Trias Aldi Prasetia** (J0403241002)
- **Risha Mufliha Safira** (J0403241094)
- **Raihan Alma Putra** (J0403241075)

---

## 🚀 Ringkasan Performa Model & Temuan Kunci

### 1. Komparasi 3 Model Machine Learning (PySpark MLlib)
Diuji pada 200.277 data uji (*unseen test set*):

| No | Model Algoritma | Tipe Arsitektur | ROC-AUC | Akurasi | F1-Score | Waktu Latih | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Logistic Regression** | Linear Baseline | **0.9906** | 94.09% | 0.9409 | 45.05 s | Baseline Pembanding |
| 2 | **Decision Tree Classifier** | Tree (White-Box) | **0.9832** | **93.95%** | **0.9395** | **22.67 s** | **🏆 CHAMPION MODEL** |
| 3 | **Random Forest Classifier** | Ensemble Bagging | **0.9884** | 93.80% | 0.9380 | 63.68 s | Black-Box Pembanding |

> **Mengapa Decision Tree Menjadi Champion Model?**  
> 1. **White-Box Explainability**: Menghasilkan aturan logika nyata (**IF-THEN Rules**) yang terbaca manusia untuk strategi bisnis operasional (Random Forest adalah black-box yang mustahil diekstrak aturan spesifiknya).  
> 2. **Efisiensi Big Data**: Waktu latih **hanya 22 detik** di cluster PySpark, hampir 3x lebih cepat dibanding Random Forest (64 detik).  
> 3. **Hyperparameter Tuning & Pruning**: Konfigurasi terbaik `maxDepth=10`, `minInstancesPerNode=20`, `impurity=gini` mencegah overfitting dan mempertahankan akurasi 93.95%.

### 2. Top-2 Faktor Dominan Penentu Pembatalan (Feature Importance)
- 🥇 **`impulse_buying_score`**: **94.99%** kontribusi keputusan.
- 🥈 **`stress_from_financial_decisions`**: **4.72%** kontribusi keputusan.
- **Insight Bisnis**: Pembeli sangat impulsif memasukkan barang ke keranjang (*impulse browsing*), namun saat tiba di rincian checkout dan melihat akumulasi total harga, muncul stres finansial dan penyesalan belanja (*buyer's remorse*) yang memicu pembatalan massal (melonjak dari 4.7% hingga 77.0%).

---

## 📊 Panduan Dashboard Interaktif Google Looker Studio

Untuk visualisasi Business Intelligence (BI) tanpa membebani browser, PySpark telah mengekspor 5 file agregasi ringkas (**Data Marts < 10 KB**) di folder `reports/looker_data/`:

| No | Nama File Data Mart | Ukuran | Visualisasi yang Dibuat di Looker Studio |
| :-: | :--- | :---: | :--- |
| 1 | **`looker_scorecard_global.csv`** | 126 B | **4 Kartu Skor (KPI)**: Total User (1M), Avg Abandon (40.21%), Avg Spend (\$2.498), AOV (\$255). |
| 2 | **`looker_stres_impulse.csv`** | 422 B | **Line Chart (Bintang Utama)**: Hubungan Stres Finansial (0-10) vs Lonjakan Pembatalan (4.7% $\rightarrow$ 77.0%). |
| 3 | **`looker_model_feature_importance.csv`** | 406 B | **Horizontal Bar Chart**: Bobot Feature Importance Model Decision Tree (Impulse 95% & Stress 4.7%). |
| 4 | **`looker_demografi.csv`** | 6.0 KB | **Donut Chart & Peta**: Sebaran Perangkat (Mobile 60%, Desktop 30%) & Pembatalan per Negara. |
| 5 | **`looker_metode_bayar.csv`** | 2.5 KB | **Bar Chart Transaksi**: Volume & Pembatalan per Metode Bayar (Credit Card, PayPal, Apple Pay, dll). |

*Panduan klik detail tata letak dashboard tersedia di [docs/06_LOOKER_STUDIO_GUIDE.md](docs/06_LOOKER_STUDIO_GUIDE.md).*

---

## 📌 Alur Pipeline & Reproducibility (Panduan Menjalankan Proyek)

Karena ukuran dataset asli adalah **233,6 MB** (melebihi batas upload GitHub 100 MB), dataset mentah, file Parquet, dan bobot model sengaja tidak di-commit ke Git. 

Setiap anggota tim dapat mereplikasi seluruh data, model, dan laporan dari awal di laptop masing-masing dengan mengikuti langkah-langkah berikut:

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
*Memvalidasi missing value, membuat target biner valid `cart_abandonment_rate >= 40.0%`, dan menyimpan data bersih ke `data/processed/ecommerce_processed_parquet/`.*

### 4. Feature Engineering & Split Train/Test (80:20)
```bash
python src/feature_engineering.py
```
*Melakukan StringIndexer pada 14 kolom kategorikal, merakit 53 dimensi fitur via VectorAssembler (bebas dari data leakage), dan membagi data: Train (799.723 baris) serta Test (200.277 baris).*

### 5. Benchmark Model, Hyperparameter Tuning & Ekstraksi Aturan
```bash
# Komparasi 3 Algoritma (Logistic Regression vs Decision Tree vs Random Forest)
python src/compare_models.py

# Hyperparameter Grid Tuning & Tree Pruning (Champion Model: depth=10, minInstances=20)
python src/tune_tree.py

# Ekstraksi Feature Importance & IF-THEN Rules
python src/extract_rules_insights.py
```

### 6. Ekspor Data Mart untuk Dashboard Looker Studio
```bash
python src/export_looker_data.py
```
*Menghasilkan file agregasi ringkas di `reports/looker_data/` yang siap di-upload ke Looker Studio.*

---

## 📂 Struktur Direktori Repositori
- `docs/`: Ruang kendali dokumentasi proyek:
  - 📘 **`PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md`**: **Panduan utama persiapan presentasi, bedah korelasi, dan simulasi tanya-jawab dosen.**
  - `00_PROJECT_CHARTER.md`: Rumusan masalah & hipotesis bisnis.
  - `01_DATA_DICTIONARY.md`: Kamus 60 kolom fitur dataset e-commerce.
  - `02_DATA_PIPELINE_LOG.md`: Log teknis preprocessing Parquet & binarisasi target.
  - `03_MODEL_EXPERIMENTS.md`: Log metrik evaluasi komparasi 3 model & tuning Decision Tree.
  - `04_BUSINESS_RECOMMENDATIONS.md`: Strategi intervensi bisnis (BNPL, Save-for-Later, Exit-Intent Voucher).
  - `05_TIMELINE_TRACKER.md`: Pelacak progres 18 aktivitas proyek (sinkron dengan Google Sheets).
  - `06_LOOKER_STUDIO_GUIDE.md`: Blueprint panduan visualisasi dashboard.
- `src/`: Modul kode produksi PySpark modular (`preprocessing.py`, `tune_tree.py`, dll).
- `reports/`: Hasil visualisasi grafik feature importance, diagram EDA, teks aturan IF-THEN (`decision_tree_rules.txt`), dan Data Mart Looker Studio (`reports/looker_data/`).
- `data/`: Folder penyimpanan lokal data mentah dan data Parquet (dilindungi `.gitignore`).
- `models/`: Folder penyimpanan lokal model terlatih PySpark MLlib (`models/decision_tree_tuned/`).

---

## 📋 Task List & Status Pengerjaan (Tracking Delegasi Tim)

Berikut adalah status aktivitas proyek mengacu pada [Google Sheets Tracker](https://docs.google.com/spreadsheets/d/199qGgqLir1qj1HC9fz9oXuhW5leLQrQtXzch0n_XTgg/edit?gid=0#gid=0):

### ✅ Tugas yang Sudah Selesai (Completed - 12 / 18 = 66,7%)
- [x] **Aktivitas 1**: Penyusunan Desain Teknis & Alur Data (Worksheet 4)
- [x] **Aktivitas 2**: Setup Lingkungan & Ingestion Dataset E-Commerce (1M baris, PySpark 4.3)
- [x] **Aktivitas 3**: Eksplorasi Data Awal (EDA) Sesi vs Abandonment & Visualisasi Korelasi
- [x] **Aktivitas 4**: Pembersihan Data Skala Besar & Ekspor Apache Parquet (Kompresi 84.3%)
- [x] **Aktivitas 5**: Feature Engineering: Binerisasi Label Risiko Abandonment
- [x] **Aktivitas 6**: Encoding Variabel Kategorikal (StringIndexer 14 kolom & VectorAssembler 53 fitur)
- [x] **Aktivitas 7**: Pembagian Dataset Terdistribusi (Train 80% / Test 20%)
- [x] **Aktivitas 8**: Pelatihan Baseline Decision Tree Classifier
- [x] **Aktivitas 9**: Hyperparameter Tuning & Pruning Decision Tree (Champion Model M2: Depth=10, ROC-AUC 0.9832)
- [x] **Aktivitas 10**: Evaluasi Kinerja Model (Benchmark 3 Model: LR vs DT vs RF)
- [x] **Aktivitas 11**: Ekstraksi Feature Importance & IF-THEN Rules Operasional
- [x] **Aktivitas 12**: Perumusan Rekomendasi Tim UI/UX & Pemasaran (BNPL, Split Wishlist, Anti-Idle Discount)

---

### ⏳ Tugas yang Sedang Berjalan & Siap Didelegasikan (Next Milestones)

#### 1. Finalisasi Dashboard Visualisasi Interaktif Looker Studio (Aktivitas 13)
- **Status**: **Sedang Berjalan**
- **Fokus Pekerjaan**:
  - Mengunggah 5 file CSV data mart dari folder `reports/looker_data/` ke Looker Studio.
  - Merangkai grafik: Scorecard KPI, Line Chart Korelasi Stres-Impulsif, Horizontal Bar Feature Importance, dan Donut Chart Perangkat.
  - Mengikuti blueprint tata letak di `docs/06_LOOKER_STUDIO_GUIDE.md`.

#### 2. Penulisan Draft Jurnal Ilmiah / Manuskrip Publikasi (Aktivitas 14 & 16)
- **Status**: Siap Dikerjakan (Target W8 - W12)
- **Fokus Pekerjaan**:
  - Menyusun naskah ilmiah: Abstrak, Pendahuluan, Tinjauan Pustaka, Metodologi Big Data (PySpark & Decision Tree), Hasil Evaluasi & Pembahasan, serta Kesimpulan.
  - Memasukkan tabel perbandingan model dan grafik feature importance ke dalam manuskrip.

#### 3. Penyusunan Laporan Akhir & Dokumentasi Sistem (Aktivitas 15)
- **Status**: Siap Dikerjakan (Target W10 - W12)
- **Fokus Pekerjaan**:
  - Mengompilasi seluruh dokumen teknis di folder `docs/` menjadi satu berkas Laporan Akhir Praktikum/Proyek komprehensif.

#### 4. Persiapan Slide Presentasi & Skenario Demo Showcase (Aktivitas 17 & 18)
- **Status**: Menjelang Minggu 11 - 12
- **Fokus Pekerjaan**:
  - Membuat slide presentasi showcase mengacu pada `docs/PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md`.
  - Pembagian peran pembicara untuk demo akhir produk di depan dosen penguji.
