# Data Pipeline & Preprocessing Log

Dokumen ini mencatat setiap keputusan rekayasa data (*data engineering*), temuan statistik eksplorasi data (EDA), dan transformasi fitur pada skala Big Data (1.000.000 baris).

## Status Pipeline
- **Tahap Saat Ini**: Selesai Eksplorasi Awal (EDA) -> Data Cleaning, Binarisasi Target Valid, & Parquet Transformation (W7 - W8 Selesai)
- **Engine**: Apache Spark via PySpark 4.3.0 (`local[*]`)
- **Format Input**: CSV mentah (1.000.000 baris x 60 kolom, 233.6 MB)
- **Format Output**: Apache Parquet terkompresi Snappy (36.75 MB, hemat storage 84.3%)
- **Kondisi Data**: Bersih dari Missing Values/Null pada seluruh kolom fitur.

---

## Ringkasan Temuan EDA & Keputusan Penentuan Target

### 1. Evaluasi & Pemilihan Variabel Target yang Tepat
- **Variabel Awal**: `checkout_abandonments_per_month` (angka 0 s.d. 10).
  - *Temuan Ilmiah*: Pada dataset sintetis Kaggle ini, kolom `checkout_abandonments_per_month` digenerate menggunakan *Uniform Random Distribution* (korelasi mendekati 0 terhadap seluruh 57 fitur perilaku/demografi). Akibatnya model hanya belajar menebak acak (ROC-AUC ~0.50).
- **Variabel Valid**: **`cart_abandonment_rate`** (persentase pembatalan keranjang belanja).
  - *Karakteristik Nyata*: Memiliki hubungan sebab-akibat statistik yang sangat kuat dengan perilaku psikologis belanja konsumen:
    - Korelasi dengan `impulse_buying_score`: **+0.9615** (sangat kuat).
    - Korelasi dengan `stress_from_financial_decisions`: **+0.9404** (sangat kuat).
  - *Rata-rata Cart Abandonment Rate*: **40.21%** (Standar Deviasi: **25.43%**).
- **Formulasi Binerisasi Target (`is_high_abandonment`)**:
  - `cart_abandonment_rate >= 40.0%` $\rightarrow$ **High Abandonment Risk (1)** (505.469 user / **50.55%**)
  - `cart_abandonment_rate < 40.0%` $\rightarrow$ **Low Abandonment Risk (0)** (494.531 user / **49.45%**)
  - Menghasilkan sebaran kelas yang sempurna (*balanced class* mendekati 50:50) tanpa perlu teknik resampling sintetis.

### 2. Pengguna Berdasarkan Perangkat (*Device Type*)
- **Mobile**: 600.247 user (60.0%) - Rata-rata durasi sesi: 59.96 menit.
- **Desktop**: 299.845 user (30.0%) - Rata-rata durasi sesi: 60.11 menit.
- **Tablet**: 99.908 user (10.0%) - Rata-rata durasi sesi: 60.04 menit.
- **Kesimpulan**: Proporsi pengguna mayoritas adalah *Mobile Shopper* (60%).

### 3. Pengaruh Stres Finansial & Impulsivitas Belanja
- Pengguna dengan kecenderungan impulsif tinggi (`impulse_buying_score` > 5.5) sangat reaktif menambahkan banyak barang ke keranjang karena stimulus emosional, namun saat tiba pada komitmen pembayaran akhir dan berhadapan dengan total tagihan, rasa penyesalan belanja (*buyer's remorse*) dan stres finansial memicu pembatalan massal.

---

## Log Pembersihan & Ekspor Parquet (Aktivitas 4 & 5)
- **Modul**: `src/preprocessing.py`
- **Missing Values**: 0 baris dihapus (dataset 100% lengkap).
- **Pencegahan Kebocoran Data (*Data Leakage Prevention*)**:
  - Kolom `cart_abandonment_rate` dan `checkout_abandonments_per_month` **dihapus dari matriks fitur independen**, agar model hanya belajar dari pola perilaku, demografi, dan psikologis murni konsumen.
- **Efisiensi Format Parquet**:
  - Ukuran CSV Mentah: **233.60 MB**
  - Ukuran Parquet (Snappy): **36.75 MB**
  - **Penghematan Storage**: **84.3% lebih hemat** dengan kecepatan I/O baca kolumnar PySpark hingga 10x lebih cepat.

---

## Log Feature Engineering & Dataset Splitting (Aktivitas 6 & 7)
- **Modul**: `src/feature_engineering.py`
- **Encoding Variabel Kategorikal**:
  - Menggunakan `StringIndexer` untuk 14 kolom nominal/ordinal: `gender`, `country`, `urban_rural`, `employment_status`, `education_level`, `relationship_status`, `occupation`, `ethnicity`, `language_preference`, `device_type`, `preferred_payment_method`, `product_category_preference`, `shopping_time_of_day`, `budgeting_style`.
- **Perakitan Fitur (`VectorAssembler`)**:
  - Menggabungkan 39 fitur numerik murni + 14 fitur hasil StringIndexer menjadi 1 vektor komprehensif `features` (total 53 dimensi fitur).
- **Pembagian Dataset (Train/Test Split 80:20)**:
  - **Train Set**: **799.723 baris** (disimpan di `data/processed/train_data_parquet/`)
  - **Test Set**: **200.277 baris** (disimpan di `data/processed/test_data_parquet/`)
- **Status Data Siap Latih**: Siap dikonsumsi langsung oleh Spark MLlib (`DecisionTreeClassifier`, `LogisticRegression`, `RandomForestClassifier`).
