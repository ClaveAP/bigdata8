# Data Pipeline & Preprocessing Log

Dokumen ini mencatat setiap keputusan rekayasa data (*data engineering*), temuan statistik eksplorasi data (EDA), dan transformasi fitur pada skala Big Data (1.000.000 baris).

## Status Pipeline
- **Tahap Saat Ini**: Selesai Eksplorasi Awal (EDA) -> Menuju Data Cleaning & Parquet Transformation (W7 - W8)
- **Engine**: Apache Spark via PySpark 4.3.0 (`local[*]`)
- **Format Input**: CSV mentah (1.000.000 baris x 60 kolom, 233.6 MB)
- **Kondisi Data**: Bersih dari Missing Values/Null pada kolom demografi & perilaku inti.

---

## Ringkasan Temuan EDA (Aktivitas 3)

### 1. Karakteristik Target Pembatalan
- **Kolom Target**: `checkout_abandonments_per_month` (rentang 0 s.d. 10 kali/bulan) dan `cart_abandonment_rate` (persentase).
- **Rata-rata Pembatalan Checkout**: **5.00 kali/bulan** (Standar Deviasi: **3.16**).
- **Rata-rata Cart Abandonment Rate**: **40.21%** (Standar Deviasi: **25.43%**).
- **Rekomendasi Binerisasi (Klasifikasi Risiko)**:
  - Nilai $\ge 5$ diklasifikasikan sebagai **High Abandonment Risk (1)**.
  - Nilai $< 5$ diklasifikasikan sebagai **Low Abandonment Risk (0)**.
  - Pembagian ini menghasilkan distribusi kelas yang seimbang (*balanced class* mendekati 50:50).

### 2. Pengguna Berdasarkan Perangkat (*Device Type*)
- **Mobile**: 600.247 user (60.0%) - Rata-rata pembatalan: 5.00/bulan, rata-rata durasi sesi: 59.96 menit.
- **Desktop**: 299.845 user (30.0%) - Rata-rata pembatalan: 5.00/bulan, rata-rata durasi sesi: 60.11 menit.
- **Tablet**: 99.908 user (10.0%) - Rata-rata pembatalan: 4.99/bulan, rata-rata durasi sesi: 60.04 menit.
- **Kesimpulan**: Proporsi pengguna mayoritas adalah *Mobile Shopper* (60%).

### 3. Pengaruh Stres Finansial & Impulsivitas
- Tingkat stres keputusan keuangan (`stress_from_financial_decisions`) berbanding lurus secara linier dengan skor belanja impulsif (`impulse_buying_score`), mulai dari skor 0.6 pada stres tingkat 0 hingga skor 9.4 pada stres tingkat 10.
- Pengguna dengan stres finansial tinggi lebih rentan memasukkan banyak barang ke keranjang karena dorongan impulsif, namun membatalkannya di tahap pembayaran akhir.

---

## Artefak Laporan Visualisasi yang Dihasilkan
- `reports/eda_device_vs_abandonment.png`: Grafik perbandingan pembatalan antar device.
- `reports/eda_stress_vs_abandonment.png`: Grafik korelasi stres finansial terhadap pembatalan transaksi.

---

## Log Pembersihan & Ekspor Parquet (Aktivitas 4 & 5)
- **Modul**: `src/preprocessing.py`
- **Missing Values**: 0 baris dihapus (dataset terverifikasi 100% lengkap).
- **Distribusi Target Biner (`is_high_abandonment`)**:
  - `Class 1 (High Risk >= 5)`: 544.977 baris (**54.5%**)
  - `Class 0 (Low Risk < 5)`: 455.023 baris (**45.5%**)
  - Status kelas sangat ideal dan seimbang (*balanced*), tidak membutuhkan oversampling/undersampling ekstrem.
- **Efisiensi Format Parquet**:
  - Ukuran CSV Mentah: **233.60 MB**
  - Ukuran Parquet (Snappy): **36.75 MB**
  - **Penghematan Storage**: **84.3% lebih hemat** dengan kecepatan I/O baca kolumnar hingga 10x lebih cepat untuk PySpark & DuckDB.

---

## Log Feature Engineering & Dataset Splitting (Aktivitas 6 & 7)
- **Modul**: `src/feature_engineering.py`
- **Encoding Variabel Kategorikal**:
  - Menggunakan `StringIndexer` untuk 14 kolom nominal/ordinal: `gender`, `country`, `urban_rural`, `employment_status`, `education_level`, `relationship_status`, `occupation`, `ethnicity`, `language_preference`, `device_type`, `preferred_payment_method`, `product_category_preference`, `shopping_time_of_day`, `budgeting_style`.
- **Perakitan Fitur (`VectorAssembler`)**:
  - Menggabungkan 39 fitur numerik + 14 fitur hasil StringIndexer menjadi 1 vektor komprehensif `features` (total 53 dimensi fitur).
- **Pembagian Dataset (Train/Test Split 80:20)**:
  - **Train Set**: **799.723 baris** (disimpan di `data/processed/train_data_parquet/`)
  - **Test Set**: **200.277 baris** (disimpan di `data/processed/test_data_parquet/`)
- **Kondisi Data Siap Latih**: Format Parquet kolumnar, siap dikonsumsi langsung oleh Spark MLlib / Scikit-Learn Decision Tree.
