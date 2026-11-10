# Data Pipeline & Preprocessing Log

Dokumen ini mencatat setiap keputusan rekayasa data (*data engineering*), audit skema 60 kolom, pencegahan kebocoran data (*Split-Before-Fit*), temuan eksplorasi data (EDA), dan transformasi fitur pada skala *Big Data* (`1.000.000` baris).

## Status Pipeline
- **Tahap Saat Ini**: Selesai 100% — Ingestion, Validasi 60 Kolom, Derivasi Fitur Temporal, Parquet Snappy Export, & Split-Before-Fit Feature Engineering.
- **Engine**: Apache Spark via PySpark 4.3.0 (`local[*]`, Apache Arrow diaktifkan).
- **Format Input**: CSV mentah (`1.000.000` baris $\times$ `60` kolom, `233,60 MB`).
- **Format Output**:
  - `data/processed/ecommerce_processed_parquet/`: `1.000.000` baris $\times$ `62` kolom (`37,96 MB`, hemat ruang penyimpanan **83,75%**).
  - `data/processed/train_data_parquet/`: `799.723` baris (`79,97%`).
  - `data/processed/test_data_parquet/`: `200.277` baris (`20,03%`).
- **Kondisi Data**: `0` nilai *Missing / Null* di seluruh `60` kolom mentah maupun `62` kolom terproses.

---

## Ringkasan Temuan EDA & Keputusan Penentuan Target

### 1. Evaluasi Variabel Target Valid vs Variabel Kontrol Negatif
- **Variabel Kontrol Negatif (`checkout_abandonments_per_month`)**:
  - Rentang nilai `0` s.d. `10` (rata-rata `4,9975`, SD `3,1628`).
  - *Temuan Audit Matematis*: Kolom ini dibangkitkan dari distribusi seragam independen (*uniform random noise*) dengan korelasi Pearson $r = -0,000944$ terhadap `cart_abandonment_rate` dan $|r| < 0,0025$ terhadap seluruh 57 fitur prediktor. Pada eksperimen kontrol negatif (**E7**), pelatihan model terhadap variabel ini menghasilkan $\text{ROC-AUC} \approx 0,500$, membuktikan bahwa *pipeline* evaluasi tidak mengalami kebocoran atau *overfitting* palsu.
- **Variabel Target Utama (`cart_abandonment_rate`)**:
  - Rentang nilai `0%` s.d. `90%` (rata-rata `40,2123%`, SD `25,4333%`).
  - *Klaster 4 Fitur Psikologis Kolinear*: Pembatalan keranjang digerakkan secara dominan oleh 4 variabel psikologis/impulsivitas yang saling berkolinearitas tinggi ($r$ antar-fitur `0,8419` s.d. `0,9162`):
    1. `impulse_buying_score`: $r = +0,9615$ terhadap `cart_abandonment_rate` ($r = +0,8414$ terhadap target biner).
    2. `stress_from_financial_decisions`: $r = +0,9404$ terhadap `cart_abandonment_rate` ($r = +0,8282$ terhadap target biner).
    3. `overall_stress_level`: $r = +0,8635$ terhadap `cart_abandonment_rate` ($r = +0,7697$ terhadap target biner).
    4. `impulse_purchases_per_month`: $r = +0,8501$ terhadap `cart_abandonment_rate` ($r = +0,7426$ terhadap target biner).
  - *53 Fitur Prediktor Lainnya*: Variabel finansial dan operasional seperti `monthly_spend` ($r = -0,0019$), `average_order_value` ($r = -0,0008$), `daily_session_time_minutes` ($r = -0,0003$), `purchase_conversion_rate` ($r = +0,0007$), dan `return_rate` ($r = -0,0005$) bersifat ortogonal terhadap pembatalan keranjang.
- **Formulasi Binerisasi Target (`is_high_abandonment`)**:
  - `cart_abandonment_rate >= 40.0%` $\rightarrow$ **High Abandonment Risk (`1`)**: `505.469` pengguna (**`50,55%`**).
  - `cart_abandonment_rate < 40.0%` $\rightarrow$ **Low Abandonment Risk (`0`)**: `494.531` pengguna (**`49,45%`**).
  - Menghasilkan keseimbangan kelas alami (~50:50) tanpa memerlukan *oversampling* sintetis (SMOTE).

### 2. Distribusi Dimensi Demografis & Perangkat (`N = 1.000.000`)
- **Perangkat (`device_type` — 3 kategori)**:
  - `Mobile`: `600.247` pengguna (`60,02%`) — rata-rata `cart_abandonment_rate`: `40,23%`, rata-rata sesi: `59,96` menit.
  - `Desktop`: `299.845` pengguna (`29,98%`) — rata-rata `cart_abandonment_rate`: `40,17%`, rata-rata sesi: `60,11` menit.
  - `Tablet`: `99.908` pengguna (`9,99%`) — rata-rata `cart_abandonment_rate`: `40,20%`, rata-rata sesi: `60,04` menit.
- **Negara (`country` — 10 negara)**: `Australia`, `Brazil`, `Canada`, `China`, `France`, `Germany`, `India`, `Japan`, `UK`, `USA` (masing-masing ~10% populasi).
- **Gender (`gender` — 4 kategori)**: `Female`, `Male`, `Non-binary`, `Other`.
- **Metode Pembayaran (`preferred_payment_method` — 6 metode)**: `Apple Pay`, `Bank Transfer`, `Credit Card`, `Debit Card`, `Google Pay`, `PayPal`.

---

## Log Pembersihan, Derivasi Fitur Temporal, & Ekspor Parquet
- **Modul**: [`src/preprocessing.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/preprocessing.py) & [`src/pipeline_utils.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/pipeline_utils.py).
- **Validasi 60 Kolom Mentah**: Seluruh 60 kolom divalidasi bebas nilai *null* (`total_nulls_across_60_cols = 0`).
- **Derivasi Fitur Tanggal (`days_since_last_purchase`)**:
  - Kolom `last_purchase_date` (`2025-01-01` s.d. `2027-01-01`) dikonversi menjadi selisih hari numerik terhadap tanggal maksimum referensi dataset (`2027-01-01`), menghasilkan fitur numerik `days_since_last_purchase` (`0` s.d. `730` hari).
- **Inklusi Penuh Fitur Numerik yang Sebelumnya Terlewat**:
  - Kolom `review_writing_frequency`, `reading_habits`, `app_usage_frequency`, dan `days_since_last_purchase` kini disertakan penuh ke dalam 43 fitur numerik.
- **Efisiensi Penyimpanan Kolumnar Apache Parquet**:
  - Ukuran CSV Mentah: **`233,60 MB`**
  - Ukuran Parquet (Snappy): **`37,96 MB`**
  - **Penghematan Ruang Disk**: **`83,75%`** dengan akselerasi I/O pembacaan dan agregasi kolumnar di PySpark.

---

## Log Feature Engineering Bebas Kebocoran (*Strict Split-Before-Fit*)
- **Modul**: [`src/feature_engineering.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/feature_engineering.py).
- **Protokol Split-Before-Fit**:
  1. Dataset dibagi terlebih dahulu menjadi **Train Set (`799.723` baris / `79,97%`)** dan **Test Set (`200.277` baris / `20,03%`)** menggunakan `randomSplit([0.8, 0.2], seed=42)`.
  2. Seluruh transformator (`StringIndexer`, `OneHotEncoder`, dan `StandardScaler`) di-*fit* **hanya pada `train_df`** (`pipeline.fit(train_df)`), lalu digunakan untuk mentransformasi `train_df` dan `test_df`. Hal ini menjamin `0%` kebocoran informasi distribusi data uji ke tahap pelatihan.
- **Dua Representasi Vektor Fitur untuk Komparasi Adil antar-Arsitektur Model**:
  1. **`features` (`57` dimensi)**: Menggabungkan `43` fitur numerik asli + `14` fitur kategorikal hasil `StringIndexer` untuk keluarga model berbasis pohon (**Decision Tree**, **Random Forest**, **GBT**).
  2. **`features_lr` (`132` dimensi)**: Menggabungkan `43` fitur numerik + `14` fitur kategorikal yang telah di-*One-Hot Encode* (`OneHotEncoder`), lalu distandarisasi (`StandardScaler(withMean=False, withStd=True)`) khusus untuk model linier (**Logistic Regression**), sehingga bebas dari bias urutan indeks semu pada variabel nominal.

---

## Log Dukungan Multi-Platform & Sanitasi Metadata Model (Google Colab & VS Code)
- **Kompatibilitas Lintas Versi PySpark (Local PySpark 4.3.0 vs Colab PySpark 4.0.5)**:
  - PySpark 4.3.0 menyisipkan atribut `"pruneTree"` di dalam `metadata/part-00000` saat menyimpan model pohon keputusan, yang memicu `IllegalArgumentException` pada PySpark 4.0.x di Google Colab.
  - Fungsi [`sanitize_spark_model_metadata`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/pipeline_utils.py#L96-L120) secara otomatis menghapus kunci `"pruneTree"` dan berkas `.crc` setiap kali `models/decision_tree_baseline` maupun `models/decision_tree_tuned` disimpan ke disk.
