# Data Pipeline & Preprocessing Log

Dokumen ini mencatat setiap keputusan rekayasa data (*data engineering*), transformasi fitur, dan optimasi pemrosesan Big Data.

## Status Pipeline
- **Tahap Saat Ini**: Setup Inisial & Perancangan Arsitektur Pipeline
- **Engine**: Apache Spark via PySpark (`local[*]`)
- **Format Input**: CSV (1.000.000 baris x 57 kolom)
- **Target Format Output**: Apache Parquet (Compressed Snappy)

---

## Log Keputusan Rekayasa Data

### 1. Definisi Variabel Target (Checkout Abandonment)
Untuk model klasifikasi prediktif, target kontinu `checkout_abandonments_per_month` atau `cart_abandonment_rate` akan ditransformasikan menjadi label biner:
- `is_high_abandonment = 1` jika nilai berada di atas median/persentil ke-75.
- `is_high_abandonment = 0` jika nilai berada di bawah ambang batas tersebut.

### 2. Strategi Encoding Fitur Kategorik
- Fitur ordinal (misal: tingkat pendidikan, level pendapatan) akan menggunakan *Integer/Ordinal Encoding*.
- Fitur nominal kardinalitas rendah (misal: `device_type`, `gender`, `urban_rural`) menggunakan *One-Hot Encoding* / `StringIndexer` + `OneHotEncoderEstimator` di Spark.
- Fitur nominal kardinalitas tinggi (misal: `country`, `occupation`) menggunakan *Frequency/Target Encoding*.

### 3. Normalisasi & Scaling
- Fitur numerik dengan skewness tinggi (seperti `monthly_spend`, `average_order_value`) akan dievaluasi untuk transformasi logaritmik atau `StandardScaler` / `RobustScaler`.
