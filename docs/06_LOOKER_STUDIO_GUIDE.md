# Panduan Penyusunan Dashboard Google Looker Studio

Dokumen ini adalah panduan teknis untuk menyusun *dashboard* interaktif di **Google Looker Studio** menggunakan 5 tabel *Data Mart* agregasi *Big Data* (`1.000.000` baris) hasil olahan PySpark.

## 📂 Lokasi & Spesifikasi 5 Tabel Data Mart (`reports/looker_data/`)
Seluruh berkas CSV dihasilkan secara deterministik oleh [`src/export_looker_data.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/export_looker_data.py) dan [`src/extract_rules_insights.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/extract_rules_insights.py) di folder [`reports/looker_data/`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/looker_data):

1. **[`looker_scorecard_global.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/looker_data/looker_scorecard_global.csv)** (`1` baris ringkasan populasi):
   - Kolom: `total_users` (`1.000.000`), `avg_checkout_abandon` (`5,00`), `avg_cart_abandonment_rate` (`40,21`), `avg_monthly_spend` (`2.498,78`), `avg_order_value` (`255,03`).
2. **[`looker_demografi.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/looker_data/looker_demografi.csv)** (`120` baris agregasi = `10` Negara $\times$ `3` Perangkat $\times$ `4` Gender):
   - Kolom: `country`, `device_type`, `gender`, `total_users`, `avg_checkout_abandon`, `avg_cart_abandonment_rate`, `avg_monthly_spend`, `avg_session_minutes`.
3. **[`looker_metode_bayar.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/looker_data/looker_metode_bayar.csv)** (`48` baris agregasi = `6` Metode Bayar $\times$ `8` Kategori Produk):
   - Kolom: `preferred_payment_method`, `product_category_preference`, `total_transactions`, `avg_checkout_abandon`, `avg_cart_abandonment_rate`, `avg_order_value`, `avg_return_rate`.
4. **[`looker_stres_impulse.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/looker_data/looker_stres_impulse.csv)** (`11` baris agregasi = Skor Stres Finansial `0` s.d. `10`):
   - Kolom: `stress_from_financial_decisions`, `total_users`, `avg_impulse_score`, `avg_cart_items`, `avg_checkout_abandon`, `avg_cart_abandonment_rate`.
5. **[`looker_model_feature_importance.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/looker_data/looker_model_feature_importance.csv)** (`10` baris Top Fitur *Champion Decision Tree*):
   - Kolom: `feature_name`, `importance_score`, `percentage_contribution`.

---

## 🎨 Blueprint Desain Dashboard (5 Komponen Visual Utama)

### Bagian 1: Header & Executive Summary Scorecards
- **Sumber Data**: `looker_scorecard_global.csv`
- **Komponen Visual**:
  - **Scorecard 1**: `total_users` $\rightarrow$ **"Total Pengguna Teranalisis"** (`1.000.000`)
  - **Scorecard 2**: `avg_cart_abandonment_rate` $\rightarrow$ **"Rata-rata Cart Abandonment Rate"** (`40,21%`)
  - **Scorecard 3**: `avg_monthly_spend` $\rightarrow$ **"Rata-rata Pengeluaran Bulanan"** (`$2.498,78`)
  - **Scorecard 4**: `avg_order_value` $\rightarrow$ **"Rata-rata Nilai Pesanan / AOV"** (`$255,03`)

### Bagian 2: Pangsa Perangkat & Sebaran Geografis 10 Negara
- **Sumber Data**: `looker_demografi.csv`
- **Komponen Visual**:
  - **Donut Chart**: Dimensi `device_type`, Metrik `total_users` (Menampilkan dominasi `Mobile` `60,02%`, `Desktop` `29,98%`, `Tablet` `9,99%`).
  - **Geo Map / Bar Chart**: Dimensi `country` (`10` negara: `Australia`, `Brazil`, `Canada`, `China`, `France`, `Germany`, `India`, `Japan`, `UK`, `USA`), Metrik `avg_cart_abandonment_rate` dan `total_users`.
  - **Filter Kontrol Interaktif**: Dropdown `country`, `device_type`, dan `gender` (`4` kategori).

### Bagian 3: Metode Pembayaran & Kategori Produk
- **Sumber Data**: `looker_metode_bayar.csv`
- **Komponen Visual**:
  - **Grouped Bar Chart**: Dimensi `preferred_payment_method` (`6` metode), Metrik `avg_cart_abandonment_rate` dan `avg_order_value`.
  - **Pivot Heatmap Table**: Baris `product_category_preference` (`8` kategori), Kolom `preferred_payment_method`, Metrik `total_transactions` dan `avg_return_rate`.

### Bagian 4: Kurva Eskalasi Stres Finansial vs Lonjakan Pembatalan Keranjang
- **Sumber Data**: `looker_stres_impulse.csv`
- **Komponen Visual**:
  - **Combo Line & Bar Chart**:
    - Sumbu X: `stress_from_financial_decisions` (Skor `0` s.d. `10`).
    - Sumbu Y Kiri: `avg_cart_abandonment_rate` (melonjak linier dari **`4,71%`** pada skor stres `0` menjadi **`77,03%`** pada skor stres `10`).
    - Sumbu Y Kanan: `avg_impulse_score` (`0,60` s.d. `9,40`) dan kontrol negatif `avg_checkout_abandon` (datar di `~5,00`).

### Bagian 5: Explainable AI — Top Feature Importance Model Decision Tree
- **Sumber Data**: `looker_model_feature_importance.csv`
- **Komponen Visual**:
  - **Horizontal Bar Chart**: Dimensi `feature_name`, Metrik `percentage_contribution` (`impulse_buying_score` dan `stress_from_financial_decisions` mendominasi $>99\%$ *Gini importance* pada pohon keputusan tunggal).
