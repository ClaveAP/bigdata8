# Panduan Penyusunan Dashboard Looker Studio

Dokumen ini adalah panduan praktis untuk menyusun dashboard interaktif di Google Looker Studio menggunakan data agregasi Big Data (1.000.000 baris) hasil olahan PySpark.

## 📂 Lokasi Data Mart (CSV)
Seluruh file CSV agregasi yang sangat ringan (total < 10 KB) telah tersedia di:
👉 `reports/looker_data/`

1. **`looker_scorecard_global.csv`** (126 Bytes) - Data Ringkasan Eksekutif
2. **`looker_demografi.csv`** (6.0 KB) - Data Demografi & Perangkat
3. **`looker_metode_bayar.csv`** (2.5 KB) - Data Transaksi & Metode Pembayaran
4. **`looker_stres_impulse.csv`** (422 Bytes) - Data Psikologis & Finansial

---

## 🎨 Blueprint Desain Dashboard (4 Bagian Cerita)

### Bagian 1: Header & Executive Summary Scorecards
- **Sumber Data**: `looker_scorecard_global.csv`
- **Komponen Visual**:
  - Scorecard 1: `total_users` -> Label: **"Total Pengguna Teranalisis"** (Format: 1.000.000)
  - Scorecard 2: `avg_cart_abandonment_rate` -> Label: **"Rata-rata Cart Abandonment"** (Format: 40.21%)
  - Scorecard 3: `avg_checkout_abandon` -> Label: **"Rata-rata Batal / User / Bulan"** (Format: 5.00 kali)
  - Scorecard 4: `avg_order_value` -> Label: **"Rata-rata Nilai Pesanan (AOV)"**

### Bagian 2: Pola Perilaku & Segmentasi Perangkat (Mobile Friction)
- **Sumber Data**: `looker_demografi.csv`
- **Komponen Visual**:
  - **Donut Chart**: Dimensi `device_type`, Metrik `total_users` (Menunjukkan dominasi Mobile 60%).
  - **Geo Map / Bar Chart**: Dimensi `country`, Metrik `avg_checkout_abandon` (Menampilkan variasi pembatalan per negara).
  - **Tabel Filter Kontrol**: Filter Dropdown `device_type` dan `country`.

### Bagian 3: Metode Pembayaran & Kategori Produk
- **Sumber Data**: `looker_metode_bayar.csv`
- **Komponen Visual**:
  - **Bar Chart Komparasi**: Dimensi `preferred_payment_method`, Metrik `avg_checkout_abandon` dan `avg_cart_abandonment_rate`.
  - **Heatmap Table / Stacked Bar**: Dimensi `product_category_preference`, Metrik `total_transactions` dan `avg_return_rate`.

### Bagian 4: Faktor Psikologis (Stres Finansial vs Impulsivitas)
- **Sumber Data**: `looker_stres_impulse.csv`
- **Komponen Visual**:
  - **Line Chart Tren Ganda**:
    - Sumbu X: `stress_from_financial_decisions` (Skor 0 s.d. 10)
    - Sumbu Y: `avg_impulse_score` (Garis Merah) & `avg_cart_items` (Garis Biru)
  - **Interpretasi**: Membuktikan bahwa stres finansial mendorong perilaku impulsif memasukkan banyak barang ke keranjang yang berujung pembatalan massal saat checkout.
