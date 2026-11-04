# Katalog Artefak & Hasil Laporan Analisis (Reports Directory)

Folder ini menyimpan seluruh keluaran visual, matriks evaluasi model prediktif, ekstraksi aturan bisnis, data mart Business Intelligence, serta dashboard eksekutif interaktif dari analisis pembatalan keranjang (*cart/checkout abandonment*) berbasis 1.000.000 data pengguna e-commerce.

---

## 📊 Daftar Artefak Visual & Analitis

| Nama Berkas | Jenis / Resolusi | Deskripsi & Interpretasi Utama |
| :--- | :--- | :--- |
| **`confusion_matrix.png`** | Gambar PNG (300 DPI) | Evaluasi performa model Champion Decision Tree pada 200.277 data uji (*unseen test data*). Menunjukkan **92.994 True Negative**, **95.162 True Positive**, **7.170 False Positive**, dan **4.951 False Negative** (Akurasi: 93,95%, ROC-AUC: 0,9832). |
| **`correlation_heatmap.png`** | Gambar PNG (300 DPI) | Matriks korelasi Pearson antar 7 fitur perilaku. Membuktikan korelasi sangat kuat antara `impulse_buying_score` dan `stress_from_financial_decisions` ($r = 0,72$) serta pengaruh langsungnya terhadap `cart_abandonment_rate` ($r = 0,63$). |
| **`customer_persona_comparison.png`** | Gambar PNG (300 DPI) | Komparasi *head-to-head* profil persona pelanggan **High Abandonment** (Skor Impulsif 7,68 & Stres 7,59) vs **Low Abandonment** (Skor Impulsif 2,26 & Stres 2,35). Menunjukkan bahwa daya beli kedua kelompok setara, namun friksi psikologis belanja berbeda jauh. |
| **`decision_tree_feature_importance.png`** | Gambar PNG (300 DPI) | Peringkat 12 fitur paling berpengaruh dalam pemisahan kelas. Menunjukkan dominasi absolut faktor psikologis: `impulse_buying_score` (94,99%) dan `stress_from_financial_decisions` (4,72%) yang bersama-sama mengontrol 99,71% keputusan pohon. |
| **`decision_tree_rules.txt`** | Dokumen Teks (25 KB) | Ekstraksi 509 node aturan keputusan pohon (*IF-THEN Business Rules*) dalam format teks terbaca manusia untuk kebutuhan integrasi sistem *rule engine* otomatis. |
| **`eda_device_vs_abandonment.png`** | Gambar PNG (300 DPI) | Analisis pola pembatalan per perangkat belanja: Mobile App Shoppers mendominasi volume pengguna (60,02%), disusul Desktop (29,98%), dan Tablet (9,99%). |
| **`eda_stress_vs_abandonment.png`** | Gambar PNG (300 DPI) | Kurva non-linear hubungan antara stres finansial dan laju pembatalan keranjang (melonjak dari 4,71% pada skor stres 1 menjadi 77,03% pada skor stres 10). |
| **`dashboard_visualisasi.html`** | Web Dashboard Interaktif | Dashboard eksekutif berbasis Chart.js mandiri (*zero-dependency*, dapat dibuka langsung di peramban web tanpa server) dengan KPI Scorecards, kurva lonjakan belanja, distribusi perangkat, bar feature importance, dan profil persona. |

---

## 🗄️ Direktori Data Mart BI (`looker_data/`)

Berisi 5 file CSV agregasi terstruktur yang dirancang khusus untuk visualisasi di **Google Looker Studio**, **Tableau**, atau **Power BI**:

1. **`looker_scorecard_global.csv`**: Metrik makro global (Total Pengguna 1.000.000, Rata-rata Laju Pembatalan 40,21%, Rata-rata Sesi Belanja 60,02 menit, Rata-rata Pengeluaran \$2.498,59).
2. **`looker_demografi.csv`**: Agregasi metrik belanja dan pembatalan berdasarkan 5 negara (*United States, United Kingdom, Germany, Canada, Australia*).
3. **`looker_metode_bayar.csv`**: Analisis efektivitas metode pembayaran (*Credit Card, PayPal, Buy Now Pay Later, Debit Card, Cryptocurrency*) terhadap risiko pembatalan.
4. **`looker_stres_impulse.csv`**: Matriks cross-tabulation tingkat stres finansial (skala 1–10) vs rata-rata skor impulsif dan laju pembatalan keranjang.
5. **`looker_model_feature_importance.csv`**: Nilai bobot kepentingan fitur pohon keputusan untuk grafik horizontal bar interaktif.

---

## 🔄 Cara Menghasilkan Ulang Seluruh Plot

Untuk menghasilkan kembali grafik laporan secara otomatis menggunakan PySpark:
```bash
# Menghasilkan Confusion Matrix, Correlation Heatmap, dan Persona Comparison
python src/export_report_plots.py

# Menghasilkan Feature Importance dan Aturan Pohon
python src/extract_rules_insights.py
```
