# TIMELINE & TRACKER PROYEK TEKNOLOGI BIG DATA (`05_TIMELINE_TRACKER.md`)

**Kelompok**: Kelompok 8 — Anggrek Mekar Cilibende (Galih, Trias, Risha, Raihan)  
**Mata Kuliah**: Teknologi Big Data (Semester 5)  
**Fokus Proyek**: Analisis Faktor Pendorong Pembatalan Transaksi (*Cart & Checkout Abandonment*) E-Commerce Skala 1 Juta Baris  
**Metode Utama**: PySpark MLlib Decision Tree Classifier (Lengkap dengan Validation-Based Grid Tuning, Tree Pruning Ablation, Multi-Model Benchmark, SHAP, Permutation Importance, IF-THEN Rules, & K-Means Clustering)  
**Dataset**: `dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based` (`1.000.000` baris $\times$ `60` kolom mentah $\rightarrow$ `57` fitur prediktor independen)  
**Sumber Spreadsheet**: [Google Sheets Tracker](https://docs.google.com/spreadsheets/d/199qGgqLir1qj1HC9fz9oXuhW5leLQrQtXzch0n_XTgg/edit?gid=0#gid=0)

---

## Ringkasan Progres Proyek
- **Total Tugas**: 18 Aktivitas
- **Selesai**: 15 (`83,3%`)
- **Dalam Proses**: 3 (Finalisasi Manuskrip Jurnal JIKO Sinta 4, Laporan Akhir, & Materi Showcase)
- **Target Milestone Akhir**: Minggu 12 (Final Showcase, Demo Produk, Laporan Akhir, & Submisi Draft Jurnal JIKO)

---

## Matriks Aktivitas & Jadwal Kerja (Minggu 6 – 12)

| No | Aktivitas Proyek Big Data | Modul / Tahapan | Prioritas | Status | Timeline | PIC / Catatan |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | Penyusunan Desain Teknis & Alur Data (Worksheet 4) | Perencanaan & Desain | Tinggi | **Selesai** | W6 | Fondasi arsitektur proyek & Project Charter |
| **2** | Setup Lingkungan & Ingestion Dataset E-Commerce | Infrastruktur & Ingestion | Tinggi | **Selesai** | W6 - W7 | Dataset `1.000.000` baris (`233,60 MB`) & PySpark 4.1.1 aktif |
| **3** | Eksplorasi Data Awal (EDA) Sesi vs Abandonment | Exploratory Data Analysis | Sedang | **Selesai** | W6 - W7 | Audit korelasi 60 kolom (`schema_and_correlation_audit.csv`) & plot 300 DPI |
| **4** | Pembersihan Data Skala Besar & Konversi Parquet | Data Preprocessing | Tinggi | **Selesai** | W7 - W8 | `62` kolom terproses di Snappy Parquet (`37,96 MB`, hemat `83,75%`, I/O `15,39x` lebih cepat) |
| **5** | Feature Engineering: Label Risiko Abandonment | Feature Engineering | Tinggi | **Selesai** | W8 | Target biner `is_high_abandonment` (`cart_abandonment_rate >= 40%`: `50,55%` vs `49,45%`) |
| **6** | Encoding Variabel Kategorikal & Vektorisasi Dual | Feature Engineering | Sedang | **Selesai** | W8 | Split-Before-Fit (`seed=42`): `57` fitur (`features` pohon) & `132` dimensi (`features_lr` OHE+Scaled) |
| **7** | Pembagian Dataset (Train-Validation-Test Split) | Model Preparation | Sedang | **Selesai** | W8 - W9 | `799.723` Train (`599.774` Train-Sub + `199.949` Val) vs `200.277` Test |
| **8** | Pelatihan Baseline Decision Tree Classifier | Machine Learning | Tinggi | **Selesai** | W9 | `DT_Baseline_d5` (`11` node, Akurasi `94,09%`, ROC-AUC `0,9318`) & `DT_Default_d7` (`41` node, ROC-AUC `0,9773`) |
| **9** | Hyperparameter Tuning & Ablasi Pruning Decision Tree | Machine Learning | Tinggi | **Selesai** | W9 - W10 | 72-Config Validation Grid & Ablasi `M0..M4`; Champion `M2` (`maxDepth=10, minInst=20, gini`, `473` node, Test ROC-AUC `0,9904`, Akurasi `93,99%`) |
| **10** | Benchmark Multi-Model, Uji McNemar & Kontrol Negatif | Model Evaluation | Tinggi | **Selesai** | W10 | Komparasi 6 Model (`LR`, `GBT`, `DT_d10`, `RF`, `DT_d7`, `DT_d3`), Bootstrap CI, McNemar Test, & Skalabilitas PySpark |
| **11** | Ekstraksi IF-THEN Rules, SHAP & Segmentasi K-Means | Explainability & Clustering | Tinggi | **Selesai** | W10 | Aturan 4-daun `98,28%` fidelitas, SHAP + Permutation Importance (resolusi kolinearitas), & K-Means Persona (`K=2..5`) |
| **12** | Perumusan Rekomendasi Tim UI/UX & Pemasaran | Business Insights | Sedang | **Selesai** | W10 - W11 | Matriks Intervensi 4-Kuartil (`Q1–Q4`), BNPL, Mindful Cooling-Off, & Gamifikasi |
| **13** | Pengembangan Dashboard Visualisasi Interaktif | BI & Visualisasi | Tinggi | **Selesai** | W10 - W11 | 5 Data Marts Looker Studio, Dashboard HTML Mandiri, 12 Grafik 300 DPI, & Notebook Showcase |
| **14** | Rencana & Blueprint Manuskrip Jurnal Ilmiah (JIKO Sinta 4) | Publikasi Ilmiah | Tinggi | **Selesai** | W10 - W11 | Blueprint lengkap JIKO (`07_JOURNAL_PLAN_JIKO.md` & Google Docs) terverifikasi dengan eksperimen `E0–E9` |
| **15** | Pengujian Otomatis Integritas Pipeline (`unittest`) | Quality Assurance | Tinggi | **Selesai** | W11 | 5 Suite pengujian otomatis (`tests/test_pipeline_integrity.py`) lulus 100% |
| **16** | Penulisan Naskah Lengkap Jurnal Ilmiah JIKO | Publikasi Ilmiah | Tinggi | **Sedang Berjalan** | W11 - W12 | Penyusunan naskah esai Bahasa Indonesia 5.000–8.500 kata sesuai template JIKO |
| **17** | Persiapan Laporan Akhir, Slide Presentasi & Demo | Presentasi & Demo | Tinggi | **Sedang Berjalan** | W11 - W12 | Mengacu pada `docs/PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md` |
| **18** | Final Showcase, Demo Produk & Submisi Berkas Proyek | Showcase & Penyerahan | Tinggi | **Sedang Berjalan** | W12 | Penyerahan seluruh berkas proyek & naskah publikasi |
