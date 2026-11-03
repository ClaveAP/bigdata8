# TIMELINE & TRACKER PROYEK TEKNOLOGI BIG DATA

**Kelompok**: Anggrek Mekar Cilibende  
**Anggota Tim**: Galih, Trias, Risha, Raihan  
**Mata Kuliah**: Teknologi Big Data (Semester 5)  
**Fokus Proyek**: Analisis Checkout Abandonment E-Commerce  
**Metode Utama**: Decision Tree Classifier (Baseline, Tuning, Tree Pruning, Feature Importance & IF-THEN Rules)  
**Dataset**: E-Commerce Shopper Behavior & Lifestyle (Kaggle - 1M baris)  
**Sumber Spreadsheet**: [Google Sheets Tracker](https://docs.google.com/spreadsheets/d/199qGgqLir1qj1HC9fz9oXuhW5leLQrQtXzch0n_XTgg/edit?gid=0#gid=0)

---

## Ringkasan Progres Proyek
- **Total Tugas**: 18 Aktivitas
- **Selesai**: 10 (55,6%)
- **Dalam Proses**: 1 (Ekstraksi Feature Importance & IF-THEN Rules)
- **Target Milestone Akhir**: Minggu 12 (Final Showcase, Demo Produk, Laporan Akhir, & Draft Jurnal)

---

## Matriks Aktivitas & Jadwal Kerja (Minggu 6 - 12)

| No | Aktivitas Proyek Big Data | Modul / Tahapan | Prioritas | Status | Timeline | PIC / Catatan |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | Penyusunan Desain Teknis & Alur Data (Worksheet 4) | Perencanaan & Desain | Tinggi | **Selesai** | W6 | Fondasi arsitektur proyek |
| **2** | Setup Lingkungan & Ingestion Dataset E-Commerce | Infrastruktur & Ingestion | Tinggi | **Selesai** | W6 - W7 | Dataset 1M (233.6 MB) & PySpark 4.3 aktif |
| **3** | Eksplorasi Data Awal (EDA) Sesi vs Abandonment | Exploratory Data Analysis | Sedang | **Selesai** | W6 - W7 | Analisis korelasi & plot visualisasi |
| **4** | Pembersihan Data Skala Besar & Missing Values | Data Preprocessing | Tinggi | **Selesai** | W7 - W8 | Validasi 1M baris & Parquet export (hemat 84.3%) |
| **5** | Feature Engineering: Label Risiko Abandonment | Feature Engineering | Tinggi | **Selesai** | W8 | Target biner is_high_abandonment (54.5% vs 45.5%) |
| **6** | Encoding Variabel Kategorikal & Profil User | Feature Engineering | Sedang | **Selesai** | W8 | StringIndexer pada 14 kolom kategorikal |
| **7** | Pembagian Dataset (Train-Test Split & Balance) | Model Preparation | Sedang | **Selesai** | W8 - W9 | 799.723 Train (80%) vs 200.277 Test (20%) |
| **8** | Pelatihan Baseline Decision Tree Classifier | Machine Learning | Tinggi | **Selesai** | W9 | Spark MLlib DT Model (799k Train, 200k Test) |
| **9** | Hyperparameter Tuning & Pruning Decision Tree | Machine Learning | Tinggi | **Selesai** | W9 - W10 | Grid Tuning: maxDepth=7, minInstances=20, Gini |
| **10** | Evaluasi Kinerja Model (Akurasi, F1, Matrix) | Model Evaluation | Tinggi | **Selesai** | W10 | Evaluasi multi-parameter & model selection |
| **11** | Ekstraksi Feature Importance & IF-THEN Rules | Business Insights | Tinggi | **Dalam Proses** | W10 | Ekstraksi aturan bisnis dari Tree |
| **12** | Perumusan Rekomendasi Tim UI/UX & Pemasaran | Business Insights | Sedang | Belum Dimulai | W10 - W11 | Tindak lanjut strategi checkout |
| **13** | Pengembangan Dashboard Visualisasi Interaktif | BI & Visualisasi | Tinggi | Belum Dimulai | W10 - W11 | Streamlit interactive UI |
| **14** | Draft Jurnal Ilmiah (Metodologi & Analisis) | Publikasi Ilmiah | Tinggi | Belum Dimulai | W8 - W10 | Penulisan artikel ilmiah |
| **15** | Penyusunan Laporan Akhir & Dokumentasi Sistem | Laporan Akhir | Tinggi | Belum Dimulai | W10 - W12 | Laporan komprehensif kuliah |
| **16** | Review Manuskrip Jurnal & Finalisasi Publikasi | Publikasi Ilmiah | Sedang | Belum Dimulai | W11 - W12 | Peer review & finishing draft jurnal |
| **17** | Persiapan Slide Presentasi & Skenario Demo | Presentasi & Demo | Tinggi | Belum Dimulai | W11 - W12 | Deck presentasi & video demo |
| **18** | Final Showcase, Demo Produk & Berkas Proyek | Showcase & Penyerahan | Tinggi | Belum Dimulai | W12 | Penyerahan seluruh berkas proyek |
