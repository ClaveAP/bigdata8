# E-Commerce Shopper Behavior & Checkout Abandonment Analysis (Big Data)

Repositori ini berfokus pada analisis pola perilaku pengguna e-commerce (skala 1.000.000 data pengguna) untuk mengidentifikasi faktor pendorong pembatalan di tahap akhir (*cart & checkout abandonment*) serta merancang strategi retensi berbasis data.

## 📌 Tujuan Proyek
1. Menganalisis faktor pendorong utama pembatalan transaksi di tahap keranjang/checkout.
2. Membangun pipeline pemrosesan Big Data terdistribusi dengan PySpark.
3. Melatih model Machine Learning klasifikasi prediktif (*High Abandonment Risk* vs *Low Risk*).
4. Melakukan segmentasi pengguna (*Customer Clustering*) untuk personalisasi strategi retensi.
5. Memberikan rekomendasi bisnis nyata berbasis interpretabilitas model (SHAP values).

## 📂 Struktur Repositori
- `docs/`: Dokumentasi terstruktur (Project Charter, Data Dictionary, Pipeline Log, Model Log, Business Strategy).
- `data/`: Penyimpanan data mentah (*raw*) dan data terproses (*processed*).
- `notebooks/`: Notebook eksperimen dan visualisasi interaktif.
- `src/`: Modul kode modular (ETL, Preprocessing, Modeling, Inference).
- `models/`: Artefak model hasil pelatihan.
- `reports/`: Hasil visualisasi grafik, matriks evaluasi, dan SHAP plot.
