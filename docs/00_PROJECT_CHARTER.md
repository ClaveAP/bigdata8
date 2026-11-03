# Project Charter: E-Commerce Shopper Behavior & Checkout Abandonment

## 1. Latar Belakang & Masalah
Dalam industri e-commerce global, *cart and checkout abandonment* (pengguna menambahkan barang ke keranjang/tahap akhir pembayaran namun membatalkan transaksi) merupakan salah satu penyebab terbesar hilangnya potensi pendapatan (omzet). Memahami pemicu pembatalan secara mendalam sangat esensial agar platform dapat mengintervensi dengan strategi yang presisi.

## 2. Rumusan Masalah
1. Faktor-faktor apa saja (faktor demografis, finansial/psikologis, perangkat, atau kebiasaan belanja) yang paling signifikan mendorong pengguna membatalkan transaksi pada tahap akhir?
2. Bagaimana segmentasi perilaku pengguna dapat dikelompokkan untuk membedakan tipe-tipe pembeli yang rentan melakukan *abandonment*?
3. Strategi intervensi bisnis apa yang paling efektif dirancang untuk setiap segmen agar angka konversi transaksi meningkat?

## 3. Dataset
- **Sumber**: Kaggle (`dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based`)
- **Skala Data**: 1.000.000 data pengguna (1 juta baris), 57 fitur.
- **Karakteristik**: Big Data tabular sintetis dengan korelasi statistik realistis (simulasi perilaku Amazon & Shopify).

## 4. Metodologi & Arsitektur
1. **Big Data Processing Pipeline**: Menggunakan **PySpark** untuk pengolahan, pembersihan data, dan transformasi fitur skala 1 juta baris secara terdistribusi.
2. **Machine Learning - Supervised**: Klasifikasi biner (*High Abandonment Risk* vs *Low Risk*) menggunakan algoritma Tree-based (LightGBM / XGBoost / PySpark GBTClassifier).
3. **Machine Learning - Unsupervised**: Segmentasi pengguna menggunakan **K-Means Clustering** untuk membentuk persona shopper.
4. **Model Explainability**: Menggunakan **SHAP (SHapley Additive exPlanations)** untuk membedah kontribusi tiap fitur terhadap pembatalan.
5. **Business Recommendation**: Perumusan strategi intervensi berdasarkan temuan segmen dan SHAP.
