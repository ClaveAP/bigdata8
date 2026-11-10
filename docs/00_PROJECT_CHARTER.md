# Project Charter: E-Commerce Shopper Behavior & Cart/Checkout Abandonment

## 1. Latar Belakang & Masalah
Dalam industri *e-commerce* global, *cart and checkout abandonment* (pengguna menambahkan barang ke keranjang namun membatalkan transaksi sebelum pembayaran tuntas) merupakan salah satu penyebab utama hilangnya potensi pendapatan (*revenue leakage*). Memahami faktor pendorong pembatalan secara akurat serta mengevaluasi *trade-off* antara akurasi prediktif dan interpretabilitas model pembelajaran mesin pada skala *Big Data* sangat krusial agar sistem dapat mengeksekusi intervensi secara *real-time*.

## 2. Rumusan Masalah
1. Faktor-faktor apa saja (psikologis/impulsivitas, finansial, demografis, perangkat, atau kebiasaan sesi) yang secara statistik menggerakkan pembatalan keranjang belanja pada populasi 1.000.000 pengguna?
2. Bagaimana *trade-off* performa prediktif ($\text{ROC-AUC}$, $\text{PR-AUC}$, $\text{F1-Score}$), kompleksitas struktur, dan kecepatan komputasi antara model *white-box* (**Decision Tree Classifier** dengan *Pruning*) dibandingkan model linier (**Logistic Regression**) dan ansambel (**Random Forest** & **Gradient-Boosted Trees**) di atas **Apache Spark MLlib**?
3. Bagaimana fenomena kolinearitas antar-fitur psikologis memengaruhi atribusi fitur pada algoritma pohon tunggal (*Gini Importance*) dibandingkan metode *ensemble bagging*, **Permutation Importance**, dan **SHAP (SHapley Additive exPlanations)**?
4. Bagaimana segmentasi perilaku pengguna (**K-Means Clustering**) dan ekstraksi aturan keputusan (**IF-THEN Rules**) dapat diterjemahkan menjadi strategi intervensi bisnis otomatis yang presisi?

## 3. Dataset & Lisensi
- **Sumber**: Kaggle (`dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based`, lisensi terbuka untuk pendidikan/penelitian dengan atribusi sumber).
- **Skala Data**: `1.000.000` baris pengguna unik $\times$ `60` kolom mentah (`233,60 MB` CSV $\rightarrow$ `37,96 MB` Apache Parquet Snappy).
- **Ruang Fitur**: `57` fitur prediktor independen (`43` numerik + `14` kategorikal, termasuk fitur turunan temporal `days_since_last_purchase`), bebas dari kebocoran variabel target (`cart_abandonment_rate`) maupun variabel kontrol negatif (`checkout_abandonments_per_month`).
- **Karakteristik**: *Big Data* tabular sintetis skala besar yang memodelkan interaksi antara klaster faktor psikologis kolinear dan perilaku belanja *e-commerce*.

## 4. Metodologi & Arsitektur
1. **Big Data Processing Pipeline**: Menggunakan **Apache Spark (PySpark 4.3.0)** dengan protokol **Split-Before-Fit** (pembagian *Train-Test* `80:20` dilakukan sebelum *fitting* `StringIndexer`, `OneHotEncoder`, dan `StandardScaler`) serta penyimpanan kolumnar **Apache Parquet**.
2. **Machine Learning - Supervised Classification**:
   - Fokus utama pada **Decision Tree Classifier** dengan desain faktorial 72 konfigurasi (`maxDepth`, `minInstancesPerNode`, `impurity`) pada *Validation Split* (`75:25` di dalam *Train Set*).
   - Komparasi adil terhadap **Logistic Regression** (OHE + *StandardScaler*), **Random Forest Classifier**, dan **Gradient-Boosted Trees (GBT)**, dilengkapi uji statistik berpasangan **McNemar** dan **Bootstrap 95% Confidence Interval**.
3. **Machine Learning - Unsupervised Segmentation**: Segmentasi perilaku pengguna menggunakan **PySpark K-Means Clustering** ($k \in \{2, 3, 4, 5\}$ dievaluasi dengan *Silhouette Score* dan WSSSE) untuk memetakan *Customer Personas*.
4. **Model Explainability (XAI)**: Menggabungkan ekstraksi aturan **IF-THEN** terhumanisasi (beserta pengukuran *Fidelity* pohon dangkal terhadap pohon penuh), **Permutation Importance** ($\Delta \text{ROC-AUC}$), dan **SHAP (SHapley Additive exPlanations)** untuk membedah efek kolinearitas fitur.
5. **Business Recommendation & BI**: Perumusan intervensi bisnis berbasis 3-Tier Risiko serta penyajian visualisasi di **Google Looker Studio**, **HTML Executive Dashboard**, dan **Jupyter Notebook Showcase**.
