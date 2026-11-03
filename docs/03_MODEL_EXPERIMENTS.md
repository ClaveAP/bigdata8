# Model Experiments & Benchmark Log

Dokumen ini memantau rekam jejak pelatihan model machine learning, metrik evaluasi, dan perbandingan antar algoritma.

## Target & Metrik Evaluasi Utama
- **Problem Type**: Binary Classification & Customer Clustering
- **Primary Metric**: **ROC-AUC** & **F1-Score** (karena memprioritaskan keseimbangan deteksi risiko tinggi tanpa terlalu banyak false alarm)
- **Secondary Metrics**: Precision, Recall, Accuracy, Log-Loss

---

## Tabel Hasil Eksperimen

| Exp ID | Algoritma | Preprocessing | ROC-AUC | F1-Score | Precision | Recall | Catatan |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *EXP-00* | *TBD (Baseline)* | *Standard Preprocessing* | - | - | - | - | *Akan diperbarui saat pelatihan model dimulai* |
| *EXP-01* | *TBD (LightGBM)* | *Encoded + Scaled* | - | - | - | - | - |
| *EXP-02* | *TBD (PySpark RF)*| *VectorAssembler* | - | - | - | - | - |

---

## Log Evaluasi Segmentasi (K-Means)
- Penentuan nilai $k$ optimal menggunakan **Elbow Method** (Sum of Squared Errors) dan **Silhouette Score**.
- Profil tiap klaster akan didokumentasikan di sini.
