# Model Experiments & Benchmark Log

Dokumen ini memantau rekam jejak pelatihan model machine learning, metrik evaluasi komparatif, tuning hyperparameter, dan penetapan champion model untuk implementasi bisnis.

## Target & Metrik Evaluasi Utama
- **Problem Type**: Binary Classification (`is_high_abandonment`)
- **Dataset Splitting**: 80% Train (**799.723 baris**) vs 20% Test (**200.277 baris**)
- **Primary Metrics**: **ROC-AUC** & **F1-Score**
- **Secondary Metrics**: Accuracy, Training Time (Latency)

---

## 1. Tabel Komparasi Benchmark 3 Model (PySpark MLlib)

Eksperimen benchmarking dijalankan secara terdistribusi pada cluster lokal PySpark untuk membandingkan 3 arsitektur model terhadap 200.277 baris data uji (*unseen test set*):

| No | Algoritma | Tipe Arsitektur | ROC-AUC | Accuracy | F1-Score | Waktu Latih (s) | Interpretabilitas Bisnis | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **Logistic Regression** | Linear Baseline | **0.9906** | **94.09%** | **0.9409** | 45.05 s | Rendah (Hanya menghasilkan koefisien bobot linier tanpa interaksi non-linear) | Baseline |
| **2** | **Decision Tree Classifier** | Tree-based (White-Box) | **0.9669** | **94.07%** | **0.9407** | **22.67 s** | **Sangat Tinggi (Aturan IF-THEN transparan & pemetaan ambang batas fitur)** | **CHAMPION MODEL** |
| **3** | **Random Forest Classifier**| Tree Ensemble (Black-Box)| **0.9884** | **93.80%** | **0.9380** | 63.68 s | Sangat Rendah (Black-box ensemble gabungan puluhan pohon, mustahil diekstrak aturan operasionalnya) | Pembanding |

---

## 2. Eksperimen Hyperparameter Tuning & Pruning (Decision Tree)

Untuk mencegah *overfitting* dan memaksimalkan generalisasi model Decision Tree, dilakukan pencarian parameter (*grid search*) dengan variasi kedalaman pohon (`maxDepth`), kriteria *impurity* (Gini vs Entropy), serta *minimum instances per node* (`minInstancesPerNode`):

| Model ID | Kedalaman (`maxDepth`) | Min Instances (`minInstancesPerNode`) | Kriteria Impurity | ROC-AUC | Accuracy | F1-Score | Catatan Pruning |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1** | 7 | 20 | Gini | 0.9669 | 94.07% | 0.9407 | Pohon ringkas, generalisasi kuat |
| **M2** | **10** | **20** | **Gini** | **0.9832** | **93.95%** | **0.9395** | **🏆 CHAMPION MODEL (Sensitivitas probabilitas tertinggi)** |
| **M3** | 10 | 50 | Entropy | 0.9235 | 94.02% | 0.9402 | Pruning terlalu agresif, penurunan ROC-AUC |

- **Champion Model Terpilih**: **Model M2** (`maxDepth=10`, `minInstancesPerNode=20`, `impurity=gini`) dengan **ROC-AUC = 0.9832** dan **Akurasi = 93.95%**.
- **Lokasi Penyimpanan**: `models/decision_tree_tuned`

---

## 3. Top-10 Feature Importance (Bobot Pengaruh Fitur)

Hasil ekstraksi bobot fitur pada model champion membuktikan faktor-faktor psikologis dan perilaku belanja yang paling dominan mendorong pembatalan transaksi:

| Peringkat | Nama Fitur | Bobot Kepentingan (*Importance Score*) | Kontribusi Relatif |
| :---: | :--- | :---: | :---: |
| **1** | `impulse_buying_score` | **0.949871** | **94.99%** |
| **2** | `stress_from_financial_decisions` | **0.047223** | **4.72%** |
| **3** | `product_category_preference_encoded` | 0.000195 | 0.02% |
| **4** | `country_encoded` | 0.000174 | 0.02% |
| **5** | `occupation_encoded` | 0.000173 | 0.02% |
| **6** | `preferred_payment_method_encoded` | 0.000139 | 0.01% |
| **7** | `notification_response_rate` | 0.000120 | 0.01% |
| **8** | `ad_views_per_day` | 0.000107 | 0.01% |
| **9** | `referral_count` | 0.000098 | 0.01% |
| **10**| `daily_session_time_minutes` | 0.000091 | 0.01% |

- **Artefak Visual**: Tersimpan di `reports/decision_tree_feature_importance.png`
- **Aturan IF-THEN Lengkap**: Tersimpan di `reports/decision_tree_rules.txt`

---

## 4. 🏆 Mengapa Decision Tree Dipilih sebagai Champion Model untuk Rekomendasi Bisnis?

Meskipun Logistic Regression dan Random Forest memiliki metrik ROC-AUC yang bersaing tipis (~0.98 - 0.99), **Decision Tree Classifier** adalah satu-satunya pilihan yang paling tepat untuk kebutuhan industri dan rekomendasi bisnis karena:

1. **Interpretabilitas Mutlak (*White-Box Explainability*)**:
   - Random Forest menggabungkan puluhan pohon acak yang menghasilkan keputusan *black-box* tanpa bisa diterjemahkan menjadi aturan bisnis.
   - Decision Tree memecah data menjadi aturan nyata yang terbaca manusia (*Human-Readable IF-THEN Rules*), misalnya:
     > *"JIKA `impulse_buying_score` > 5.5 DAN `stress_from_financial_decisions` > 4.5, MAKA probabilitas pembatalan transaksi melonjak menjadi **TINGGI (Class 1)**."*
2. **Efisiensi & Skalabilitas Komputasi Big Data**:
   - Decision Tree melatih 800.000 data hanya dalam **22.67 detik**, hampir **3x lebih cepat** daripada Random Forest (63.68 detik) dan 2x lebih cepat dibanding Logistic Regression (45.05 detik).
3. **Kesesuaian dengan Dokumen Praktikum 2 & Worksheet 4**:
   - Target pembelajaran mata kuliah Teknologi Big Data adalah menghasilkan model interpretable yang dapat diekstrak menjadi logika bisnis terapan.
