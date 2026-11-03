# Model Experiments & Benchmark Log

Dokumen ini memantau rekam jejak pelatihan model machine learning, metrik evaluasi, dan perbandingan antar algoritma.

## Target & Metrik Evaluasi Utama
- **Problem Type**: Binary Classification (`is_high_abandonment`)
- **Primary Metrics**: **ROC-AUC** & **F1-Score**
- **Secondary Metrics**: Accuracy, Precision, Recall

---

## Tabel Hasil Eksperimen

| Exp ID | Algoritma | Konfigurasi | Train Size | Test Size | ROC-AUC | Accuracy | F1-Score | Weighted Precision | Weighted Recall | Catatan |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-01** | **Decision Tree (Baseline)** | `maxDepth=5, seed=42` | 799.723 | 200.277 | **0.5002** | **54.52%** | **0.3852** | 0.4953 | 0.5452 | Model baseline awal PySpark MLlib. Pohon belum dituning (kedalaman terbatas). |
| **EXP-02** | **Decision Tree (Tuned & Pruned)** | `maxDepth=10, minInstances=50, gini/entropy` | 799.723 | 200.277 | *In Progress* | *In Progress* | *In Progress* | - | - | Optimasi parameter split dan tree pruning untuk menangkap interaksi non-linear. |

---

## Log Representasi Pohon Keputusan (IF-THEN Rules) Baseline
- Pada model baseline, fitur kategorikal hasil indexing (seperti `budgeting_style` dan `preferred_payment_method`) serta fitur numerik seperti `income_level` dan `daily_session_time_minutes` menjadi simpul pemisah awal (*decision nodes*).

---

## Log Hasil Hyperparameter Tuning & Pruning (Aktivitas 9 & 10)
Eksperimen grid tuning dilakukan pada skala terdistribusi (Train 799.723 baris, Test 200.277 baris):

| Konfigurasi Grid | Kriteria Impurity | Pruning (`minInstancesPerNode`) | ROC-AUC | Accuracy | F1-Score | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `maxDepth = 7` | Gini | 20 | **0.4998** | **54.39%** | **0.3952** | **CHAMPION MODEL** |
| `maxDepth = 10` | Gini | 20 | 0.4992 | 53.78% | 0.4384 | Candidate 2 |
| `maxDepth = 10` | Entropy | 50 | 0.4988 | 53.80% | 0.4362 | Candidate 3 |

### Catatan Evaluasi Karakteristik Data
- Dataset sintetis 1M profil ini memiliki variasi stokastik tinggi (*stochastic noise*) di mana target `checkout_abandonments_per_month` didistribusikan secara seragam mendekati nilai tengah 5.0.
- Model dengan kedalaman moderat (`maxDepth=7`) terbukti paling stabil mencegah overfitting dibandingkan pohon yang terlalu dalam (`maxDepth=10`), serta memberikan interpretasi aturan bisnis (*IF-THEN rules*) yang jauh lebih bersih dan aplikatif.
