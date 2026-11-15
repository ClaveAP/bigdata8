# LOG EKSPERIMEN MODEL KLASIFIKASI & EVALUASI KOMPARATIF (`03_MODEL_EXPERIMENTS.md`)

**Proyek**: Analisis Faktor Pendorong Pembatalan Transaksi (*Cart & Checkout Abandonment*) E-Commerce Skala 1 Juta Baris  
**Kelompok**: Kelompok 8 — Anggrek Mekar Cilibende  
**Algoritma Utama**: PySpark MLlib Decision Tree Classifier (dengan Ablasi Pruning, Benchmark Multi-Model, SHAP, Permutation Importance, dan K-Means Clustering)  
**Target Klasifikasi**: `is_high_abandonment` (`1` jika `cart_abandonment_rate >= 40.0%`, `0` jika `< 40.0%`)  
**Sumber Kebenaran Metrik**: `reports/metrics/*.json` & `reports/metrics/*.csv`

---

## 1. Protokol Evaluasi Bebas Kebocoran (*Zero-Leakage Evaluation Protocol*)

Seluruh eksperimen pada dokumen ini mematuhi protokol evaluasi ketat berikut:
1. **Split-Before-Fit (`seed=42`)**: Dataset 1.000.000 baris dibagi lebih dulu menjadi **Train Set (`799.723` baris / `79,97%`)** dan **Test Set (`200.277` baris / `20,03%`)** sebelum *fitting* `StringIndexer`, `OneHotEncoder`, maupun `StandardScaler`.
2. **Pemilihan Hyperparameter pada Validation Split (Aturan *Pre-Pruning A Priori*)**: Di dalam Train Set (`799.723` baris), dilakukan pembagian internal **75% Train-Sub (`599.774` baris)** dan **25% Validation (`199.949` baris)**, kemudian dicuplik secara deterministik (`seed=42`) menjadi **25% Train-Sub (`150.269` baris)** dan **50% Validation (`100.086` baris)** untuk menguji ke-**72 konfigurasi** grid Decision Tree (`maxDepth` $\in \{2..12\}$, `minInstancesPerNode` $\in \{1, 20, 50, 100\}$, `impurity` $\in \{\text{gini}, \text{entropy}\}$), dengan aturan batasan *a priori* `minInstancesPerNode >= 20` sebelum proses *tuning* (varian `minInstancesPerNode=1` diuji semata sebagai pembanding ablasi). Test Set (`200.277` baris) **sama sekali tidak disentuh** selama pemilihan model.
3. **Pelatihan Ulang pada Full Train Set**: Setelah konfigurasi dipilih dari kurva validasi, model dilatih ulang pada **100% Train Set (`799.723` baris)** dan dievaluasi **satu kali** pada **Test Set (`200.277` baris)**.
4. **Ruang Fitur Setara (`57` Prediktor Independen)**: Seluruh model pohon menggunakan vektor `features` (`57` dimensi = `43` numerik + `14` indeks kategorikal), sedangkan `LogisticRegression` menggunakan vektor `features_lr` (`132` dimensi hasil `OneHotEncoder` + `StandardScaler` pada fitur yang sama persis).

---

## 2. Eksperimen E1 — Grid Tuning & Ablasi Terkontrol Decision Tree

Skrip: [`src/tune_tree.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/tune_tree.py) | Artefak: [`dt_tuning_grid.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/dt_tuning_grid.csv), [`dt_champion_summary.json`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/dt_champion_summary.json), [`dt_pruning_depth_curve.png`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/dt_pruning_depth_curve.png)

### 2.1 Tabel Ablasi 7 Konfigurasi `M0–M4` (+ Baseline `d=5`)

Untuk menjawab pertanyaan penguji mengenai pengaruh murni `maxDepth`, `minInstancesPerNode` (*pre-pruning*), dan `impurity` (`gini` vs `entropy`) tanpa efek bauran (*confounding*), berikut hasil `7` konfigurasi ablasi di `tune_tree.py` beserta model baseline `d=5` dari `train_classifier.py` (`8` baris evaluasi total) yang dilatih penuh pada `799.723` baris Train Set dan diuji pada `200.277` baris Test Set:

| ID Model | `maxDepth` | `minInst` | `Impurity` | Jumlah Node (Val $\rightarrow$ Full) | Val ROC-AUC (`100.086` Sub) | Test ROC-AUC (`200.277`) | Test PR-AUC | Test Akurasi | Test F1-Weighted | Waktu Latih (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M0_Shallow_d3`** | 3 | 20 | `gini` | 7 $\rightarrow$ 7 | 0,9660 | 0,9651 | 0,9524 | 94,06% | 0,9406 | 5,03s |
| **`DT_Baseline_d5`** (`train_classifier.py`) | 5 | 20 | `gini` | — $\rightarrow$ 11 | — | 0,9655 | 0,9534 | 94,09% | 0,9409 | 21,70s |
| **`M1_Baseline_d7`** | 7 | 20 | `gini` | 73 $\rightarrow$ 41 | 0,9840 | 0,9773 | 0,9643 | 94,09% | 0,9409 | 5,69s |
| **`M2_Champion_d10_gini20` (CHAMPION)** | **10** | **20** | **`gini`** | **419 $\rightarrow$ 473** | **0,9897** | **0,9904** | **0,9905** | **93,99%** | **0,9399** | **7,24s** |
| **`M3a_d10_gini50`** *(Ablasi `minInst`)* | 10 | 50 | `gini` | 231 $\rightarrow$ 295 | 0,9884 | 0,9835 | 0,9779 | 94,00% | 0,9400 | 7,35s |
| **`M3b_d10_entropy20`** *(Ablasi `Impurity`)* | 10 | 20 | `entropy` | 261 $\rightarrow$ 227 | 0,9883 | 0,9883 | 0,9887 | 94,03% | 0,9403 | 7,29s |
| **`M3c_d10_entropy50`** *(Kombinasi)* | 10 | 50 | `entropy` | 179 $\rightarrow$ 161 | 0,9885 | 0,9883 | 0,9883 | 94,04% | 0,9404 | 6,63s |
| **`M4_Deep_d12_gini1`** *(Tanpa Pruning)* | 12 | 1 | `gini` | 2.461 $\rightarrow$ 3.431 | 0,9752 | 0,9871 | 0,9863 | 93,67% | 0,9367 | 8,45s |

### 2.2 Temuan Ilmiah dari Ablasi Pohon Keputusan
1. **Efek `maxDepth` & Overfitting Cabang Bawah (`M2` vs `M4`)**:
   - Seluruh **72 konfigurasi** dievaluasi 100% menggunakan *native* **PySpark `DecisionTreeClassifier`** di atas sub-sampel **25% Train-Sub (`150.269` baris) / 50% Validation (`100.086` baris)**. Berdasarkan aturan batasan *pre-pruning* yang ditetapkan **secara *a priori* sebelum *tuning*** (`minInstancesPerNode >= 20` sebagai batas praktis agar setiap daun didukung minimal 20 sampel sehingga estimasi probabilitas daun stabil dan pohon tidak menghafal derau; ini heuristik praktis, **bukan** pemenuhan syarat konsistensi asimtotik $k_n / \log n \to \infty$ yang menuntut $k_n$ tumbuh seiring $n$), **`D10_M20_GIN` (`419` node validasi $\rightarrow$ `473` node full-train)** menempati peringkat #1 kandidat ber-pruning dengan Val ROC-AUC **`0,9897`** (`0,989745`) dan Test ROC-AUC **`0,9904`**. Secara transparan, dari seluruh 72 grid tanpa pembatasan daun, `D8_M1_ENT` (`maxDepth=8, minInstancesPerNode=1, entropy`, `113` node validasi) memperoleh Val ROC-AUC **`0,9904`** (`0,990446`, selisih hanya **`0,0007` atau praktis setara**) karena pada kedalaman 8 varian `minInstancesPerNode=1` belum mengalami *overfitting*; namun ketiadaan batas daun mulai memicu ledakan simpul pada kedalaman 10 (`D10_M1_GIN` `941` node, Val AUC `0,9861`) hingga *overfitting* parah pada `M4_Deep_d12_gini1`.
   - Meningkatkan kedalaman dari `maxDepth=3` (`7` node, Val ROC-AUC `0,9660`, Test ROC-AUC `0,9651`) ke `maxDepth=10, minInstancesPerNode=20` (`473` full node) memperbaiki kalibrasi probabilitas pada sampel perbatasan (*boundary users*).
   - Namun ketika pohon dibiarkan tumbuh ke `maxDepth=12` tanpa *pre-pruning* (`minInstancesPerNode=1` pada `M4`), jumlah node meledak **`7,25x` lipat** (`473` $\rightarrow$ `3.431` full node; `2.461` node pada validasi), sedangkan **Validation ROC-AUC anjlok ke `0,9752`** dan **Test Akurasi turun ke `93,67%`**. Ini membuktikan secara empiris bahwa cabang bawah menghafal *noise* demografis/gaya hidup yang tidak memiliki sinyal sejati.
2. **Efek Murni `minInstancesPerNode` (`M2` vs `M3a` pada `maxDepth=10, gini`)**:
   - Menaikkan batas sampel minimum daun dari `20` (`M2`) menjadi `50` (`M3a`) memangkas ukuran pohon sebesar **37,6%** (`473` $\rightarrow$ `295` node), dengan penurunan Test ROC-AUC dari `0,9904` ke `0,9835` tetapi akurasi titik tetap stabil (`94,00%`).
3. **Efek Murni Kriteria `impurity` (`M2` vs `M3b` pada `maxDepth=10, minInst=20`)**:
   - Kriteria `gini` (`M2`, `473` node) menghasilkan Test ROC-AUC **`0,9904`** (sedikit di atas `entropy` pada `M3b` sebesar `0,9883`), sementara `entropy` menghasilkan pohon yang lebih ringkas (`227` node) dengan akurasi titik `94,03%`.

---

## 3. Eksperimen E2 & E3 — Komparasi Multi-Model Setara & Uji Statistik McNemar

Skrip: [`src/compare_models.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/compare_models.py) | Artefak: [`model_comparison_summary.json`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/model_comparison_summary.json), [`model_comparison.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/model_comparison.csv), [`mcnemar_tests.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/mcnemar_tests.csv), [`bootstrap_auc_deltas.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/bootstrap_auc_deltas.csv), [`model_roc_pr_curves.png`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/model_roc_pr_curves.png)

### 3.1 Tabel Benchmark 6 Model pada Full Test Set (`N = 200.277` Baris)

| Model | Spesifikasi Hyperparameter | Kompleksitas Struktur | 5-Fold CV ROC-AUC | Test ROC-AUC (95% CI) | Test PR-AUC | Test Akurasi | Test F1-Weighted | Waktu Latih (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LR_Tuned`** | `regParam=0.0, maxIter=50` (`features_lr`) | 132 bobot kontinu | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9902, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9410`** | 16,71s |
| **`GBT_Tuned`** | `maxIter=20, maxDepth=5, step=0.1` | 20 pohon (958 node) | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9901, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9409`** | 67,12s |
| **`DT_Champion_d10`** | `maxDepth=10, minInst=20, gini` | **1 pohon (473 node)** | `0,9883 ± 0,0005` | **`0,9904`** (`[0,9900, 0,9910]`) | **`0,9905`** | **`93,99%`** | **`0,9399`** | **7,24s** |
| **`RF_Tuned`** | `numTrees=40, maxDepth=10, minInst=20` | 40 pohon (24.944 node) | `0,9894 ± 0,0005` | `0,9894` (`[0,9888, 0,9900]`) | `0,9900` | `93,92%` | `0,9392` | 158,94s |
| **`DT_Default_d7`** | `maxDepth=7, minInst=20, gini` | 1 pohon (41 node) | `0,9853 ± 0,0014` | `0,9773` (`[0,9761, 0,9788]`) | `0,9643` | `94,09%` | `0,9409` | 7,05s |
| **`DT_Shallow_d3`** | `maxDepth=3, minInst=20, gini` | **1 pohon (7 node / 4 daun)** | `0,9656 ± 0,0010` | `0,9651` (`[0,9629, 0,9665]`) | `0,9524` | `94,06%` | `0,9406` | 5,99s |

> **Catatan Divergensi CV vs Test pada `DT_Default_d7` (`0,9853` $\rightarrow$ `0,9773`) dan `DT_Champion_d10` (`0,9883` $\rightarrow$ `0,9904`)**:  
> Selisih ini wajar dan **bukan** indikasi kebocoran data. 5-Fold CV dijalankan pada sub-sampel `20%` Train Set (`~128.000` baris latih per *fold*), sedangkan model uji final dilatih ulang pada `100%` Train Set (`799.723` baris), yaitu `~6×` lebih banyak data. Ada dua efek yang bekerja:  
> 1. **`DT_Champion_d10` (CV `0,9883` → Test `0,9904`)**: dengan data latih `~6×` lebih banyak, statistik di setiap daun lebih stabil sehingga pohon dalam (`473` simpul) dapat memanfaatkan kedalamannya dengan lebih andal. Skor uji yang sedikit di atas rata-rata CV adalah perilaku normal ketika model final dilatih dengan data lebih banyak daripada model-model CV.  
> 2. **`DT_Default_d7` (CV `0,9853` → Test `0,9773`)**: PySpark MLlib menggabungkan pasangan daun bersaudara yang memprediksi kelas sama. Pada data penuh, label mayoritas di tiap daun makin stabil sehingga lebih banyak pasangan daun pada pohon dangkal ini yang prediksinya sama lalu digabung (`73` simpul pada sub-sampel tuning → `41` simpul pada *full-train*). Akibatnya jumlah nilai probabilitas berbeda yang dapat dikeluarkan pohon berkurang dan pemeringkatan (ROC-AUC) menjadi lebih kasar, walaupun akurasi tetap `94,09%`. Pada `d=10` efek ini jauh lebih kecil karena pohon masih memiliki ratusan daun dengan probabilitas berbeda.

### 3.2 Matriks Konfusi Detail pada Test Set (`N = 200.277`)

| Model | True Negative (`TN`) | False Positive (`FP`) | False Negative (`FN`) | True Positive (`TP`) | Precision (`Class 1`) | Recall (`Class 1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LR_Tuned`** | 92.702 (`46,29%`) | 6.221 (`3,11%`) | 5.604 (`2,80%`) | 95.750 (`47,81%`) | 93,90% | 94,47% |
| **`GBT_Tuned`** | 92.692 (`46,28%`) | 6.231 (`3,11%`) | 5.595 (`2,79%`) | 95.759 (`47,81%`) | 93,89% | 94,48% |
| **`DT_Champion_d10`** | **93.146 (`46,51%`)** | **5.777 (`2,88%`)** | **6.264 (`3,13%`)** | **95.090 (`47,48%`)** | **94,27%** | **93,82%** |
| **`RF_Tuned`** | 92.361 (`46,12%`) | 6.562 (`3,28%`) | 5.611 (`2,80%`) | 95.743 (`47,80%`) | 93,59% | 94,46% |
| **`DT_Default_d7`** | 92.881 (`46,38%`) | 6.042 (`3,02%`) | 5.789 (`2,89%`) | 95.565 (`47,72%`) | 94,05% | 94,29% |
| **`DT_Shallow_d3`** | 94.484 (`47,18%`) | 4.439 (`2,22%`) | 7.452 (`3,72%`) | 93.902 (`46,89%`) | 95,49% | 92,65% |

### 3.3 Uji Signifikansi McNemar & Bootstrap 95% CI ($\Delta\text{ROC-AUC}$)

| Perbandingan Berpasangan ($A$ vs $B$) | Diskordan $b$ ($A$ Benar, $B$ Salah) | Diskordan $c$ ($A$ Salah, $B$ Benar) | Statistik $\chi^2_{\text{McNemar}}$ | $p\text{-value}$ | Bootstrap $\Delta\text{ROC-AUC}$ (95% CI) | Interpretasi Praktis & Ilmiah |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`LR_Tuned` vs `DT_Champion_d10`** | 1.202 | 986 | 21,13 | $4,23 \times 10^{-6}$ | `+0,00024` (`[+0,00013, +0,00037]`) | Signifikan secara statistik karena ukuran sampel raksasa ($N=200\text{k}$), namun **selisih praktis $\Delta\text{AUC}$ hanya `0,024%` (`0,00024`)**. |
| **`GBT_Tuned` vs `DT_Champion_d10`** | 1.200 | 985 | 20,96 | $4,61 \times 10^{-6}$ | `+0,00016` (`[+0,00006, +0,00030]`) | Selisih $\Delta\text{AUC}$ hanya **`0,016%`**, sementara `DT_Champion_d10` **`9,28x` lebih cepat dilatih** (`7,24s` vs `67,12s`) dan 100% *white-box*. |
| **`RF_Tuned` vs `DT_Champion_d10`** | 1.813 | 1.945 | 4,57 | $0,0326$ | `-0,00109` (`[-0,00128, -0,00089]`) | Pada data uji `DT_Champion_d10` sedikit di atas `RF_Tuned` (ROC-AUC `0,9904` vs `0,9894`; Akurasi `93,99%` vs `93,92%`), tetapi pada 5-Fold CV urutannya terbalik (`0,9883` vs `0,9894`), sehingga **keduanya setara secara praktis**. Subsampling fitur acak RF (`featureSubsetStrategy='sqrt'` = 8 dari 57 fitur per split) sering melewatkan 4 fitur psikologis utama, sehingga RF tidak memperoleh keuntungan berarti dari 40 pohonnya. |
| **`DT_Champion_d10` vs `DT_Shallow_d3`** | 1.651 | 1.801 | 6,43 | $0,0112$ | `+0,02595` (`[+0,02449, +0,02736]`) | `DT_Champion_d10` meningkatkan daya pisah peringkat probabilitas ($\Delta\text{AUC} = +0,02595$), sedangkan `DT_Shallow_d3` mempertahankan **98,28% fidelitas prediksi** hanya dengan 4 daun aturan! |

---

## 4. Eksperimen E5 & E9 — Explainable AI (SHAP, Permutation Importance, & Resolusi Kolinearitas)

Skrip: [`src/explainability_shap.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/explainability_shap.py) | Artefak: [`explainability_shap_permutation.json`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/explainability_shap_permutation.json), [`permutation_importance.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/permutation_importance.csv), [`shap_summary_plot.png`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/shap_summary_plot.png), [`permutation_vs_gini_importance.png`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/permutation_vs_gini_importance.png)

### 4.1 Mengapa Gini Importance Pohon Tunggal Berbeda dari Korelasi Pearson? (*Greedy Collinear Masking*)
Pada EDA, terdapat **4 variabel psikologis yang saling berkorelasi sangat kuat** terhadap `cart_abandonment_rate`:
1. `impulse_buying_score` ($r = +0,9615$)
2. `stress_from_financial_decisions` ($r = +0,9404$)
3. `overall_stress_level` ($r = +0,8635$)
4. `impulse_purchases_per_month` ($r = +0,8501$)

Namun, pada pohon tunggal `DT_Champion_d10`, bobot Gini Importance terkonsentrasi pada `impulse_buying_score` (**`95,00%`**) dan `stress_from_financial_decisions` (**`4,72%`**), sedangkan `overall_stress_level` dan `impulse_purchases_per_month` tampak mendekati **`0,00%`**.  
Melalui perbandingan lintas-metode atribusi fitur pada tabel di bawah, fenomena ini terbukti sebagai **Greedy Collinear Feature Masking** (ketika pohon tunggal membelah pada fitur terkuat `impulse_buying_score`, variasi dari 3 fitur kolinear lainnya sudah ikut terjelaskan sehingga tidak dipilih ulang):

| Fitur Prediktor | Korelasi Pearson ($r$ ke Target) | Single-Tree Gini (`DT_Champion_d10`) | Ensemble Gini (`RF` 100 Pohon Audit) | Koefisien Standar `LR` ($\|\beta_j\|$) | Permutation $\Delta\text{AUC}$ (`DT`) | Mean $\|\text{SHAP}\|$ (`DT`) | Mean $\|\text{SHAP}\|$ (`RF`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`impulse_buying_score`** | **`+0,9615`** | **`95,00%`** | **`35,14%`** | **`54,59%`** | **`-0,3229`** | **`0,3356`** | **`0,1804`** |
| **`stress_from_financial_decisions`** | **`+0,9404`** | **`4,72%`** | **`31,15%`** | **`32,17%`** | **`-0,0483`** | **`0,1306`** | **`0,1468`** |
| **`overall_stress_level`** | **`+0,8635`** | `0,002%` | **`17,08%`** | `0,73%` | `0,0000` | `0,0000` | **`0,0523`** |
| **`impulse_purchases_per_month`** | **`+0,8501`** | `0,001%` | **`14,04%`** | `1,07%` | `0,0000` | `0,0023` | **`0,0528`** |
| **`monthly_spend`** | `+0,0010` | `0,007%` | `0,11%` | `0,32%` | `0,0000` | `0,0048` | `0,0024` |
| **`average_order_value`** | `-0,0008` | `0,008%` | `0,10%` | `0,04%` | `0,0000` | `0,0021` | `0,0021` |

- **Bukti Permutasi Gabungan (*Joint Cluster Permutation*)**: Saat ke-4 fitur psikologis kolinear tersebut diacak bersama-sama pada data uji, **ROC-AUC jatuh sebesar `-0,4835` pada DT, `-0,4891` pada RF, dan `-0,4899` pada LR** (menuju `~0,50` tebakan acak). Ini membuktikan secara definitif bahwa hampir 100% sinyal prediktif berada di dalam klaster 4 variabel psikologis tersebut.

---

## 5. Eksperimen E6 & E7 — Uji Kontrol Negatif (*Leakage Audit*) & Sensitivitas Threshold

Skrip: [`src/compare_models.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/compare_models.py) | Artefak: [`sensitivity_and_negative_control.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/sensitivity_and_negative_control.csv)

| Eksperimen & Definisi Target | Proporsi Kelas Positif (`Class 1`) | Test ROC-AUC (`LR_Tuned`) | Test ROC-AUC (`DT_Champion_d10`) | Test ROC-AUC (`RF_Tuned`) | Kesimpulan Validasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Sensitivitas Threshold `>= 35,0%`** (`cart_abandonment_rate`) | 56,21% *(sub-uji)* | 0,9902 | 0,9892 | 0,9892 | Performa sangat stabil ($\text{AUC} \approx 0,99$). |
| **Target Utama `>= 40,0%`** (`cart_abandonment_rate`) | **50,55%** *(penuh)* / `50,53%` *(sub)* | **0,9906** *(penuh)* / `0,9905` *(sub)* | **0,9904** *(penuh)* / `0,9882` *(sub)* | **0,9894** *(penuh)* / `0,9891` *(sub)* | Titik keseimbangan kelas paling optimal (`50,55%` vs `49,45%`). |
| **Sensitivitas Threshold `>= 45,0%`** (`cart_abandonment_rate`) | 44,85% *(sub-uji)* | 0,9904 | 0,9888 | 0,9889 | Performa tetap kokoh ($\text{AUC} > 0,988$), bukan artefak angka `40%`. |
| **Kontrol Negatif**: `checkout_abandonments_per_month >= 5` | 53,29% | **`0,4975`** | **`0,4975`** | **`0,5004`** | **Terbukti $\text{AUC} \approx 0,500$ (acak murni)**: membuktikan pipeline 100% bebas *data leakage* dan mengonfirmasi temuan EDA bahwa `checkout_abandonments_per_month` adalah derau independen ($r = -0,0009$). |

---

## 6. Eksperimen E8 — Segmentasi Pelanggan K-Means Clustering

Skrip: [`src/clustering.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/clustering.py) | Artefak: [`kmeans_segmentation.json`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/kmeans_segmentation.json), [`kmeans_persona_clusters.png`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/kmeans_persona_clusters.png)

Evaluasi `KMeans` (`StandardScaler` pada 4 fitur psikologis utama: `impulse_buying_score`, `stress_from_financial_decisions`, `overall_stress_level`, `impulse_purchases_per_month`, disertai agregasi deskriptif 6 metrik perilaku belanja di seluruh `1.000.000` baris):
- **`K = 2`**: Silhouette Score tertinggi = **`0,7370`** (`WSSSE = 1.306.480,58`), memisahkan populasi secara alami menjadi segmen *Low-Stress/Low-Impulse* (`50,11%`, rata-rata pembatalan `18,91%`) dan *High-Stress/High-Impulse* (`49,89%`, rata-rata pembatalan `61,60%`).
- **`K = 3` (Dipilih Secara Operasional untuk Stratifikasi Risiko 3-Tingkat)**: Meski Silhouette tertinggi berada di `K=2`, `K = 3` (Silhouette **`0,6309`**, `WSSSE = 823.094,13`) dipilih dengan **pertimbangan operasional bisnis** guna mengisolasi segmen perbatasan **`Threshold-Zone Hesitant Shoppers` (`Persona B`)** di sekitar ambang batas `40%` antara risiko rendah dan risiko kritis (daya beli ketiga segmen identik di `~$2.497–$2.500`). *(Catatan metodologis: karena clustering memakai 4 fitur psikologis penentu langsung target `cart_abandonment_rate` sementara 53 fitur lainnya derau independen, kerapian pemisahan laju pembatalan ini merupakan proyeksi matematis untuk stratifikasi risiko operasional, bukan klaim "penemuan" pola baru tak-terawasi.)*
  1. **Persona A — Mindful & Low-Stress Shoppers (Risiko Rendah, `351.847` user / `35,18%`)**: Rata-rata `impulse_buying_score = 1,49`, `stress_from_financial_decisions = 1,53`, rata-rata `cart_abandonment_rate = 12,69%` (`0,35%` High Abandonment).
  2. **Persona B — Threshold-Zone Hesitant Shoppers (Risiko Ambang Moderat, `324.476` user / `32,45%`)**: Rata-rata `impulse_buying_score = 5,15`, `stress_from_financial_decisions = 5,15`, belanja bulanan `$2.499,45`, rata-rata `cart_abandonment_rate = 41,22%` (`55,68%` High Abandonment).
  3. **Persona C — High-Stress Impulsive Abandoners (Risiko Kritis, `323.677` user / `32,37%`)**: Rata-rata `impulse_buying_score = 8,65`, `stress_from_financial_decisions = 8,61`, belanja bulanan `$2.496,87`, rata-rata `cart_abandonment_rate = 69,11%` (`99,97%` High Abandonment).

---

## 7. Eksperimen E4 — Benchmark Skalabilitas Komputasi PySpark (`20%–100%` Train Set & `1–4` Core)

Skrip: [`src/compare_models.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/compare_models.py) | Artefak: [`scaling_rows.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/scaling_rows.csv), [`scaling_cores.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/scaling_cores.csv), [`pyspark_scaling_benchmark.png`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/pyspark_scaling_benchmark.png)

### 7.1 Skalabilitas Volume Data (`160.127` $\rightarrow$ `799.723` Baris Train Set pada `local[*]`)
| Fraksi Train Set | Baris Train | Waktu Latih `DT_Champion_d10` (s) | Waktu Latih `LR_Tuned` (s) | Waktu Latih `GBT_Tuned` (s) | Waktu Latih `RF_Tuned` (`40` trees) (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **20%** | 160.127 | **3,60s** | 4,15s | 20,34s | 50,93s |
| **40%** | 320.208 | **5,37s** | 6,92s | 31,43s | 72,04s |
| **60%** | 480.074 | **5,91s** | 8,94s | 43,99s | 94,17s |
| **80%** | 639.563 | **9,49s** | 12,36s | 59,24s | 121,82s |
| **100%** | 799.723 | **7,24s** | 16,71s | 67,12s | 158,94s |

### 7.2 Skalabilitas Paralelisme CPU (`local[1]` vs `local[2]` vs `local[4]` pada `DT_Champion_d10`)
- **`local[1]` (1 Thread)**: `17,966 detik` (Baseline `1,00x`, Throughput `44.513 baris/detik`).
- **`local[2]` (2 Thread)**: `8,293 detik` (**Speedup `2,17x`**, Throughput `96.433 baris/detik`).
- **`local[4]` (4 Thread / 2 Physical Cores HyperThreading)**: `8,198 detik` (**Speedup `2,19x`**, Throughput `97.551 baris/detik` — saturasi natural pada perangkat keras 2-core fisik / 4-thread logis).
- **Benchmark I/O Storage Berulang (`compare_models.py`)**: Pembacaan + agregasi median CSV (`29,871s`) vs Snappy Parquet (`2,498s`) mencatat percepatan **`11,96x`** (melengkapi pemindaian awal `15,39x` pada prapemrosesan dengan penghematan disk **`83,75%`**).
