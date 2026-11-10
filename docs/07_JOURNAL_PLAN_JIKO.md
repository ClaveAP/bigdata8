# Rencana & Blueprint Final Naskah Jurnal JIKO (Sinta 4) — Kelompok 8 Anggrek Mekar Cilibende (`07_JOURNAL_PLAN_JIKO.md`)

> **Status Eksperimen & Audit (`P1`)**: **100% SELESAI DIEKSEKUSI & TERVERIFIKASI**  
> Seluruh temuan metodologis (`A1–A11`) telah diperbaiki di kode sumber `src/`, dan seluruh eksperimen jurnal (`E0–E9`) telah selesai dijalankan pada dataset penuh (`1.000.000` baris: `799.723` Train / `200.277` Test). Seluruh angka dan grafik 300 DPI siap dituangkan ke dalam naskah JIKO.

---

## 1. Identitas & Pembingkaian Naskah Jurnal

- **Jurnal Target**: **JIKO (Jurnal Informatika dan Komputer)** — Terakreditasi Sinta 4 (`e-ISSN: 2477-3964`).
- **Bahasa Naskah**: **Bahasa Indonesia** (baku akademis, istilah asing dicetak *miring*).
- **Panjang & Format**: `5.000–8.500` kata (target ideal: **`6.000–6.500` kata** termasuk tabel, gambar, dan daftar pustaka; maksimal `12–16` halaman A4).
- **Fokus Ilmiah Naskah (Model-Centric & Methodological)**:
  Karena dataset bersumber dari repositori publik Kaggle yang bersifat sintetis (`1.000.000` baris $\times$ `60` kolom), naskah **difokuskan pada evaluasi komparatif algoritma pembelajaran mesin terdistribusi pada Apache Spark MLlib** — mencakup *trade-off* antara ketepatan prediksi (ROC-AUC, PR-AUC, F1-Score), efisiensi komputasi Big Data (kompresi Parquet, skalabilitas baris & *core* CPU), regularisasi *pre-pruning* pohon keputusan, ekstraksi aturan *white-box* (*surrogate fidelity*), serta fenomena **Greedy Collinear Feature Masking** pada atribusi fitur (Single-Tree Gini vs Ensemble Gini vs TreeSHAP vs Permutation Importance).

### Opsi Judul Naskah ($\le 12$ Kata, *Title Case*)
1. **Opsi A (Rekomendasi Utama — 11 Kata)**:  
   *Evaluasi Komparatif Model Klasifikasi dan Interpretabilitas Pohon Keputusan pada Apache Spark*
2. **Opsi B (11 Kata)**:  
   *Analisis Trade-Off Akurasi, Efisiensi, dan Interpretabilitas Decision Tree pada Apache Spark*
3. **Opsi C (10 Kata)**:  
   *Komparasi Algoritma Klasifikasi Terdistribusi dan Pemangkasan Pohon Keputusan pada PySpark*

---

## 2. Ringkasan Angka Eksperimen Terverifikasi (`E0–E9`) untuk Bagian Hasil & Pembahasan

Seluruh angka di bawah ini diambil langsung dari `reports/metrics/*.json` dan siap dimasukkan ke Tabel 1–5 naskah jurnal:

### 2.1 Karakteristik Dataset, Pra-pemrosesan Parquet, & Pencegahan Kebocoran (`E0`)
- **Populasi**: `1.000.000` baris $\times$ `60` kolom mentah (`0` *missing values*).
- **Efisiensi Penyimpanan Kolumnar**: Konversi CSV (`233,60 MB`) ke Apache Parquet ber-kompresi Snappy (`37,96 MB`) memangkas ukuran penyimpanan sebesar **`83,75%`** dan mempercepat waktu baca + agregasi penuh dari **`10,847 detik` menjadi `0,705 detik` (`15,39x` *speedup*)**.
- **Ruang Fitur (`57` Prediktor Independen)**: `43` fitur numerik (termasuk `days_since_last_purchase` dari `last_purchase_date`) + `14` fitur kategorikal (`10` negara, `4` gender, `3` perangkat, `6` metode bayar, `8` kategori produk, dll.).
  - Vektor `features` (Model Pohon): `57` dimensi (`StringIndexer`).
  - Vektor `features_lr` (Regresi Logistik): `132` dimensi (`OneHotEncoder` + `StandardScaler`).
- **Protokol *Split-Before-Fit* (`seed=42`)**:
  - Train Set: `799.723` baris (`79,97%`), dibagi secara internal menjadi **Train-Sub (`599.774` baris / `75%`)** dan **Validation (`199.949` baris / `25%`)** untuk pemilihan hyperparameter.
  - Test Set: `200.277` baris (`20,03%`), diisolasi penuh dan hanya dievaluasi satu kali di akhir.

### 2.2 Hasil Komparasi 6 Model Klasifikasi (`E1`, `E5`, `E9` — Tabel Utama Jurnal)

| Model | Konfigurasi Optimal | Kompleksitas Struktur | 5-Fold CV ROC-AUC | Test ROC-AUC (95% CI) | Test PR-AUC | Test Akurasi | Test F1-Weighted | Waktu Latih (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LR_Tuned`** | `regParam=0.01`, OHE+Scaled | 132 bobot kontinu | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9902, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9410`** | `18,26` |
| **`GBT_Tuned`** | `maxIter=20, maxDepth=5` | 20 pohon (958 node) | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9901, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9409`** | `76,01` |
| **`DT_Champion_d10`** | `maxDepth=10, minInst=20, gini` | **1 pohon (473 node)** | `0,9883 ± 0,0005` | **`0,9904`** (`[0,9900, 0,9910]`) | **`0,9905`** | **`93,99%`** | **`0,9399`** | **`14,42`** |
| **`RF_Tuned`** | `numTrees=40, maxDepth=10, minInst=20` | 40 pohon (24.944 node) | `0,9894 ± 0,0005` | `0,9894` (`[0,9888, 0,9900]`) | `0,9900` | `93,92%` | `0,9392` | `354,79` |
| **`DT_Default_d7`** | `maxDepth=7, minInst=20, gini` | 1 pohon (41 node) | `0,9793 ± 0,0043` | `0,9773` (`[0,9763, 0,9782]`) | `0,9643` | `94,09%` | `0,9409` | `8,34` |
| **`DT_Shallow_d3`** | `maxDepth=3, minInst=20, gini` | **1 pohon (7 node / 4 daun)** | `0,9688 ± 0,0064` | `0,9651` (`[0,9639, 0,9664]`) | `0,9524` | `94,06%` | `0,9406` | `8,80` |

### 2.3 Uji Statistik Berpasangan McNemar ($df = 1$) & Ukuran Efek Bootstrap ($\Delta\text{ROC-AUC}$)
1. **`LR_Tuned` vs `DT_Champion_d10`**: $b = 1.202$, $c = 986$, $\chi^2_{\text{McNemar}} = 21,13$, $p = 4,23 \times 10^{-6}$; $\Delta\text{ROC-AUC} = +0,00024$ (`95% CI [+0,00010, +0,00040]`).
2. **`GBT_Tuned` vs `DT_Champion_d10`**: $b = 1.168$, $c = 953$, $\chi^2_{\text{McNemar}} = 20,96$, $p = 4,61 \times 10^{-6}$; $\Delta\text{ROC-AUC} = +0,00016$ (`95% CI [+0,00001, +0,00031]`).
3. **`RF_Tuned` vs `DT_Champion_d10`**: $b = 1.384$, $c = 1.522$, $\chi^2_{\text{McNemar}} = 6,46$, $p = 0,0110$; $\Delta\text{ROC-AUC} = -0,00102$ (`95% CI [-0,00136, -0,00068]`) — **`DT_Champion_d10` secara signifikan mengungguli `RF_Tuned` baik pada ROC-AUC maupun Akurasi, dengan waktu latih `24,6x` lebih singkat**.

### 2.4 Ablasi *Pre-Pruning* Decision Tree (`E2` — 72 Konfigurasi Validasi)
- **Efek Kedalaman & *Overfitting* (`M0` $\rightarrow$ `M2` $\rightarrow$ `M4`)**: Peningkatan kedalaman dari `maxDepth=3` (`7` node, Test AUC `0,9651`) ke `maxDepth=10` (`473` node, Test AUC `0,9904`) meningkatkan kualitas pemeringkatan probabilitas. Namun tanpa *pre-pruning* pada `maxDepth=12, minInstancesPerNode=1` (`M4`), jumlah node meledak **`7,25x` lipat (`3.431` node)** diikuti penurunan Validation AUC ke `0,9787` dan Test Akurasi ke `93,67%`.
- **Ablasi Murni `minInstancesPerNode` (`M2` vs `M3a` pada `maxDepth=10, gini`)**: Menaikkan ambang daun dari `20` ke `50` memangkas node sebesar **37,6%** (`473` $\rightarrow$ `295` node) dengan Test AUC `0,9835`.
- **Ablasi Murni `impurity` (`M2` vs `M3b` pada `maxDepth=10, minInst=20`)**: `gini` (`473` node, Test AUC `0,9904`) unggul tipis atas `entropy` (`227` node, Test AUC `0,9883`).

### 2.5 Fidelitas Pohon Ringkas (`E6`) & Resolusi Kolinearitas Fitur (`E8`)
- **Surrogate Rule Fidelity**: Pohon dangkal `DT_Shallow_d3` (`maxDepth=3`, hanya **4 daun aturan IF-THEN** yang hanya memeriksa `impulse_buying_score <= 4,5` dan `stress_from_financial_decisions <= 3,5`) menghasilkan **kesepakatan prediksi `98,28%` (`196.825 / 200.277`)** terhadap `DT_Champion_d10` (`473` node) dan mencapai Test Akurasi **`94,06%`**.
- **Greedy Collinear Feature Masking**: Keempat prediktor psikologis (`impulse_buying_score` $r=+0,9615$, `stress_from_financial_decisions` $r=+0,9404$, `overall_stress_level` $r=+0,8635$, `impulse_purchases_per_month` $r=+0,8501$) membentuk klaster multikolinear. Pada pohon tunggal `DT_Champion_d10`, `99,72%` Gini importance terserap oleh 2 fitur pertama (`95,00%` + `4,72%`), sedangkan pada `RF_Tuned` dan **TreeSHAP**, bobot terdistribusi ke keempat fitur (`35,14%`, `31,15%`, `17,08%`, `14,04%`). Pengacakan serentak ke-4 fitur ini (*Joint Cluster Permutation*) menjatuhkan ROC-AUC seluruh model sebesar **`-0,4835` s.d. `-0,4899`** menuju `~0,50` (acak).

### 2.6 Uji Kontrol Negatif, Sensitivitas Threshold (`E7`), & Skalabilitas PySpark (`E3–E4`)
- **Kontrol Negatif (`checkout_abandonments_per_month >= 5`)**: Menghasilkan ROC-AUC **`0,4975`** (`LR` & `DT`) dan **`0,5004`** (`RF`), membuktikan ketiadaan kebocoran data pada pipeline.
- **Sensitivitas Ambang Target (`35%`, `40%`, `45%`)**: ROC-AUC konsisten pada rentang **`0,9882 – 0,9906`**.
- **Skalabilitas Komputasi**: Waktu pelatihan `DT_Champion_d10` meningkat secara linier dari `1,93 detik` (`100k` baris) ke `9,45 detik` (`1M` baris), dengan percepatan paralelisme **`2,30x`** dari `local[1]` (`20,68s`) ke `local[2]` (`9,01s`).

---

## 3. Daftar Gambar & Tabel yang Akan Dimasukkan ke Manuskrip JIKO

| Nomor | Jenis | Sumber Artefak di Repo | Judul Tabel / Gambar pada Naskah |
| :---: | :---: | :--- | :--- |
| **Tabel 1** | Tabel | `reports/metrics/feature_metadata.json` | Ringkasan Skema Dataset, Kelompok Fitur, dan Statistik Deskriptif Target (`N = 1.000.000`) |
| **Tabel 2** | Tabel | `reports/metrics/dt_champion_summary.json` | Hasil Ablasi Parameter *Pre-Pruning* dan Kriteria *Impurity* pada Decision Tree (`M0–M4`) |
| **Tabel 3** | Tabel | `reports/metrics/model_comparison.csv` | Perbandingan Kinerja Prediktif, Kompleksitas Struktur, dan Waktu Pelatihan 6 Model pada Data Uji (`N = 200.277`) |
| **Tabel 4** | Tabel | `reports/metrics/mcnemar_tests.csv` | Hasil Uji Signifikansi Berpasangan McNemar ($\chi^2, df=1$) dan Selang Kepercayaan Bootstrap 95% ($\Delta\text{ROC-AUC}$) |
| **Tabel 5** | Tabel | `reports/metrics/rules_and_fidelity.json` | Ekstraksi Aturan Keputusan *White-Box* 4 Daun (`DT_Shallow_d3`) beserta Tingkat Fidelitas Surrogate (`98,28%`) |
| **Gambar 1** | Diagram | Diagram Arsitektur Metodologi | Arsitektur Pipeline Pemrosesan Terdistribusi PySpark dengan Isolasi *Split-Before-Fit* |
| **Gambar 2** | Plot 300 DPI | `reports/model_roc_pr_curves.png` | Kurva ROC dan *Precision-Recall* Komparatif Enam Model Klasifikasi pada Data Uji |
| **Gambar 3** | Plot 300 DPI | `reports/dt_pruning_depth_curve.png` | Kurva Validasi 72 Konfigurasi: Pengaruh `maxDepth`, `minInstancesPerNode`, dan `Impurity` terhadap ROC-AUC dan Jumlah Node |
| **Gambar 4** | Plot 300 DPI | `reports/permutation_vs_gini_importance.png` | Perbandingan Atribusi Fitur Lintas Metode (*Single-Tree Gini*, *Ensemble RF Gini*, *LR Weight*, dan *Permutation Importance*) |
| **Gambar 5** | Plot 300 DPI | `reports/pyspark_scaling_benchmark.png` | Evaluasi Skalabilitas Waktu Pelatihan PySpark terhadap Volume Baris Data dan Jumlah *Core* Eksekutor |
