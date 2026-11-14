# 📘 BUKU PANDUAN LENGKAP & BEDAH PROYEK BIG DATA (`PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md`)
## Analisis Faktor Pendorong Pembatalan Transaksi (*Cart Abandonment*) E-Commerce Menggunakan PySpark & Decision Tree Classifier

> **Dokumen ini adalah panduan utama persiapan presentasi, contekan teknis, dan pedoman menghadapi pertanyaan kritis dosen penguji saat sidang/showcase proyek.**  
> Seluruh angka pada dokumen ini telah disinkronkan 100% dengan keluaran terverifikasi di `reports/metrics/*.json`.

---

# DAFTAR ISI
1. [Ringkasan Eksekutif (Elevator Pitch 1 Menit)](#1-ringkasan-eksekutif-elevator-pitch-1-menit)
2. [Latar Belakang & Masalah Bisnis](#2-latar-belakang--masalah-bisnis)
3. [Alur Pipeline Big Data dari Hulu ke Hilir (Zero-Leakage)](#3-alur-pipeline-big-data-dari-hulu-ke-hilir-zero-leakage)
4. [Kisah Ilmiah di Balik Data: Kontrol Negatif 0,50 vs Target Valid 0,9904](#4-kisah-ilmiah-di-balik-data-kontrol-negatif-050-vs-target-valid-09904)
5. [Bedah Korelasi, Kolinearitas 4 Fitur Psikologis, & Explainable AI (SHAP)](#5-bedah-korelasi-kolinearitas-4-fitur-psikologis--explainable-ai-shap)
6. [Benchmark 6 Model Klasifikasi & Uji Statistik McNemar](#6-benchmark-6-model-klasifikasi--uji-statistik-mcnemar)
7. [Validation Grid Tuning (72 Konfigurasi) & Ablasi Tree Pruning](#7-validation-grid-tuning-72-konfigurasi--ablasi-tree-pruning)
8. [Ekstraksi Aturan Bisnis (IF-THEN Rules), Fidelitas 98,28%, & Segmentasi K-Means](#8-ekstraksi-aturan-bisnis-if-then-rules-fidelitas-9828--segmentasi-k-means)
9. [Bank Simulasi Tanya Jawab Sidang Dosen (22 Pertanyaan Kritis & Jawaban Tuntas)](#9-bank-simulasi-tanya-jawab-sidang-dosen-22-pertanyaan-kritis--jawaban-tuntas)
10. [Kamus Istilah Teknis (Glosarium)](#10-kamus-istilah-teknis-glosarium)
11. [Panduan Menjalankan di Google Colab (*Plug & Play*)](#11-panduan-menjalankan-di-google-colab-plug--play)
12. [Panduan Lengkap 12 Grafik Visualisasi & Cara Membacanya](#12-panduan-lengkap-12-grafik-visualisasi--cara-membacanya)
13. [Apa Saja yang Berubah & Ditambahkan? (Tabel Versi Lama vs Versi Baru)](#13-apa-saja-yang-berubah--ditambahkan-tabel-versi-lama-vs-versi-baru)
14. [Kronologi Pengerjaan End-to-End dari Awal hingga Akhir (10 Tahap Metodologi)](#14-kronologi-pengerjaan-end-to-end-dari-awal-hingga-akhir-10-tahap-metodologi)
15. [Blueprint Slide Presentasi PPT (14 Slide) & Pembagian Peran Tim](#15-blueprint-slide-presentasi-ppt-14-slide--pembagian-peran-tim)

---

# 1. Ringkasan Eksekutif (Elevator Pitch 1 Menit)

Jika dosen meminta kamu menjelaskan proyek ini dalam 1 menit, ucapkan kalimat ini:

> *"Proyek kami memecahkan masalah klasik e-commerce, yaitu **tingginya pembatalan transaksi di keranjang belanja (Cart Abandonment)**.  
> Menggunakan dataset **1.000.000 pengguna (`60` kolom mentah $\rightarrow$ `57` fitur prediktor independen)**, kami membangun pipeline **Apache Spark (PySpark)** dengan protokol **Split-Before-Fit bebas kebocoran data** yang mengompresi CSV mentah `233,60 MB` menjadi Snappy Parquet `37,96 MB` (**hemat `83,75%` ruang simpan dan `15,39x` lebih cepat dibaca**).  
> Melalui *grid tuning* 72 konfigurasi pada Validation Split, **Champion Decision Tree (`maxDepth=10, minInstancesPerNode=20, gini`, `473` node)** mencapai **Test ROC-AUC `0,9904`, PR-AUC `0,9905`, dan Akurasi `93,99%`** pada `200.277` baris data uji hanya dalam **`14,42 detik`** — mengungguli Random Forest (`0,9894`, `354,79 detik`) dan hanya berbeda `0,00024` poin AUC dari Logistic Regression (`0,9906`).  
> Lebih jauh, kami membuktikan bahwa pohon keputusan ringkas **4 daun (`maxDepth=3`) memiliki fidelitas `98,28%`** terhadap pohon penuh, dikombinasikan dengan analisis **TreeSHAP**, **Permutation Importance**, dan **K-Means Clustering** untuk merancang strategi intervensi bisnis seperti **cicilan BNPL dinamis** dan **Mindful Cart Cooling-Off**."*

---

# 2. Latar Belakang & Masalah Bisnis

### Apa itu *Cart Abandonment*?
Dalam bisnis e-commerce, *Cart Abandonment* adalah kondisi ketika pengguna sudah memilih produk dan memasukkannya ke keranjang belanja (*shopping cart*), tetapi menutup aplikasi atau membatalkan pesanan sebelum menyelesaikan pembayaran.

### Mengapa Masalah Ini Kritis?
1. **Biaya Akuisisi Pelanggan (*CAC*) Terbuang**: Kampanye pemasaran yang sudah berhasil membawa pengguna hingga ke keranjang menjadi sia-sia ketika konversi gagal di tahap akhir.
2. **Kebutuhan Intervensi Berbasis Aturan (*Rule-Based Personalization*)**: Tim produk dan pemasaran membutuhkan aturan yang **transparan dan dapat dieksekusi dalam milidetik** (*white-box rules*) untuk membedakan pengguna yang batal karena impulsif sesaat vs pengguna yang batal karena kecemasan finansial (*buyer's remorse*).

---

# 3. Alur Pipeline Big Data dari Hulu ke Hilir (Zero-Leakage)

```
[1. Dataset Mentah Kaggle]
      │ CSV 233,60 MB (1.000.000 baris × 60 kolom mentah, 0 missing values)
      ▼
[2. Ingestion & Preprocessing (src/preprocessing.py)]
      │ Derivasi fitur recency: days_since_last_purchase (dari last_purchase_date)
      │ Binarisasi Target Valid: cart_abandonment_rate >= 40,0% -> is_high_abandonment (50,55% Kelas 1)
      │ Ekspor Snappy Parquet -> 37,96 MB (Hemat 83,75%, Speedup Baca+Agregasi 15,39x)
      ▼
[3. Split-Before-Fit Feature Engineering (src/feature_engineering.py)]
      │ 1) Hapus kolom bocor/ID (user_id, last_purchase_date, cart_abandonment_rate, checkout_abandonments_per_month)
      │ 2) SPLIT LEBIH DULU (seed=42): Train = 799.723 baris (79,97%) | Test = 200.277 baris (20,03%)
      │ 3) Fit StringIndexer (14 kolom kategorikal) HANYA pada Train Set -> Rakit 57 fitur ('features')
      │ 4) Fit OneHotEncoder + StandardScaler HANYA pada Train Set -> Rakit 132 fitur ('features_lr')
      ▼
[4. Validation-Based Grid Tuning & Ablasi Pohon (src/tune_tree.py)]
      │ Split internal Train: Train-Sub (599.774 baris) vs Validation (199.949 baris)
      │ Uji 72 Konfigurasi Grid (maxDepth 2..12 × minInst 1..100 × gini/entropy)
      │ Latih 7 Model Ablasi (M0..M4) pada 100% Train Set -> Evaluasi 1x pada Test Set
      ▼
[5. Multi-Model Benchmark, Statistik & Scaling (src/compare_models.py)]
      │ Benchmark 6 Model: LR_Tuned, GBT_Tuned, DT_Champion_d10, RF_Tuned, DT_Default_d7, DT_Shallow_d3
      │ Uji McNemar Berpasangan + Bootstrap 95% CI + Kontrol Negatif (AUC ~0,50) + Sensitivitas Threshold
      │ Benchmark Skalabilitas Baris (160k–800k Train Set) & Paralelisme Core (local[1], local[2], local[4])
      ▼
[6. Explainability (SHAP & Permutasi), Aturan IF-THEN, & K-Means Persona]
      │ src/extract_rules_insights.py -> Aturan terjemahan nama kolom asli & Surrogate Fidelity 98,28%
      │ src/explainability_shap.py -> TreeSHAP + Permutation Importance + Joint Collinear Permutation
      │ src/clustering.py -> K-Means (K=2..5, Silhouette K=2: 0,3921) & 3 Persona Operasional (K=3)
```

---

# 4. Kisah Ilmiah di Balik Data: Kontrol Negatif 0,50 vs Target Valid 0,9904

Bagian ini adalah **poin metodologis paling kuat** untuk dipaparkan ke dosen penguji:

### 1. Mengapa `checkout_abandonments_per_month` Menghasilkan ROC-AUC `0,4975` (`~0,50`)?
- Pada eksplorasi awal, kolom `checkout_abandonments_per_month` (skala `0–10`) memiliki korelasi Pearson **$r = -0,000944$** terhadap skor psikologis dan $< 0,002$ terhadap seluruh 57 fitur lainnya.
- Saat diuji sebagai **Eksperimen Kontrol Negatif (`checkout_abandonments_per_month >= 5`)** di `src/compare_models.py`, hasilnya:
  - `LR_Tuned`: **ROC-AUC = `0,4975`**
  - `DT_Champion_d10`: **ROC-AUC = `0,4975`**
  - `RF_Tuned`: **ROC-AUC = `0,5004`**
- **Makna Ilmiah**: Skor `~0,500` ini menjadi **bukti eksperimental bahwa pipeline PySpark kita 100% bebas dari kebocoran data (*zero data leakage*)**. Ketika diberi target acak, pipeline jujur menghasilkan tebakan acak `0,50`, tidak menciptakan akurasi palsu.

### 2. Target Valid (`cart_abandonment_rate >= 40,0%`) & Uji Sensitivitas Threshold
- Sinyal perilaku yang sesungguhnya pada dataset berada pada kolom **`cart_abandonment_rate`** (rata-rata `40,21%`, SD `25,43%`).
- Binarisasi pada ambang **`>= 40,0%`** menghasilkan distribusi kelas yang sangat seimbang:
  - **Kelas `1` (High Abandonment)**: `505.469` baris (**`50,55%`**)
  - **Kelas `0` (Low Abandonment)**: `494.531` baris (**`49,45%`**)
- Kami juga membuktikan lewat **Uji Sensitivitas Threshold (`35%`, `40%`, `45%`)** bahwa performa model tidak bergantung semu pada angka `40%`: pada ambang `35%` maupun `45%`, ROC-AUC tetap kokoh di **`0,9882 – 0,9905`**.

---

# 5. Bedah Korelasi, Kolinearitas 4 Fitur Psikologis, & Explainable AI (SHAP)

### 1. Klaster 4 Variabel Psikologis yang Saling Kolinear
Audit korelasi Pearson pada `1.000.000` baris (`reports/metrics/schema_and_correlation_audit.csv`) mengungkap **4 variabel psikologis utama** yang berkorelasi sangat kuat terhadap `cart_abandonment_rate`:

| Peringkat | Nama Fitur | Korelasi Pearson ($r$) ke Target | Single-Tree Gini (`DT_d10`) | Ensemble Gini (`RF` 100 Pohon Audit) | Mean $\|\text{SHAP}\|$ (`RF`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **`impulse_buying_score`** | **`+0,9615`** | **`95,00%`** | **`35,14%`** | **`0,1804`** |
| **2** | **`stress_from_financial_decisions`** | **`+0,9404`** | **`4,72%`** | **`31,15%`** | **`0,1468`** |
| **3** | **`overall_stress_level`** | **`+0,8635`** | `0,002%` | **`17,08%`** | **`0,0523`** |
| **4** | **`impulse_purchases_per_month`** | **`+0,8501`** | `0,001%` | **`14,04%`** | **`0,0528`** |
| — | *53 Fitur Lainnya (Demografi, Sesi, Spend)* | $\|r\| < 0,003$ | `0,28%` | `2,59%` | $< 0,005$ |

### 2. Mengapa `overall_stress_level` ($r=+0,8635$) Bobot Gini-nya `0,002%` di Pohon Tunggal? (*Greedy Collinear Masking*)
Jika dosen bertanya: *"Kenapa `overall_stress_level` korelasinya `+0,8635` tapi di Decision Tree Feature Importance hampir nol?"*, jawab dengan tegas:
- Algoritma Decision Tree tunggal bekerja secara **greedy top-down**. Karena ke-4 fitur psikologis tersebut saling berkorelasi ekstrem ($r = 0,85 - 0,96$), begitu pohon membelah menggunakan `impulse_buying_score` di akar dan `stress_from_financial_decisions` di lapis kedua, informasi dari `overall_stress_level` dan `impulse_purchases_per_month` **sudah terwakili sepenuhnya (*redundant/masked*)**.
- Bukti ilmiahnya kami tunjukkan pada **Random Forest (`RF_Tuned`)** dan **TreeSHAP** (`src/explainability_shap.py`): ketika tiap pohon di RF dipaksa mengambil subset fitur acak (`featureSubsetStrategy="sqrt"`), keempat fitur langsung berbagi bobot secara proporsional (**`35,14%`, `31,15%`, `17,08%`, dan `14,04%`**).
- Saat ke-4 fitur ini diacak bersama (*Joint Cluster Permutation*), ROC-AUC seluruh model langsung jatuh **`~0,48` poin ke `~0,50`**.

---

# 6. Benchmark 6 Model Klasifikasi & Uji Statistik McNemar

Diuji pada **`200.277` baris *unseen test set*** (`src/compare_models.py`):

| Model | Kompleksitas | 5-Fold CV AUC | Test ROC-AUC (95% CI) | PR-AUC | Akurasi | F1-Score | Waktu Latih |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LR_Tuned` (OHE+Scaled)** | 132 bobot | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9902, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9410`** | 18,26s |
| **`GBT_Tuned` (`d=5, 20 iter`)** | 958 node | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9901, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9409`** | 76,01s |
| **`DT_Champion_d10` 🏆** | **473 node** | `0,9883 ± 0,0005` | **`0,9904`** (`[0,9900, 0,9910]`) | **`0,9905`** | **`93,99%`** | **`0,9399`** | **14,42s** |
| **`RF_Tuned` (`40 pohon, d=10`)** | 24.944 node | `0,9894 ± 0,0005` | `0,9894` (`[0,9888, 0,9900]`) | `0,9900` | `93,92%` | `0,9392` | 354,79s |
| **`DT_Default_d7` (`d=7`)** | 41 node | `0,9793 ± 0,0043` | `0,9773` (`[0,9763, 0,9782]`) | `0,9643` | `94,09%` | `0,9409` | 8,34s |
| **`DT_Shallow_d3` (`d=3`)** | **7 node (4 daun)** | `0,9688 ± 0,0064` | `0,9651` (`[0,9639, 0,9664]`) | `0,9524` | `94,06%` | `0,9406` | 8,80s |

### Mengapa Kita Jujur Melaporkan `LR_Tuned` (`0,9906`) Sedikit di Atas `DT_Champion_d10` (`0,9904`)?
- Selisih $\Delta\text{ROC-AUC}$ antara `LR_Tuned` dan `DT_Champion_d10` hanya **`+0,00024` (`0,024%`)**! Karena fungsi pembangkitan data sintetis bersifat aditif mulus pada fitur psikologis, batas linier `LR_Tuned` dan `GBT_Tuned` mencapai langit-langit teoritis data (`~94,10%` akurasi).
- Namun `DT_Champion_d10` meraih performa yang **identik secara praktis (`0,9904` AUC, `93,99%` akurasi)** dengan waktu pelatihan paling cepat (**`14,42 detik`**) dan struktur pohon tunggal yang **100% *white-box***. Bahkan `DT_Champion_d10` **mengungguli `RF_Tuned`** (`0,9904` vs `0,9894`, McNemar $p = 0,0326$) sekaligus **`24,6x` lebih cepat** daripada RF (`354,79 detik`).

---

# 7. Validation Grid Tuning (72 Konfigurasi) & Ablasi Tree Pruning

Untuk mengisolasi pengaruh tiap hyperparameter secara ilmiah (satu variabel berubah setiap langkah), kami melatih **7 konfigurasi ablasi `M0–M4`** di `src/tune_tree.py` (ditampilkan bersama `DT_Baseline_d5` dari `src/train_classifier.py`) pada `799.723` baris Train Set:

| Model Ablasi | Parameter (`maxDepth`, `minInst`, `Impurity`) | Jumlah Node | Test ROC-AUC | Test Akurasi | Kesimpulan Ilmiah Ablasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`M0_Shallow_d3`** | `d=3, min=20, gini` | **7** | `0,9651` | `94,06%` | 4 daun aturan sudah menangkap 98,28% keputusan pohon penuh. |
| **`DT_Baseline_d5`** | `d=5, min=20, gini` | **11** | `0,9318` | `94,09%` | Baseline awal; resolusi ranking probabilitas masih kasar. |
| **`M1_Baseline_d7`** | `d=7, min=20, gini` | **41** | `0,9773` | `94,09%` | Peningkatan tajam ROC-AUC (`+0,0455` dari `d=5`). |
| **`M2_Champion_d10` 🏆** | **`d=10, min=20, gini`** | **473** | **`0,9904`** | **`93,99%`** | **Titik optimal kalibrasi probabilitas (ROC-AUC tertinggi).** |
| **`M3a_d10_gini50`** | `d=10, min=50, gini` | **295** | `0,9835` | `94,00%` | **Ablasi murni `minInst` (`20` $\rightarrow$ `50`)**: memangkas 37,6% node, AUC turun `0,0069`. |
| **`M3b_d10_entropy20`** | `d=10, min=20, entropy` | **227** | `0,9883` | `94,03%` | **Ablasi murni `Impurity` (`gini` vs `entropy`)**: Gini unggul `+0,0021` AUC; Entropy lebih ringkas (`227` node). |
| **`M4_Deep_d12_gini1`** | `d=12, min=1, gini` | **3.431** | `0,9871` | `93,67%` | **Bukti Overfitting Tanpa Pruning**: node meledak `7,25x` lipat (`3.431` node), namun Validation AUC anjlok ke `0,9752` (dari `0,9897` pada `M2`) dan Test Akurasi turun ke `93,67%`! |

---

# 8. Ekstraksi Aturan Bisnis (IF-THEN Rules), Fidelitas 98,28%, & Segmentasi K-Means

### 1. Aturan Eksak 4 Daun (`DT_Shallow_d3`) — Fidelitas `98,28%` terhadap Champion Model
Berikut adalah **4 aturan daun asli** dari pohon ringkas `DT_Shallow_d3` (`depth=3, 7 node`) beserta performa aktualnya pada `200.277` baris Data Uji ([`decision_tree_rules.txt`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/decision_tree_rules.txt) & [`04_BUSINESS_RECOMMENDATIONS.md`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/docs/04_BUSINESS_RECOMMENDATIONS.md)):

1. **Daun 1 — Tier 3 Risiko Rendah / Aman (`90.993` user / `45,43%`, High Abandonment Aktual = `3,67%`)**:  
   `IF impulse_buying_score <= 4,5 THEN LOW ABANDONMENT (0)` *(Intervensi: Jalur **1-Click Express Checkout** tanpa pop-up diskon agar margin profit tidak tergerus)*
2. **Daun 2 — Tier 2 Risiko Ambang / Moderat (`10.943` user / `5,46%`, High Abandonment Aktual = `37,57%`)**:  
   `IF 4,5 < impulse_buying_score <= 5,5 AND stress_from_financial_decisions <= 5,5 THEN LOW ABANDONMENT (0)` *(Intervensi: Lencana **Free Return Guarantee**, Transparansi Ongkir, & pengingat pasca-idle `> 90 detik`)*
3. **Daun 3 — Tier 1a Impulsif Sedang namun Stres Tinggi (`7.220` user / `3,61%`, High Abandonment Aktual = `74,32%`)**:  
   `IF 4,5 < impulse_buying_score <= 5,5 AND stress_from_financial_decisions > 5,5 THEN HIGH ABANDONMENT (1)` *(Intervensi: Dekompresi beban biaya melalui opsi cicilan ringan & jaminan bebas biaya admin)*
4. **Daun 4 — Tier 1b Zona Kritis Impulsif Tinggi (`91.121` user / `45,50%`, High Abandonment Aktual = `97,16%`)**:  
   `IF impulse_buying_score > 5,5 THEN HIGH ABANDONMENT (1)` *(Gabungan Daun 3 + Daun 4 membentuk **Tier 1 Risiko Kritis: `98.341` user / `49,10%`**, risiko batal **`95,49%`** — Intervensi: **Simulasi Cicilan BNPL 0%** langsung di keranjang + tombol **Split-Cart / Simpan di Wishlist**)*

### 2. Segmentasi Persona K-Means Clustering (`src/clustering.py`)
- **`K = 2`**: Silhouette Score tertinggi (**`0,3921`**), membelah populasi menjadi *Low-Risk Mindful* (`50,11%`, rata-rata batal `18,91%`) vs *High-Risk Impulsive* (`49,89%`, rata-rata batal `61,60%`).
- **`K = 3` (Operasional)**:
  - **Persona A (`46,91%` user)**: *Mindful & Low-Stress Shoppers* (Impulsif `2,13`, Stres `2,16`, rata-rata batal `17,57%`, High Abandonment `5,18%`).
  - **Persona B (`26,49%` user)**: *High-Spend Impulsive Shoppers* (Impulsif `7,52`, Stres `7,49`, pengeluaran bulanan tertinggi `$2.506,10`, rata-rata batal `60,08%`, High Abandonment `90,45%`).
  - **Persona C (`26,60%` user)**: *Conversion-Active Impulsive Browsers* (Impulsif `7,55`, Stres `7,53`, konversi historis `50,20%`, rata-rata batal `60,36%`, High Abandonment `90,82%`).

---

# 9. Bank Simulasi Tanya Jawab Sidang Dosen (22 Pertanyaan Kritis & Jawaban Tuntas)

Bagian ini disusun sebagai **perisai lengkap menghadapi pertanyaan paling kritis dari dosen penguji**, dikelompokkan dalam 6 kategori: **(A) Data Sintetis & Batas Bayes `0,99` vs `94%`**, **(B) Arsitektur Big Data & PySpark**, **(C) Metodologi & Anti-Leakage**, **(D) Komparasi Model, Pruning, & Uji Statistik**, **(E) Explainable AI (SHAP, Kolinearitas, Rules, K-Means)**, serta **(F) Implikasi Bisnis, Etika Data, & Teknis**.

---

## Kategori A — Pertanyaan Paling Kritis: Mengapa ROC-AUC `0,99` & Akurasi `94%`? (Data Sintetis)

### ❓ Q1: Kok Hasilnya `0,99` Semua & Akurasi Di Atas `93–94%`? Jangan-Jangan Bocor (*Data Leakage*) atau Mengarang?
> **Dosen**: *"Hampir semua model kalian ROC-AUC-nya `0,99` dan akurasinya `94%`. Ini terlalu bagus! Jangan-jangan ada kebocoran data?"*  
> **Jawaban Lengkap**:  
> *"Justru kami bisa membuktikan secara matematis **mengapa akurasi mentok di `94,10%` sementara ROC-AUC `0,9904–0,9906`**, dan membuktikan itu **bukan kebocoran**, melainkan konsekuensi langsung dari **rumus pembangkitan data sintetisnya**:*  
> *1. **Bukti Struktur Matematis Data (Regresi OLS pada 1 Juta Baris)**: Saat diaudit, kolom target `cart_abandonment_rate` pada dataset Kaggle ini dibangkitkan mengikuti fungsi aditif mulus $\approx 0{,}75 + 4{,}92 \cdot \text{impulse\_buying\_score} + 2{,}98 \cdot \text{stress\_from\_financial\_decisions} + \varepsilon$ ($R^2 = 0{,}9464$), dengan derau acak $\varepsilon$ bersimpangan baku $\sigma \approx 5{,}99$ poin lalu di-clip `[0, 90]`.*  
> *2. **Batas Teoritis Bayes Optimal = `94,10%` Akurasi / `0,9907` AUC**: Sekitar **`14,54%` pengguna** berada di zona abu-abu ($\pm 6$ poin dari threshold `40%`), sehingga derau acak $\sigma \approx 6$ membalik label mereka ke atas/bawah `40%` secara acak (*irreducible Bayes error* `~5,90%`). Itulah sebabnya **semua model terbaik sekalipun mentok tepat di Akurasi `93,99%–94,10%`** (bukan `99%–100%` seperti kalau ada data leakage!).*  
> *3. **Kenapa ROC-AUC Bisa `0,99` Padahal Akurasi `94%`?**: Akurasi hanya melihat satu titik potong (`40%`), sedangkan ROC-AUC mengukur peringkat probabilitas di seluruh populasi (`0%` s.d. `90%`). Karena rentang sinyal (`0–90`) jauh lebih besar daripada derau ($\sigma \approx 6$), peringkat pasangan pengguna hampir selalu tepat (`0,9904`), tepat menyentuh batas Bayes (`0,9907`)."*

### ❓ Q2: Kalau Datanya Sintetis, Apa Nilai Ilmiah Proyek Ini? Jangan-Jangan Gak Berguna di Dunia Nyata?
> **Dosen**: *"Dataset Kaggle kalian ini data sintetis. Terus kalau hasilnya tidak bisa diklaim sebagai perilaku konsumen Amazon/Shopify beneran, apa kontribusi penelitiannya?"*  
> **Jawaban**:  
> *"Kami bersikap jujur secara ilmiah: temuan angka pembatalan ini berlaku pada **benchmark data sintetis terkontrol**, dan implikasi bisnisnya butuh validasi A/B test di data produksi nyata. Namun **nilai ilmiah yang 100% valid justru terletak pada rekayasa Big Data dan perbandingan modelnya**:*  
> *1. Membuktikan efisiensi komputasi pipeline **PySpark + Snappy Parquet** pada 1 juta baris (`83,75%` kompresi, `15,39x` speedup I/O, `2,30x` paralelisme CPU).*  
> *2. Membuktikan **trade-off model** (LR vs GBT vs DT vs RF) saat sinyal terkonsentrasi: bagaimana Decision Tree tunggal (`14,42s`) dan pohon 4-daun (`98,28%` fidelitas) menyaingi model kompleks dengan selisih AUC hanya `0,00024`.*  
> *3. Membongkar fenomena **Greedy Collinear Feature Masking** dengan membandingkan Gini Pohon Tunggal vs RF Ensemble vs TreeSHAP vs Permutation Importance."*

### ❓ Q3: Kenapa Target Awal (`checkout_abandonments_per_month`) Gagal ke `0,50` lalu Pindah ke `cart_abandonment_rate`? Apakah Ini *Cherry-Picking* Target?
> **Dosen**: *"Kalian awalnya menargetkan `checkout_abandonments_per_month`, dapat `0,50`, terus ganti ke `cart_abandonment_rate >= 40%`. Bukannya itu pilih-pilih target (*researcher degrees of freedom*)?"*  
> **Jawaban**:  
> *"Betul Pak/Bu, kami mengungkap perubahan target ini secara transparan di dokumentasi dan metode:*  
> *1. Hasil audit korelasi membuktikan `checkout_abandonments_per_month` pada dataset Kaggle ini dibangkitkan sebagai **derau acak murni** ($r = -0{,}000944$ terhadap seluruh fitur), sehingga tidak mungkin dimodelkan oleh algoritma mana pun.*  
> *2. Sebaliknya, kami **tetap mempertahankan `checkout_abandonments_per_month` sebagai Kontrol Negatif (`AUC = 0,4975 – 0,5004`)** untuk membuktikan pipeline kami bebas kebocoran data.*  
> *3. Untuk memastikan pilihan ambang `40,0%` bukan kebetulan semu, kami melakukan **Uji Sensitivitas Threshold (`35%`, `40%`, `45%`)** dan membuktikan ROC-AUC seluruh model tetap kokoh di **`0,9882 – 0,9906`**."*

### ❓ Q4: Kenapa Akurasi Model di `cart_abandonment_rate` Antar Negara (10 Negara), Gender (4), dan Perangkat (3) Sama Semua (~`40,2%`)?
> **Dosen**: *"Saya lihat di grafik EDA kalian, rata-rata pembatalan di Mobile, Desktop, Tablet semuanya ~`40,2%`. Kenapa datar semua?"*  
> **Jawaban**:  
> *"Justru itu adalah **temuan EDA penting** (yang juga terkonfirmasi lewat korelasi Pearson $|r| < 0{,}003$ terhadap 53 fitur non-psikologis): pada dataset ini, demografi (`10` negara, `4` gender), perangkat (`Mobile 60,02%`, `Desktop 29,98%`, `Tablet 9,99%`), pengeluaran bulanan (`~$2.499`), dan durasi sesi (`~60` menit) di-generate secara independen terhadap laju pembatalan. Artinya, **pembatalan bukan dipicu oleh bug UI salah satu perangkat atau asal negara** — namun karena `60,02%` trafik datang dari Mobile, seluruh intervensi UI tetap wajib dirancang **Mobile-First**."*

---

## Kategori B — Arsitektur Big Data, PySpark, & Google Colab

### ❓ Q5: Dataset Kalian 233,60 MB, Kenapa Harus Pakai PySpark dan Parquet? Pakai Pandas Kan Bisa?
> **Dosen**: *"Dataset kalian 233,60 MB masih muat di RAM laptop. Di mana letak Big Data-nya?"*  
> **Jawaban**:  
> *"Ukuran CSV 233,60 MB memang masih muat di RAM 16 GB untuk bacaan tunggal, tetapi **operasi join/One-Hot Encoding matriks 1 juta $\times$ 132 kolom, 72 kali grid search, 5-Fold Cross Validation, dan K-Means** pada Pandas satu core cepat membengkak memorinya dan melambat. Dengan PySpark & Parquet, kami membuktikan tiga metrik Big Data terukur:*  
> *1. **Efisiensi Storage**: CSV (`233,60 MB`) $\rightarrow$ Snappy Parquet (`37,96 MB`), hemat **`83,75%`**.*  
> *2. **Kecepatan I/O Kolumnar**: Benchmark baca + agregasi penuh turun dari **`10,847 detik` (CSV) menjadi `0,705 detik` (Parquet)** — **`15,39x` lebih cepat**.*  
> *3. **Skalabilitas & Portabilitas Cluster**: Waktu latih `DT_Champion_d10` mempercepat **`2,30x`** saat naik dari `local[1]` (`20,68s`) ke `local[2]` (`9,01s`, throughput **`88.799 baris/detik`**), dan kode Spark MLlib ini 100% siap berjalan di cluster multi-node tanpa ubah arsitektur."*

### ❓ Q6: Kenapa Kecepatan `local[4]` (`9,45s`) Tidak 2x Lebih Cepat Dibanding `local[2]` (`9,01s`)?
> **Dosen**: *"Lihat grafik scaling core kalian: dari 1 ke 2 core waktu turun drastis dari `20,68s` ke `9,01s`, tapi di `local[4]` kok malah `9,45s`?"*  
> **Jawaban**:  
> *"Pertanyaan bagus Pak/Bu! Mesin uji yang kami pakai adalah **Intel Core i5-7267U yang hanya punya 2 physical cores (dan 4 logical threads via Hyper-Threading)**. Ketika Spark dipaksa memakai `local[4]`, 4 worker berkompetisi di hanya 2 ALU/FPU fisik yang sama sambil menambah biaya sinkronisasi/shuffle antar-partisi — jadi **saturasi paralelisme terjadi tepat di 2 core fisik** (`speedup 2,30x`). Ini sekaligus membuktikan benchmark kami mengukur hardware riil, bukan angka karangan."*

### ❓ Q7: Apa Masalah Kompatibilitas Model PySpark di Google Colab & Bagaimana Solusinya?
> **Dosen**: *"Kalau saya buka notebook kalian di Google Colab, apa model hasil training laptop lokal ini langsung bisa di-load?"*  
> **Jawaban**:  
> *"Bisa langsung jalan! Sebelumnya muncul error kompatibilitas parameter `pruneTree` karena versi PySpark lokal lebih baru (`4.3.0`) daripada versi Spark di Google Colab (`4.0.x`), di mana Spark baru menulis kunci `pruneTree` di JSON metadata model.*  
> *Solusinya kami membuat fungsi otomatis [`sanitize_spark_model_metadata()`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/pipeline_utils.py#L127-L158) yang membersihkan key `pruneTree` dan file `.crc` dari folder `metadata/`, serta dikunci oleh unit test `test_02_colab_model_metadata_sanitized` agar selalu lolos di Colab."*

### ❓ Q8: Apa Status Lisensi Dataset Kaggle untuk Publikasi & GitHub?
> **Dosen**: *"Kalian menaruh data Parquet turunan di GitHub publik. Apakah lisensi dataset Kaggle-nya memperbolehkan redistribusi?"*  
> **Jawaban**:  
> *"Sudah kami verifikasi langsung pada halaman resmi dataset Kaggle `dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based`: dataset tersebut memakai lisensi **Apache 2.0** (`https://www.apache.org/licenses/LICENSE-2.0`), yang secara sah memperbolehkan pemrosesan dan redistribusi turunan untuk riset akademik dengan menyertakan atribusi resmi di [`README.md`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/README.md)."*

---

## Kategori C — Metodologi, *Feature Engineering*, & Pencegahan *Leakage*

### ❓ Q9: Apa Buktinya *Feature Engineering* Kalian Bebas *Data Leakage*?
> **Dosen**: *"Kadang orang melakukan `StandardScaler` atau `StringIndexer` dulu pada 100% data baru dibagi Train-Test. Kalian bagaimana?"*  
> **Jawaban**:  
> *"Kami menerapkan protokol **Split-Before-Fit (`seed=42`)** di [`src/feature_engineering.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/feature_engineering.py):*  
> *1. Seluruh kolom non-prediktor (`user_id`, `cart_abandonment_rate`, `checkout_abandonments_per_month`, `last_purchase_date`) dibuang dulu.*  
> *2. Data dipecah menjadi **Train (`799.723` baris / `79,97%`)** dan **Test (`200.277` baris / `20,03%`)**.*  
> *3. Fitting `StringIndexer`, `OneHotEncoder`, dan `StandardScaler(withMean=False, withStd=True)` (penskalakan deviasi baku tanpa pemusatan rata-rata untuk menjaga struktur matriks *sparse*) dilakukan **hanya pada Train Set**, baru kemudian `.transform()` diterapkan pada Test Set. Test Set sama sekali tidak ikut memengaruhi parameter penskalakan standar deviasi maupun kamus indeks kategori."*

### ❓ Q10: Mengapa Ada 60 Kolom Mentah, 62 Kolom di Parquet, dan 57 Fitur di Model? Ke Mana Sisa Kolomnya?
> **Dosen**: *"Jelaskan ke mana perginya selisih jumlah kolom tersebut!"*  
> **Jawaban**:  
> *"Rinciannya teraudit 100% di [`feature_metadata.json`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/feature_metadata.json) dan [`01_DATA_DICTIONARY.md`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/docs/01_DATA_DICTIONARY.md):*  
> *- **60 Kolom Mentah** di CSV $\rightarrow$ **62 Kolom di Parquet** (karena ditambah 1 fitur turunan temporal `days_since_last_purchase` dari `last_purchase_date`, plus 1 kolom target `is_high_abandonment`).*  
> *- **57 Fitur Prediktor Independen** = `43` fitur numerik + `14` fitur kategorikal — berasal dari `62` kolom dikurangi **5 kolom non-prediktor** (`user_id`, `last_purchase_date`, `cart_abandonment_rate`, `checkout_abandonments_per_month`, dan target `is_high_abandonment`).*  
> *- Untuk **Logistic Regression**, 14 fitur kategorikal tersebut di-One-Hot Encode menjadi 89 kolom biner + 43 numerik = **132 dimensi `features_lr`** yang distandarisasi, sedangkan model pohon memakai **57 dimensi `features`** (`maxBins=32` sudah aman karena kardinalitas maksimum per kategori hanya `10`)."*

### ❓ Q11: Kenapa `purchase_conversion_rate` dan `return_rate` Tetap Dimasukkan? Bukannya Itu Berisiko Bocor?
> **Dosen**: *"Kolom `purchase_conversion_rate` dan `return_rate` itu kedengarannya fitur agregat historis pasca-transaksi. Apakah tidak membocorkan label?"*  
> **Jawaban**:  
> *"Kami sudah mengaudit korelasi seluruh 60 kolom terhadap target di [`schema_and_correlation_audit.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/schema_and_correlation_audit.csv): nilai korelasi Pearson `purchase_conversion_rate` hanya **$r = +0{,}0007$** dan `return_rate` hanya **$r = -0{,}0005$**, sementara **Permutation $\Delta\text{AUC}$ dan SHAP keduanya $\approx 0{,}000$**. Artinya keduanya murni profil historis akun yang ortogonal terhadap `cart_abandonment_rate` dan tidak membocorkan informasi target sama sekali."*

### ❓ Q12: Bagaimana Penanganan *Missing Value*, *Outlier*, dan *Class Imbalance*?
> **Dosen**: *"Bagaimana kalian menangani data kosong (*missing value*), outlier, dan ketidakseimbangan kelas pada 1 juta baris?"*  
> **Jawaban**:  
> *"- **Missing Values**: Skrip `eda_profiling.py` memeriksa seluruh 60 kolom dan mengonfirmasi **0 *missing values*** di semua kolom (`100%` lengkap).*  
> *- **Class Imbalance**: Pada threshold `cart_abandonment_rate >= 40,0%`, proporsi kelas sangat seimbang alami: **Kelas `1` = `505.469` (`50,55%`) vs Kelas `0` = `494.531` (`49,45%`)**, sehingga tidak diperlukan teknik sintetis seperti SMOTE/undersampling.*  
> *- **Outlier**: Seluruh fitur psikologis berskala terikat (`0–10`) dan fitur finansial berada pada batas rasional; untuk model linier (`LR`), kami menerapkan `StandardScaler(withMean=False, withStd=True)` pada vektor fitur guna menjaga stabilitas konvergensi gradien sambil tetap mempertahankan sparsitas vektor One-Hot Encoding."*

---

## Kategori D — Komparasi Model, *Grid Tuning*, & Uji Statistik

### ❓ Q13: Di Tabel Kalian, `LR_Tuned` (`0,9906`) dan `GBT_Tuned` (`0,9906`) Lebih Tinggi daripada `DT_Champion_d10` (`0,9904`). Kenapa Tetap Pakai Decision Tree?
> **Dosen**: *"Kenapa pilih Decision Tree kalau Logistic Regression dan GBT ROC-AUC-nya lebih tinggi?"*  
> **Jawaban**:  
> *"Selisih $\Delta\text{ROC-AUC}$ antara `LR_Tuned`/`GBT_Tuned` (`0,9906`) dan `DT_Champion_d10` (`0,9904`) hanyalah **`0,00024` (`0,024%`)** (`Bootstrap 95% CI [+0,00010, +0,00040]`). Secara praktis performa prediksinya setara karena sama-sama menabrak batas Bayes `0,9907`, namun `DT_Champion_d10` memberikan **keunggulan telak di dua sumbu lain**:*  
> *1. **Kecepatan Pelatihan Tercepat (`14,42 detik`)**: lebih cepat dari `LR` (`18,26s`), **`5,27x` lebih cepat dari `GBT` (`76,01s`)**, dan **`24,6x` lebih cepat dari `RF` (`354,79s`)**.*  
> *2. **100% Transparan (*White-Box*)**: menghasilkan aturan logika `IF-THEN` eksplisit yang tidak dimiliki GBT maupun RF."*

### ❓ Q14: Kok Bisa Pohon Tunggal `DT_Champion_d10` (`0,9904`) Mengalahkan Ensemble `Random Forest` (`0,9894`)?
> **Dosen**: *"Secara teori Random Forest adalah ensemble dari banyak Decision Tree dan biasanya lebih akurat dari pohon tunggal. Kenapa di eksperimen kalian `DT_Champion_d10` (`0,9904`, `93,99%`) justru mengalahkan `RF_Tuned` (`0,9894`, `93,92%`)?"*  
> **Jawaban**:  
> *"Ini temuan menarik dan sangat masuk akal secara algoritmik:*  
> *Di PySpark `RandomForestClassifier`, setiap simpul pembelahan secara default memakai `featureSubsetStrategy='sqrt'` — artinya dari 57 fitur, tiap split hanya boleh mengintip $\sqrt{57} \approx 8$ fitur acak! Karena pada dataset ini sinyal terkuat sangat terkonsentrasi di klaster 4 fitur psikologis (dan 53 fitur lainnya hanyalah derau ortogonal), pada sebagian besar split di dalam 40 pohon RF, ke-4 fitur psikologis tersebut **tidak terpilih dalam sampel acak 8 fitur** sehingga pohon dipaksa membelah memakai fitur acak (noise). Akibatnya agregasi 40 pohon (`24.944` node) justru sedikit tercemar variansi dari 53 fitur noise dibanding 1 pohon `DT_Champion_d10` (`473` node) yang selalu bisa melihat semua 57 fitur sekaligus."*

### ❓ Q15: Uji McNemar Menghasilkan $p = 4,23 \times 10^{-6}$ antara LR dan DT — Bukannya Itu Artinya LR Jauh Lebih Unggul?
> **Dosen**: *"Lihat hasil uji McNemar kalian: `p-value = 4,23e-6 < 0,05`. Berarti perbedaan LR dan DT sangat signifikan dong?"*  
> **Jawaban**:  
> *"Secara **statistik (*statistical significance*)** ya ($p < 0,001$), tetapi secara **praktis (*practical significance*)** perbedaannya amat kecil! Pada ukuran data uji raksasa ($N = 200.277$), selisih error diskordan hanya **216 baris dari 200.277** ($b=1.202$ vs $c=986$, sekitar `0,11%` dari populasi) sudah cukup membuat $\chi^2 = 21{,}13$ signifikan. Karena itu kami tidak berhenti di p-value, melainkan wajib melaporkan **Effect Size $\Delta\text{AUC} = +0,00024$ (`95% Bootstrap CI [+0,00013, +0,00037]`)**."*

### ❓ Q16: Bagaimana Proses *Hyperparameter Tuning* Agar Tidak *Overfitting* dan Terisolasi Antar-Parameter?
> **Dosen**: *"Kenapa dulu kalian membandingkan `M3` yang ubah dua parameter sekaligus? Sekarang gimana caranya?"*  
> **Jawaban**:  
> *"Sudah kami perbaiki total di [`src/tune_tree.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/tune_tree.py):*  
> *1. Seluruh **72 konfigurasi** (`maxDepth` $2..12$ $\times$ `minInstancesPerNode` $\{1,20,50,100\}$ $\times$ `impurity` $\{\text{gini},\text{entropy}\}$) diuji **murni pada Validation Split (`199.949` baris)** di dalam Train Set, tanpa menyentuh Test Set.*  
> *2. Efek tiap parameter diisolasi satu-per-satu (`M2 vs M3a` mengisolasi `minInst` $20 \rightarrow 50$ yang memangkas `37,6%` node; `M2 vs M3b` mengisolasi `gini vs entropy` pada `maxDepth=10, minInst=20`).*  
> *3. Pohon `M4` tanpa pruning (`maxDepth=12, minInst=1`) terbukti mengalami **overfitting**: jumlah node meledak `7,25x` (`473` $\rightarrow$ `3.431` node), namun **Validation AUC anjlok ke `0,9752` (dari puncak `0,9897` pada `M2`) dan Test Akurasi turun ke `93,67%`**."*

### ❓ Q17: Kenapa Tidak Pakai Deep Learning (MLP / Neural Network)?
> **Dosen**: *"Datanya 1 juta baris, kenapa tidak pakai Deep Learning?"*  
> **Jawaban**:  
> *"Pertama, data kita adalah **tabular terstruktur** dengan sinyal yang batas Bayes-nya sudah `0,9907` AUC — sudah dicapai oleh `LR` (`0,9906`), `GBT` (`0,9906`), dan `DT` (`0,9904`), sehingga arsitektur Deep Learning tidak mungkin menaikkan akurasi di atas batas Bayes. Kedua, Deep Learning bersifat **black-box total** dan biaya pelatihannya jauh lebih tinggi, bertentangan dengan tujuan kita menghasilkan aturan **white-box IF-THEN** yang ringkas dan dapat diaudit."*

---

## Kategori E — Explainable AI (SHAP, Kolinearitas, Aturan IF-THEN, & K-Means)

### ❓ Q18: Kenapa `overall_stress_level` ($r=+0,8635$) Bobot Gini-nya Cuma `0,002%` di Decision Tree?
> **Dosen**: *"Di heatmap korelasinya `+0,86`, tapi kok di Feature Importance Decision Tree nyaris `0%`?"*  
> **Jawaban**:  
> *"Ini fenomena klasik bernama **Greedy Collinear Feature Masking**! Keempat fitur psikologis tersebut saling berkorelasi ekstrem satu sama lain ($r = 0,84 – 0,92$). Begitu pohon keputusan tunggal membelah di akar menggunakan `impulse_buying_score` (`95,00%`) dan `stress_from_financial_decisions` (`4,72%`), sebagian besar pengurangan impuritas yang bisa diberikan oleh `overall_stress_level` dan `impulse_purchases_per_month` **sudah terwakili**, sehingga pohon greedy tidak memilihnya lagi.*  
> *Kami membuktikannya lewat [`src/explainability_shap.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/explainability_shap.py): pada **Random Forest Gini**, keempat fitur langsung terbagi merata (`35,14%`, `31,15%`, `17,08%`, `14,04%`) disertai nilai **TreeSHAP positif**; dan saat **ke-4 fitur ini diacak bersama (*Joint Cluster Permutation*)**, ROC-AUC seluruh model jatuh **`-0,4835` s.d. `-0,4899` ke `~0,50` (acak murni)**."*

### ❓ Q19: Champion Tree Kalian Punya `473` Node! Mana Mungkin Manusia Baca `473` Cabang?!
> **Dosen**: *"Katanya Decision Tree itu white-box, tapi pohon champion kalian ada `473` node. Bagaimana tim bisnis membacanya?"*  
> **Jawaban**:  
> *"Tepat sekali Pak/Bu! Untuk operasional bisnis, kami menyarikan **Surrogate Shallow Tree (`DT_Shallow_d3`, `maxDepth=3`, `7` node)** yang hanya terdiri dari **4 daun aturan IF-THEN** menggunakan ambang utama (`impulse_buying_score <= 4,5`, `<= 5,5`, dan `stress_from_financial_decisions <= 5,5`). Luar biasanya, di [`src/extract_rules_insights.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/extract_rules_insights.py) terbukti bahwa **pohon 4 daun ini menyamai `98,28%` (`196.825 / 200.277`) keputusan pohon penuh 473 node**, dengan akurasi Test Set mencapai **`94,06%`**!"*

### ❓ Q20: Di K-Means Clustering (`K=2..5`), Silhouette Tertinggi Ada di `K=2` (`0,3921`). Kenapa Kalian Pakai `K=3` untuk Persona?
> **Dosen**: *"Kenapa pakai 3 persona jika skor Silhouette terbaik justru di `K=2` (`0,3921` vs `0,2653` pada `K=3`)?"*  
> **Jawaban**:  
> *"Kami melaporkan keduanya secara jujur di [`kmeans_segmentation.json`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/kmeans_segmentation.json):*  
> *- **`K=2`** menghasilkan pemisahan geometris paling bersih (`Silhouette = 0,3921`): membagi populasi hampir simetris menjadi *Low-Risk* (`50,11%`, rata-rata batal `18,91%`) dan *High-Risk* (`49,89%`, rata-rata batal `61,60%`).*  
> *- **`K=3`** (`Silhouette = 0,2653`) digunakan sebagai **segmentasi operasional opsional** yang membelah populasi berisiko tinggi berdasarkan perilaku belanja sekunder menjadi **Persona B (*High-Spend Impulsive Shoppers*, `26,49%`, belanja bulanan `$2.506,10`, High Abandonment `90,45%`)** dan **Persona C (*Conversion-Active Impulsive Browsers*, `26,60%`, High Abandonment `90,82%`)**, sambil tetap berpedoman pada **3 Tier Operasional** dari aturan 4-daun Decision Tree untuk penargetan intervensi berdasarkan tingkat risiko (`3,67%`, `37,57%`, dan `95,49%`)."*

---

## Kategori F — Trade-Off Bisnis, Etika, & Kualitas Rekayasa Perangkat Lunak

### ❓ Q21: Dari Confusion Matrix, Ada `5.777` False Positive & `6.264` False Negative. Mana yang Lebih Berbahaya bagi Bisnis E-Commerce?
> **Dosen**: *"Di data uji ada `5.777` FP dan `6.264` FN. Apa dampak bisnisnya?"*  
> **Jawaban**:  
> *"Pada skenario ini **False Negative (`6.264` orang = `3,13%`)** lebih merugikan secara *revenue*: mereka sebenarnya hampir membatalkan transaksi, tetapi gagal terdeteksi oleh sistem sehingga tidak mendapat bantuan cicilan/split-cart dan transaksi batal sepenuhnya (`loss of sale`). Sedangkan **False Positive (`5.777` orang = `2,88%`)** hanya berarti kita menampilkan opsi BNPL atau ringkasan harga transparan kepada pengguna yang sebenarnya sudah mantap membeli — biayanya nyaris nol. Karena itu, jika bisnis ingin menekan FN lebih lanjut, platform dapat menggeser ambang probabilitas (*decision threshold*) dari `0,50` ke `0,40`."*

### ❓ Q22: Bagaimana Kalian Menjamin Seluruh Angka di Kode, Grafik, Notebook, & Laporan Tidak Saling Bertentangan?
> **Dosen**: *"Proyek dengan banyak file Python dan dokumen sering angka di README beda sama grafik dan notebook. Bagaimana kalian mencegahnya?"*  
> **Jawaban**:  
> *"Kami menerapkan arsitektur **Single Source of Truth (`reports/metrics/*.json`)** dan **5 Suite Automated Unit & Integration Tests** di [`tests/test_pipeline_integrity.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/tests/test_pipeline_integrity.py):*  
> *1. Tidak ada lagi angka hardcoded di skrip plot, Looker CSV, maupun dashboard HTML — semuanya membaca JSON metrik hasil eksekusi PySpark terbaru.*  
> *2. Test otomatis memverifikasi anti-kebocoran `Split-Before-Fit`, penghapusan `pruneTree`, terjemahan aturan `decision_tree_rules.txt`, keberadaan 12 grafik + 5 Looker CSV, dan konsistensi angka di seluruh dokumen `.md` — dan ke-5 test tersebut **lulus 100%**."*

---

# 10. Kamus Istilah Teknis (Glosarium)
- **Split-Before-Fit**: Prinsip mengisolasi data uji sebelum menghitung statistik *transformer* (`StringIndexer`/`StandardScaler`) agar distribusi data uji tidak bocor ke proses pelatihan.
- **Surrogate Tree Fidelity**: Persentase kesepakatan prediksi antara pohon keputusan dangkal yang mudah dibaca manusia (`maxDepth=3`, `4` daun) terhadap model penuh (`maxDepth=10`, `473` node), yaitu **`98,28%`**.
- **Greedy Collinear Feature Masking**: Fenomena di mana pohon keputusan tunggal hanya memilih satu fitur terkuat dari sekelompok fitur yang saling berkorelasi tinggi (`impulse_buying_score` `95,00%`), sehingga fitur kolinear lainnya tampak memiliki bobot Gini mendekati nol padahal terbukti penting saat diuji dengan Random Forest dan TreeSHAP.
- **McNemar's Test**: Uji statistik non-parametrik ($\chi^2$, $df=1$) pada sampel berpasangan untuk menguji apakah perbedaan jumlah kesalahan klasifikasi (*discordant errors*) antara dua model signifikan secara statistik.

---

# 11. Panduan Menjalankan di Google Colab (*Plug & Play*)
Notebook [`notebooks/01_e_commerce_abandonment_showcase.ipynb`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/notebooks/01_e_commerce_abandonment_showcase.ipynb) telah dilengkapi deteksi otomatis lingkungan Google Colab dan sanitasi metadata `pruneTree` (`src/pipeline_utils.py`), sehingga model yang dilatih di PySpark 4.3.0 lokal dapat dimuat langsung di Google Colab tanpa error kompatibilitas versi Spark.

---

# 12. Panduan Lengkap 12 Grafik Visualisasi & Cara Membacanya

Seluruh **12 grafik beresolusi tinggi (300 DPI)** tersedia di folder `reports/` (6 grafik lama yang sudah diperbaiki + 6 grafik ilmiah baru). Berikut panduan cara membaca dan menjelaskannya:

### Kelompok A — 6 Grafik Eksplorasi Data (EDA) & Evaluasi Dasar (Sudah Diperbaiki Total)
1. **`reports/eda_stress_vs_abandonment.png`** *(Diperbaiki)*:
   - **Dulu**: Salah mem-plot kolom noise `checkout_abandonments_per_month` (garis lurus datar `~5,00`).
   - **Sekarang**: Menampilkan 2 panel berdampingan: Panel kiri menunjukkan lonjakan eksponensial **`cart_abandonment_rate`** dari **`4,71%` (stres `0–1`) menjadi `77,03%` (stres `10`)** seiring naiknya `impulse_buying_score`. Panel kanan menampilkan **Kontrol Negatif** `checkout_abandonments_per_month` yang datar di `~5,00` ($r = -0,0009$).
2. **`reports/eda_device_vs_abandonment.png`** *(Diperbaiki)*:
   - **Isi**: Distribusi `1.000.000` pengguna per perangkat (**Mobile `60,02%`**, **Desktop `29,98%`**, **Tablet `9,99%`**) beserta rata-rata `cart_abandonment_rate` (`~40,2%` merata di semua perangkat).
   - **Poin Presentasi**: Membuktikan bahwa pembatalan keranjang **bukan disebabkan bug teknis di satu jenis layar**, namun karena mayoritas (60%) trafik berasal dari Mobile, maka intervensi UI/UX wajib berfokus pada layar *mobile*.
3. **`reports/correlation_heatmap.png`** *(Diperbarui)*:
   - **Isi**: Matriks korelasi Pearson yang menampilkan **klaster 4 fitur psikologis kolinear** (`impulse_buying_score` $r=+0,9615$, `stress_from_financial_decisions` $r=+0,9404$, `overall_stress_level` $r=+0,8635$, `impulse_purchases_per_month` $r=+0,8501$) vs variabel finansial/sesi yang independen ($|r| < 0,003$).
4. **`reports/confusion_matrix.png`** *(Diperbarui dengan Angka Eksak)*:
   - **Isi**: Matriks konfusi `DT_Champion_d10` pada `200.277` baris data uji: **`TN = 93.146` (`46,51%`)**, **`FP = 5.777` (`2,88%`)**, **`FN = 6.264` (`3,13%`)**, **`TP = 95.090` (`47,48%`)** — Akurasi **`93,99%`**, ROC-AUC **`0,9904`**.
5. **`reports/decision_tree_feature_importance.png`** *(Diperbarui)*:
   - **Isi**: Kontribusi penurunan impuritas Gini pada pohon tunggal `DT_Champion_d10`: `impulse_buying_score` (**`95,00%`**) dan `stress_from_financial_decisions` (**`4,72%`**).
6. **`reports/customer_persona_comparison.png`** *(Diperbarui)*:
   - **Isi**: Komparasi profil **High Abandonment (`505.469` user)** vs **Low Abandonment (`494.531` user)**: skor impulsif (`7,68` vs `2,26`) dan stres finansial (`7,59` vs `2,35`) kontras tajam, sedangkan `monthly_spend` (`$2.496,58` vs `$2.501,02`) dan `daily_session_time_minutes` (`60,01` vs `60,02` menit) identik.

### Kelompok B — 6 Grafik Ilmiah Baru (Untuk Evaluasi Model, Jurnal JIKO, & Pertanyaan Kritis Dosen)
7. **`reports/model_roc_pr_curves.png`** *(BARU)*:
   - **Isi**: Panel A menampilkan kurva **ROC** dan Panel B menampilkan kurva **Precision-Recall** untuk keenam model pada `200.277` data uji (`LR_Tuned 0,9906`, `GBT_Tuned 0,9906`, `DT_Champion_d10 0,9904`, `RF_Tuned 0,9894`, `DT_Default_d7 0,9773`, `DT_Shallow_d3 0,9651`).
   - **Poin Presentasi**: Kurva `DT_Champion_d10` berhimpit hampir sempurna dengan `LR_Tuned` dan `GBT_Tuned`, bahkan berada di atas `RF_Tuned`.
8. **`reports/dt_pruning_depth_curve.png`** *(BARU)*:
   - **Isi**: Hasil 72 konfigurasi pada Validation Set (`maxDepth 2..12`): Panel A menunjukkan ROC-AUC dan F1-Score (`Gini` vs `Entropy`), sedangkan Panel B menunjukkan ledakan jumlah node hingga **`2.000+` node (dan `3.431` node pada Full Train `M4`)** saat `maxDepth=12, minInstancesPerNode=1`.
   - **Poin Presentasi**: Bukti visual mengapa kita berhenti di `maxDepth=10, minInstancesPerNode=20` (`473` node) dan mengapa *pre-pruning* wajib dilakukan.
9. **`reports/pyspark_scaling_benchmark.png`** *(BARU)*:
   - **Isi**: Panel A menampilkan waktu latih vs volume data (`20%–100%` Train Set / `160.127` s.d. `799.723` baris) untuk `DT`, `LR`, `RF (40 trees)`, dan `GBT`. Panel B menampilkan percepatan paralelisme CPU (`local[1]` `20,68s` $\rightarrow$ `local[2]` `9,01s`, **speedup `2,30x`**). Panel C membandingkan ukuran disk & I/O CSV vs Snappy Parquet (**hemat `83,8%` ruang, I/O `15,4x` lebih cepat**).
10. **`reports/kmeans_persona_clusters.png`** *(BARU)*:
    - **Isi**: Panel A menampilkan kurva **Silhouette Score & WSSSE** (`K=2..5`, puncak Silhouette `0,3921` pada `K=2`), dan Panel B menampilkan profil karakteristik 3 Persona Pelanggan operasional (`K=3`).
11. **`reports/shap_summary_plot.png`** *(BARU)*:
    - **Isi**: Panel A membandingkan Mean $|\text{SHAP}|$ antara Pohon Tunggal vs Ensemble Random Forest; Panel B menampilkan *beeswarm summary plot* nilai SHAP lokal (warna merah = nilai fitur tinggi mendorong probabilitas pembatalan ke kanan positif).
12. **`reports/permutation_vs_gini_importance.png`** *(BARU)*:
    - **Isi**: Panel A membandingkan 3 metode bobot internal (*Single-Tree Gini*, *RF Ensemble Gini*, *LR Standardized Weight*); Panel B menampilkan **Permutation Importance ($\Delta\text{ROC-AUC}$)** saat tiap fitur diacak sendiri vs saat **ke-4 fitur psikologis kolinear diacak bersama (`[JOINT 4-FEATURE CLUSTER]`, AUC langsung jatuh `~0,48` poin)**.

---

# 13. Apa Saja yang Berubah & Ditambahkan? (Tabel Versi Lama vs Versi Baru)

Tabel ini merangkum seluruh perubahan dari versi lama proyek ke versi baru yang telah diperkeras (*hardened*):

| Aspek Proyek | Versi Lama (Sebelum Hardening) | Versi Baru (Sekarang — 100% Terverifikasi) |
| :--- | :--- | :--- |
| **Jumlah Kolom & Fitur** | Tertulis 57 kolom / 53 fitur (`4` kolom terlewat tanpa alasan; `income_level` salah ditulis kategorikal). | **60 kolom mentah $\rightarrow$ 62 kolom Parquet $\rightarrow$ 57 fitur prediktor** (`43` numerik termasuk `days_since_last_purchase` + `14` kategorikal). `features` = 57 dim, `features_lr` = 132 dim. |
| **Isolasi Data (*Leakage*)** | `StringIndexer` di-fit pada 1 juta baris *sebelum* `randomSplit`. | **Split-Before-Fit (`seed=42`)**: Train (`799.723`) dan Test (`200.277`) dipisah lebih dulu sebelum fitting `StringIndexer`, `OneHotEncoder`, dan `StandardScaler`. |
| **Cara Tuning Decision Tree** | Hanya 3 model (`M1, M2, M3`), dipilih dengan mengintip Test Set, dan `M3` mengubah 2 parameter sekaligus (`minInst` + `impurity`). | **72 konfigurasi grid** pada Validation Split (`199.949` baris) + **7 konfigurasi ablasi terisolasi (`M0, M1, M2, M3a, M3b, M3c, M4`)** + baseline `d=5` yang memisahkan efek `maxDepth`, `minInstancesPerNode`, dan `gini` vs `entropy`. |
| **Perbandingan Model** | 3 model (`LR`, `DT d=7`, `RF d=7`); `LR` diberi input indeks integer mentah (tidak adil). | **6 model setara** (`LR_Tuned` OHE+Scaled `0,9906`, `GBT_Tuned` `0,9906`, `DT_Champion_d10` `0,9904`, `RF_Tuned` `0,9894`, `DT_Default_d7` `0,9773`, `DT_Shallow_d3` `0,9651`) + **5-Fold CV**, **Uji McNemar**, & **Bootstrap 95% CI**. |
| **Metrik Champion DT (`M2`)** | ROC-AUC `0,9832`, Akurasi `93,95%`, `509` node, angka `FP/FN` beda-beda antar dokumen (`5.929` vs `7.170`). | **ROC-AUC `0,9904`**, **PR-AUC `0,9905`**, **Akurasi `93,99%`**, **F1 `0,9399`**, **`473` node**, Waktu Latih **`14,42s`**, **`TN=93.146, FP=5.777, FN=6.264, TP=95.090`** (100% konsisten di semua file!). |
| **Aturan IF-THEN (`decision_tree_rules.txt`)** | Masih tertulis `feature 15 <= 4.5` (tanpa nama kolom), tapi di dokumen tertulis manual `impulse > 5.5`. | Otomatis diterjemahkan ke nama asli (`impulse_buying_score [f15] <= 4.5`) + **Surrogate Tree 4 Daun (`DT_Shallow_d3`) dengan fidelitas `98,28%`**. |
| **SHAP & K-Means (`Rule 4`)** | Tertulis di `AGENTS.md` & Charter, tapi **belum ada kodenya** di `src/`. | **Sudah ada kodenya & dijalankan penuh**: [`src/explainability_shap.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/explainability_shap.py) (TreeSHAP + Permutation Importance) & [`src/clustering.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/clustering.py) (K-Means `K=2..5`). |
| **Visualisasi Grafik (`reports/`)** | 6 gambar PNG (2 plot EDA salah pakai kolom noise, beberapa angka di-hardcode). | **12 gambar PNG 300 DPI** (semua dibaca dinamis dari JSON metrik + 6 grafik ilmiah baru untuk evaluasi model & jurnal). |
| **Data Mart Looker Studio** | 5 CSV (`looker_model_feature_importance.csv` dibuat manual, `looker_demografi.csv` dikira 5 negara). | **5 CSV otomatis dari [`src/export_looker_data.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/src/export_looker_data.py)** mencakup **10 negara**, **4 gender**, **3 perangkat**, **6 metode bayar**, dan **10 fitur teratas**. |
| **Unit Testing Otomatis** | Belum ada folder `tests/`. | Ada [`tests/test_pipeline_integrity.py`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/tests/test_pipeline_integrity.py) (5 pengujian otomatis anti-leakage, Colab metadata, aturan pohon, grafik, dan konsistensi dokumen). |

---

# 14. Kronologi Pengerjaan End-to-End dari Awal hingga Akhir (10 Tahap Metodologi)

Gunakan urutan 10 tahap ini untuk menjelaskan **proses kerja tim dari awal hingga akhir** baik di laporan, saat sidang, maupun saat menyusun PPT:

1. **Tahap 1 — Akuisisi & Ingesti Data Skala Besar (`src/download_data.py` & `src/spark_session.py`)**:
   - Mengunduh dataset *E-Commerce Shopper Behavior & Lifestyle* (`1.000.000` baris $\times$ `60` kolom mentah, `233,60 MB` CSV) dan menginisialisasi `SparkSession` (`PySpark 4.3.0`) dengan manajemen memori optimal (`4g` driver memory, Adaptive Query Execution aktif).
2. **Tahap 2 — Audit Skema, Pembersihan, & Konversi Kolumnar Parquet (`src/preprocessing.py`)**:
   - Memverifikasi 0 *missing values* di seluruh 60 kolom mentah.
   - Mengekstrak fitur durasi `days_since_last_purchase` dari kolom tanggal `last_purchase_date`.
   - Membinerkan target klasifikasi `is_high_abandonment` (`1` jika `cart_abandonment_rate >= 40,0%` [`50,55%`], `0` jika `< 40,0%` [`49,45%`]).
   - Menyimpan ke **Apache Parquet ber-kompresi Snappy** (`37,96 MB`, hemat **`83,75%`** storage, dan mempercepat baca+agregasi **`15,39x` lipat** dari `10,847s` menjadi `0,705s`).
3. **Tahap 3 — Exploratory Data Analysis & Penemuan Klaster Kolinear (`src/eda_profiling.py`)**:
   - Menghitung korelasi Pearson seluruh fitur terhadap `cart_abandonment_rate` dan menemukan **4 variabel psikologis yang saling berkolinearitas tinggi**: `impulse_buying_score` ($r=+0,9615$), `stress_from_financial_decisions` ($r=+0,9404$), `overall_stress_level` ($r=+0,8635$), dan `impulse_purchases_per_month` ($r=+0,8501$).
   - Membuktikan bahwa `checkout_abandonments_per_month` adalah derau acak ($r = -0,0009$) sehingga ditetapkan sebagai variabel **Kontrol Negatif**.
4. **Tahap 4 — Rekayasa Fitur Terdistribusi dengan *Split-Before-Fit* (`src/feature_engineering.py`)**:
   - Membuang kolom non-prediktor (`user_id`, `last_purchase_date`, `cart_abandonment_rate`, `checkout_abandonments_per_month`) sehingga tersisa **57 fitur prediktor murni** (`43` numerik + `14` kategorikal).
   - Membagi data **lebih dulu (`seed=42`)** menjadi **Train Set (`799.723` baris / `79,97%`)** dan **Test Set (`200.277` baris / `20,03%`)**.
   - Melakukan *fitting* `StringIndexer` (untuk vektor pohon `features` 57 dimensi) serta `OneHotEncoder` + `StandardScaler` (untuk vektor Regresi Logistik `features_lr` 132 dimensi) **hanya pada Train Set**, lalu mentransformasikannya ke Test Set.
5. **Tahap 5 — Pemodelan Baseline Decision Tree (`src/train_classifier.py`)**:
   - Melatih model awal `DT_Baseline_d5` (`maxDepth=5, minInstancesPerNode=20, gini`, `11` node) yang menghasilkan Test Akurasi `94,09%` namun ROC-AUC baru mencapai `0,9318`.
6. **Tahap 6 — Validation Grid Tuning (72 Konfigurasi) & Ablasi Pruning (`src/tune_tree.py`)**:
   - Membagi Train Set menjadi **Train-Sub (`599.774` baris)** dan **Validation (`199.949` baris)** untuk menguji **72 kombinasi** (`maxDepth 2..12` $\times$ `minInstancesPerNode 1,20,50,100` $\times$ `gini, entropy`).
   - Melatih 7 model referensi ablasi (`M0..M4`) pada 100% Train Set dan menetapkan **`M2_Champion_d10` (`maxDepth=10, minInstancesPerNode=20, gini`, `473` node)** sebagai Champion Model dengan **Test ROC-AUC `0,9904`, PR-AUC `0,9905`, dan Akurasi `93,99%`**.
7. **Tahap 7 — Benchmark Multi-Model, Uji Statistik, Kontrol Negatif, & Scaling (`src/compare_models.py`)**:
   - Membandingkan 6 model (`LR_Tuned`, `GBT_Tuned`, `DT_Champion_d10`, `RF_Tuned`, `DT_Default_d7`, `DT_Shallow_d3`) menggunakan 5-Fold Cross Validation, Uji McNemar berpasangan, dan Bootstrap 95% CI.
   - Menjalankan uji kontrol negatif ($\text{AUC} \approx 0,500$), uji sensitivitas ambang target (`35%`, `40%`, `45%`), serta benchmark skalabilitas baris (`20%–100%` Train Set / `160k–800k` baris) dan core CPU (`local[1]..local[4]`, speedup `2,30x`).
8. **Tahap 8 — Ekstraksi Aturan IF-THEN & Audit Explainable AI (`src/extract_rules_insights.py` & `src/explainability_shap.py`)**:
   - Menerjemahkan pohon keputusan ke nama kolom asli dan membuktikan bahwa pohon 4 daun (`DT_Shallow_d3`) memiliki **fidelitas `98,28%`** terhadap Champion Model.
   - Menjalankan analisis **TreeSHAP**, **Permutation Importance**, dan **Joint Collinear Cluster Permutation** untuk membuktikan kontribusi keempat variabel psikologis yang tertutup oleh *Greedy Collinear Masking* pada pohon tunggal.
9. **Tahap 9 — Segmentasi Pelanggan K-Means Clustering (`src/clustering.py`)**:
   - Mengelompokkan `1.000.000` pengguna menggunakan PySpark K-Means (`K=2..5`, Silhouette terbaik `0,3921` pada `K=2`) dan memetakan **3 Persona Pelanggan operasional (`K=3`)** untuk penargetan intervensi bisnis.
10. **Tahap 10 — Otomatisasi Visualisasi BI, Notebook Showcase, & Pengujian Integritas (`src/export_*.py` & `tests/`)**:
    - Mengekspor 5 CSV Data Mart Looker Studio, membangun dashboard interaktif mandiri (`reports/dashboard_visualisasi.html`), menghasilkan 12 grafik 300 DPI, memperbarui notebook Google Colab (`notebooks/01_e_commerce_abandonment_showcase.ipynb`), dan mengunci kualitas repo dengan 5 unit test otomatis (`tests/test_pipeline_integrity.py`).

---

# 15. Blueprint Slide Presentasi PPT (14 Slide) & Pembagian Peran Tim

Saat membuat slide PowerPoint / Canva untuk sidang akhir atau showcase, ikuti struktur **14 Slide** berikut agar alur cerita sangat tajam, sistematis, dan memukau dosen penguji:

| Slide | Judul Slide | Poin Isi Utama di Slide (Bullet Points Singkat) | Grafik / Tabel yang Dipasang | Pembicara (Saran) |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Judul & Identitas Kelompok** | Analisis Faktor Pendorong *Cart Abandonment* E-Commerce Skala 1 Juta Baris Menggunakan PySpark & Decision Tree Classifier — Kelompok 8 (Anggrek Mekar Cilibende: Galih, Trias, Risha, Raihan). | Logo Kampus + Diagram ringkas ikon Big Data | **Pembicara 1** |
| **2** | **Latar Belakang & Masalah Bisnis** | • Tingginya *cart abandonment* (`40,21%` rata-rata populasi) membuang biaya iklan (CAC).<br>• Industri butuh model yang tidak hanya akurat di skala juta baris, tetapi juga **transparan (*white-box IF-THEN rules*)** untuk intervensi *real-time*. | Angka KPI dari `looker_scorecard_global.csv` | **Pembicara 1** |
| **3** | **Arsitektur Pipeline Big Data (Zero-Leakage)** | • Dataset Kaggle: `1.000.000` baris $\times$ `60` kolom mentah (`233,60 MB` CSV) $\rightarrow$ `0` null.<br>• Kompresi **Apache Parquet Snappy (`37,96 MB`)**: hemat **`83,75%`** storage & **`15,39x`** lebih cepat dibaca (`0,705s` vs `10,847s`).<br>• Isolasi **Split-Before-Fit (`seed=42`)**: `799.723` Train vs `200.277` Test (`57` fitur prediktor murni). | Diagram Alur Bab 3 + Tabel Benchmark CSV vs Parquet | **Pembicara 1** |
| **4** | **Temuan EDA & Bukti Kontrol Negatif** | • Lonjakan pembatalan keranjang dari **`4,71%` (stres `1`) ke `77,03%` (stres `10`)**.<br>• Sebaliknya, `checkout_abandonments_per_month` datar di `~5,00` ($r=-0,0009$) dan menghasilkan **ROC-AUC `0,4975` (`~0,50`)** saat diuji — bukti eksperimental pipeline **100% bebas *data leakage***. | 📊 `reports/eda_stress_vs_abandonment.png` & `reports/eda_device_vs_abandonment.png` | **Pembicara 2** |
| **5** | **Analisis Korelasi & Klaster 4 Variabel Psikologis** | • Ditemukan 4 prediktor psikologis berkolinearitas tinggi terhadap pembatalan: `impulse_buying_score` ($r=+0,9615$), `stress_from_financial_decisions` ($r=+0,9404$), `overall_stress_level` ($r=+0,8635$), `impulse_purchases_per_month` ($r=+0,8501$).<br>• Variabel pengeluaran bulanan (`$2.498,78`) dan durasi sesi (`60,01` mnt) bernilai korelasi $\approx 0$. | 📊 `reports/correlation_heatmap.png` | **Pembicara 2** |
| **6** | **Validation Grid Tuning (72 Konfigurasi) & Ablasi Pruning** | • Pemilihan model dilakukan murni pada Validation Split (`199.949` baris).<br>• Ablasi terisolasi membuktikan: tanpa *pre-pruning* (`M4: d=12, minInst=1`), pohon meledak ke **`3.431` node** dan mengalami *overfitting* (Akurasi turun ke `93,67%`).<br>• Titik optimal: **`M2_Champion_d10` (`d=10, minInst=20, gini`, `473` node)** dengan **Test ROC-AUC `0,9904`**. | 📊 `reports/dt_pruning_depth_curve.png` + Tabel Ablasi `M0..M4` | **Pembicara 2** |
| **7** | **Pertarungan 6 Model Klasifikasi & Uji McNemar** | • Perbandingan adil pada `200.277` baris Test Set: `LR_Tuned` (`0,9906`), `GBT_Tuned` (`0,9906`), **`DT_Champion_d10` (`0,9904`)**, `RF_Tuned` (`0,9894`), `DT_Default_d7` (`0,9773`), `DT_Shallow_d3` (`0,9651`).<br>• Selisih AUC LR vs DT hanya **`0,00024` (`0,024%`)**, sementara DT **mengalahkan RF (`0,9904` vs `0,9894`)** dan **`24,6x` lebih cepat dilatih (`14,42s` vs `354,79s`)**! | 📊 `reports/model_roc_pr_curves.png` + Tabel Benchmark 6 Model | **Pembicara 3** |
| **8** | **Evaluasi Matriks Konfusi & Sensitivitas Threshold** | • Pada `200.277` baris data uji, `DT_Champion_d10` meraih **Akurasi `93,99%` (`188.236` prediksi tepat)**: `TN = 93.146`, `FP = 5.777`, `FN = 6.264`, `TP = 95.090`.<br>• Uji sensitivitas ambang target (`35%`, `40%`, `45%`) membuktikan ROC-AUC stabil di `0,9882 – 0,9904`. | 📊 `reports/confusion_matrix.png` | **Pembicara 3** |
| **9** | **Explainable AI: Mengapa Gini Pohon Tunggal 95% di 1 Fitur?** | • Pada pohon tunggal, `impulse_buying_score` menyerap `95,00%` Gini importance karena efek **Greedy Collinear Feature Masking**.<br>• Melalui **TreeSHAP**, **RF Ensemble Gini**, dan **Joint Cluster Permutation**, terbukti ke-4 variabel psikologis berbagi sinyal (`35,1%`, `31,1%`, `17,1%`, `14,0%` di RF) dan jika ke-4 fitur diacak bersama, AUC jatuh **`~0,48` poin ke `~0,50`**. | 📊 `reports/shap_summary_plot.png` & `reports/permutation_vs_gini_importance.png` | **Pembicara 3** |
| **10** | **Skalabilitas Komputasi PySpark (Baris & Core CPU)** | • Waktu latih `DT_Champion_d10` berskala efisien dari `5,95s` (`20%` Train / `160k` baris) ke `14,42s` (`100%` Train / `800k` baris) — tetap **`24,6x` lebih cepat** dari `RF_Tuned` (`354,79s`).<br>• Paralelisme multi-thread mempercepat pelatihan **`2,30x` lipat** (`20,68s` pada `local[1]` $\rightarrow$ `9,01s` pada `local[2]`, throughput **`88.799 baris/detik`**). | 📊 `reports/pyspark_scaling_benchmark.png` | **Pembicara 3** |
| **11** | **Aturan IF-THEN 4 Daun (`98,28%` Fidelitas) & Persona K-Means** | • Pohon ringkas `DT_Shallow_d3` (hanya **4 aturan daun IF-THEN** dengan ambang `impulse <= 4,5`, `<= 5,5`, dan `stress <= 5,5`) menyamai **`98,28%`** keputusan pohon penuh `473` node!<br>• Segmentasi K-Means (`Silhouette K=2 = 0,3921`, `K=3 = 0,2653`) memetakan 3 Persona: *Mindful Shoppers* (`17,57%` batal) vs *High-Spend Impulsive* (`60,08%` batal) dan *Conversion-Active Impulsive* (`60,36%` batal) yang daya belinya setara (`~$2.499`). | 📊 `reports/kmeans_persona_clusters.png` & `reports/customer_persona_comparison.png` | **Pembicara 4** |
| **12** | **Rekomendasi Intervensi Bisnis (3 Tier Operasional)** | • **Tier 1 Kritis (`49,10%` — `95,49%` high abandon)**: `impulse > 5,5` atau `(impulse==5 & stress > 5,5)` $\rightarrow$ Simulasi **BNPL / Cicilan 0%** di keranjang + **Split-Cart (Simpan untuk Nanti)**.<br>• **Tier 2 Ambang (`5,46%` — `37,57%` high abandon)**: `impulse==5 & stress <= 5,5` $\rightarrow$ Lencana **Free Return** & intervensi anti-idle `> 90s`.<br>• **Tier 3 Aman (`45,43%` — `3,67%` high abandon)**: `impulse <= 4,5` $\rightarrow$ **1-Click Express Checkout** tanpa diskon agresif. | Tabel Matriks Strategi dari `docs/04_BUSINESS_RECOMMENDATIONS.md` | **Pembicara 4** |
| **13** | **Demo Interaktif: Dashboard BI & Google Colab Showcase** | • Demo langsung [`reports/dashboard_visualisasi.html`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/dashboard_visualisasi.html) / Looker Studio (5 Data Mart CSV).<br>• Demo simulator prediksi pelanggan *real-time* di [`notebooks/01_e_commerce_abandonment_showcase.ipynb`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/notebooks/01_e_commerce_abandonment_showcase.ipynb). | Tangkapan layar Dashboard HTML & Simulator Notebook | **Pembicara 4** |
| **14** | **Kesimpulan Utama** | • Pipeline PySpark *Split-Before-Fit* dengan format Snappy Parquet menghemat `83,75%` storage dan mempercepat I/O `15,39x`.<br>• `DT_Champion_d10` meraih **ROC-AUC `0,9904` & Akurasi `93,99%`** dalam **`14,42 detik`**, sedangkan pohon surrogate 4 daun memberikan **`98,28%` fidelitas** yang siap diintegrasikan ke mesin aturan *checkout* e-commerce. | Ringkasan 4 Angka Kunci | **Semua Anggota** |
