# E-Commerce Shopper Behavior & Checkout Abandonment Analysis (Big Data)

Repositori ini memuat implementasi proyek **Teknologi Big Data (Semester 5)** oleh **Kelompok 8 (Anggrek Mekar Cilibende)** untuk menganalisis faktor-faktor pendorong pembatalan transaksi di tahap akhir (*cart/checkout abandonment*) pada platform e-commerce berskala **1.000.000 baris data pengguna (`60` kolom mentah $\rightarrow$ `57` prediktor independen)** menggunakan **Apache Spark (PySpark 4.3.0)** dan **Decision Tree Classifier**.

---

## 👥 Anggota Kelompok (Anggrek Mekar Cilibende)
- **Raden Mas Galih Pradityo** (J0403241030)
- **Trias Aldi Prasetia** (J0403241002)
- **Risha Mufliha Safira** (J0403241094)
- **Raihan Alma Putra** (J0403241075)

---

## ⚡ Showcase Interaktif & Demo Google Colab (*Plug & Play*)

Untuk presentasi ke dosen atau uji coba langsung tanpa instalasi rumit, repositori ini dilengkapi notebook interaktif:  
📁 **[`notebooks/01_e_commerce_abandonment_showcase.ipynb`](notebooks/01_e_commerce_abandonment_showcase.ipynb)**

- **Auto-Detection Multi-Platform**: Berjalan langsung di **VS Code lokal** maupun **[Google Colab](https://colab.research.google.com)** (kompatibel lintas versi Spark melalui sanitasi metadata `pruneTree`).
- **12 Visualisasi Resolusi Publikasi (300 DPI)** di folder `reports/`:
  1. `eda_stress_vs_abandonment.png` — Lonjakan pembatalan keranjang (`4,71%` $\rightarrow$ `77,03%`) vs kontrol negatif (`checkout_abandonments_per_month` datar di `~5,00`).
  2. `eda_device_vs_abandonment.png` — Distribusi pembatalan per perangkat (`Mobile 60,02%`, `Desktop 29,98%`, `Tablet 9,99%`).
  3. `correlation_heatmap.png` — Matriks korelasi Pearson yang mengungkap **klaster 4 variabel psikologis kolinear** (`impulse_buying_score` $r=+0,9615$, `stress_from_financial_decisions` $r=+0,9404$, `overall_stress_level` $r=+0,8635$, `impulse_purchases_per_month` $r=+0,8501$).
  4. `confusion_matrix.png` — Evaluasi Champion Decision Tree pada `200.277` baris *unseen test set* (`TN = 93.146`, `FP = 5.777`, `FN = 6.264`, `TP = 95.090`; Akurasi **`93,99%`**, ROC-AUC **`0,9904`**).
  5. `decision_tree_feature_importance.png` — Peringkat fitur teratas pada pohon tunggal (`impulse_buying_score 95,00%`, `stress_from_financial_decisions 4,72%`).
  6. `customer_persona_comparison.png` — Profil High Abandonment (`7,68` impulsif, `7,59` stres) vs Low Abandonment (`2,26` impulsif, `2,35` stres) dengan pengeluaran bulanan setara (`$2.496,58` vs `$2.501,02`).
  7. `model_roc_pr_curves.png` — Kurva ROC & Precision-Recall komparatif 6 model.
  8. `dt_pruning_depth_curve.png` — Kurva validasi 72 konfigurasi (`maxDepth 2..12`) & bukti *overfitting* pada pohon tanpa pruning (`3.431` node).
  9. `pyspark_scaling_benchmark.png` — Benchmark skalabilitas baris (`20%–100%` Train Set / `160k–800k` baris) & CPU core (`2,30x` speedup).
  10. `kmeans_persona_clusters.png` — Segmentasi K-Means (`K=2..5`, Silhouette terbaik `0,3921`) & 3 Persona Pelanggan.
  11. `shap_summary_plot.png` — Distribusi **TreeSHAP** pada data uji.
  12. `permutation_vs_gini_importance.png` — Komparasi *Single-Tree Gini*, *Ensemble RF Gini*, *LR Standardized Weight*, dan *Permutation Importance* untuk membongkar *Greedy Collinear Feature Masking*.

---

## 🚀 Ringkasan Performa Model & Temuan Kunci

### 1. Benchmark Komparatif 6 Model (PySpark MLlib pada `N = 200.277` Test Set)
Seluruh model dilatih menggunakan protokol **Split-Before-Fit (`seed=42`)** dan pemilihan hyperparameter pada **Validation Split (`199.949` baris)** sebelum diuji satu kali pada **Test Set (`200.277` baris)**:

| No | Model Algoritma | Kompleksitas Struktur | Test ROC-AUC (95% CI) | Test PR-AUC | Test Akurasi | F1-Weighted | Waktu Latih | Peran Evaluasi |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **`LR_Tuned` (OHE + Scaled)** | 132 bobot linier | **`0,9906`** (`[0,9902, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9410`** | 18,26 s | Pembanding Linier |
| 2 | **`GBT_Tuned` (`maxIter=20, d=5`)** | 20 pohon (958 node) | **`0,9906`** (`[0,9901, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9409`** | 76,01 s | Pembanding Boosting |
| 3 | **`DT_Champion_d10` (`d=10, min=20, gini`)** | **1 pohon (473 node)** | **`0,9904`** (`[0,9900, 0,9910]`) | **`0,9905`** | **`93,99%`** | **`0,9399`** | **14,42 s** | **🏆 CHAMPION MODEL** |
| 4 | **`RF_Tuned` (`40 pohon, d=10`)** | 40 pohon (24.944 node) | `0,9894` (`[0,9888, 0,9900]`) | `0,9900` | `93,92%` | `0,9392` | 354,79 s | Pembanding Bagging |
| 5 | **`DT_Default_d7` (`d=7, min=20, gini`)** | 1 pohon (41 node) | `0,9773` (`[0,9763, 0,9782]`) | `0,9643` | `94,09%` | `0,9409` | 8,34 s | Baseline Menengah |
| 6 | **`DT_Shallow_d3` (`d=3, min=20, gini`)** | **1 pohon (7 node / 4 daun)** | `0,9651` (`[0,9639, 0,9664]`) | `0,9524` | `94,06%` | `0,9406` | 8,80 s | **Surrogate Rule Engine (`98,28%` Fidelitas)** |

> **Mengapa `DT_Champion_d10` (bersama Surrogate `DT_Shallow_d3`) Menjadi Pilihan Terbaik?**  
> 1. **Kinerja Setara Model Kompleks ($\Delta\text{ROC-AUC}$ Hanya `0,00024`)**: Selisih ROC-AUC antara `LR_Tuned` (`0,9906`) / `GBT_Tuned` (`0,9906`) dan `DT_Champion_d10` (`0,9904`) hanya **`0,024%`**, bahkan `DT_Champion_d10` mengungguli `RF_Tuned` (`0,9894`, `24.944` node) dengan waktu latih **`24,6x` lebih cepat** (`14,42s` vs `354,79s`).  
> 2. **100% *White-Box* & Aturan 4-Daun Berfidelitas `98,28%`**: Pohon dangkal `DT_Shallow_d3` yang hanya terdiri dari **4 daun aturan (`IF-THEN`)** menyamai **`98,28%` (`196.825 / 200.277`)** keputusan `DT_Champion_d10` dengan akurasi **`94,06%`**.

### 2. Temuan Explainable AI (SHAP, Permutation Importance, & Kolinearitas)
- Pada pohon tunggal `DT_Champion_d10`, penurunan impuritas Gini terkonsentrasi pada `impulse_buying_score` (**`95,00%`**) dan `stress_from_financial_decisions` (**`4,72%`**).
- Melalui audit **TreeSHAP**, **Ensemble RF Gini**, dan **Joint Cluster Permutation** (`src/explainability_shap.py`), dibuktikan bahwa `overall_stress_level` ($r=+0,8635$, RF Gini `17,08%`) dan `impulse_purchases_per_month` ($r=+0,8501$, RF Gini `14,04%`) juga membawa sinyal kuat namun tertutup oleh efek **Greedy Collinear Feature Masking** pada pohon tunggal. Ketika ke-4 fitur psikologis ini diacak bersama, ROC-AUC seluruh model jatuh **`~0,48` poin menuju `~0,50` (acak)**.

---

## 📊 Dashboard Interaktif Google Looker Studio & HTML Mandiri

Skrip `src/export_looker_data.py` menghasilkan 5 berkas CSV Data Mart terverifikasi di `reports/looker_data/`:
1. **`looker_scorecard_global.csv`** (`1` baris): Total User (`1.000.000`), Avg Cart Abandonment (`40,21%`), High Abandonment (`50,55%`), Avg Spend (`$2.498,78`), Avg AOV (`$255,03`), Avg Session (`60,01` menit).
2. **`looker_stres_impulse.csv`** (`11` baris): Kurva `stress_from_financial_decisions` (`0–10`) vs lonjakan pembatalan (`4,71%` $\rightarrow$ `77,03%`).
3. **`looker_model_feature_importance.csv`** (`10` baris): Bobot Feature Importance Champion Decision Tree.
4. **`looker_demografi.csv`** (`120` baris): Dimensi silang **10 negara**, **3 perangkat**, dan **4 kategori gender**.
5. **`looker_metode_bayar.csv`** (`48` baris): Dimensi silang **6 metode bayar** dan **8 kategori produk**.

Panduan langkah demi langkah tersedia di [`docs/06_LOOKER_STUDIO_GUIDE.md`](docs/06_LOOKER_STUDIO_GUIDE.md), dan dashboard mandiri siap pakai dapat dibuka langsung di [`reports/dashboard_visualisasi.html`](reports/dashboard_visualisasi.html).

---

## 📌 Alur Pipeline & Reproducibility

```bash
# 1. Buat virtual environment & pasang dependensi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Unduh dataset 1 juta baris dari Kaggle (233,60 MB)
python src/download_data.py

# 3. Jalankan pipeline end-to-end secara berurutan
python src/preprocessing.py
python src/eda_profiling.py
python src/feature_engineering.py
python src/train_classifier.py
python src/tune_tree.py
python src/extract_rules_insights.py
python src/clustering.py
python src/explainability_shap.py
python src/compare_models.py
python src/export_looker_data.py
python src/export_report_plots.py
python src/generate_showcase_notebook.py

# 4. Jalankan pengujian otomatis integritas pipeline (5 test suites)
python -m unittest discover -s tests -v
```

---

## 📂 Struktur Direktori Repositori
- `docs/`: Dokumentasi hidup proyek (*Living Documentation*):
  - 📘 **[`PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md`](docs/PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md)**: Panduan utama presentasi, bedah arsitektur, dan simulasi tanya-jawab sidang dosen.
  - [`00_PROJECT_CHARTER.md`](docs/00_PROJECT_CHARTER.md): Kontrak proyek, hipotesis bisnis, & ruang lingkup.
  - [`01_DATA_DICTIONARY.md`](docs/01_DATA_DICTIONARY.md): Kamus lengkap 60 kolom mentah + 2 kolom turunan (`62` kolom total).
  - [`02_DATA_PIPELINE_LOG.md`](docs/02_DATA_PIPELINE_LOG.md): Log teknis Snappy Parquet (`83,75%` kompresi, `15,39x` speedup I/O) & protokol *Split-Before-Fit*.
  - [`03_MODEL_EXPERIMENTS.md`](docs/03_MODEL_EXPERIMENTS.md): Hasil 72-config grid tuning, ablasi `M0..M4`, benchmark 6 model, uji McNemar, SHAP, & K-Means.
  - [`04_BUSINESS_RECOMMENDATIONS.md`](docs/04_BUSINESS_RECOMMENDATIONS.md): Aturan IF-THEN pohon asli (`<= 4,5`), 3 Tier Operasional, & matriks intervensi bisnis.
  - [`05_TIMELINE_TRACKER.md`](docs/05_TIMELINE_TRACKER.md): Pelacak progres 18 aktivitas proyek.
  - [`06_LOOKER_STUDIO_GUIDE.md`](docs/06_LOOKER_STUDIO_GUIDE.md): Blueprint pembuatan dashboard BI.
  - [`07_JOURNAL_PLAN_JIKO.md`](docs/07_JOURNAL_PLAN_JIKO.md): Blueprint penulisan naskah jurnal Sinta 4 JIKO.
  - [`08_PROJECT_FIX_AND_HARDENING_PLAN.md`](docs/08_PROJECT_FIX_AND_HARDENING_PLAN.md): Log audit & pengerasan ilmiah seluruh repositori.
- `src/`: Modul produksi PySpark modular.
- `reports/`: 12 grafik PNG (300 DPI), `reports/metrics/*.json|*.csv`, `reports/looker_data/*.csv`, `decision_tree_rules.txt`, & `dashboard_visualisasi.html`.
- `tests/`: Suite pengujian otomatis `test_pipeline_integrity.py`.

---

## 📜 Lisensi & Atribusi Dataset
Dataset sumber (*E-Commerce Shopper Behavior — Amazon/Shopify Based*, `dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based`) didistribusikan di bawah lisensi **[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)**. Berkas turunan Snappy Parquet dan model terlatih yang disertakan di repositori ini semata-mata digunakan untuk keperluan riset akademik dan reproduksibilitas notebook showcase (*Plug & Play* Google Colab).
