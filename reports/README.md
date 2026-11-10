# Katalog Artefak, Grafik Resolusi Tinggi, & Metrik Evaluasi (`reports/README.md`)

Folder `reports/` menyimpan seluruh keluaran visual beresolusi publikasi (**300 DPI**), artefak metrik JSON/CSV terverifikasi (`reports/metrics/`), ekstraksi aturan bisnis pohon keputusan (`decision_tree_rules.txt`), data mart Business Intelligence (`reports/looker_data/`), serta dashboard eksekutif interaktif (`dashboard_visualisasi.html`) dari analisis pembatalan keranjang (*cart/checkout abandonment*) berbasis **1.000.000 baris data pengguna e-commerce**.

---

## 📊 1. Daftar 12 Grafik Publikasi & Laporan (300 DPI)

| No | Nama Berkas | Dihasilkan Oleh | Deskripsi & Temuan Ilmiah Utama |
| :---: | :--- | :--- | :--- |
| **1** | **`confusion_matrix.png`** | `src/export_report_plots.py` | Matriks konfusi **Champion Decision Tree (`maxDepth=10, minInst=20, gini`)** pada `200.277` baris *unseen test set*: **`TN = 93.146` (`46,51%`)**, **`FP = 5.777` (`2,88%`)**, **`FN = 6.264` (`3,13%`)**, **`TP = 95.090` (`47,48%`)** — Akurasi **`93,99%`**, ROC-AUC **`0,9904`**, F1-Weighted **`0,9399`**. |
| **2** | **`correlation_heatmap.png`** | `src/export_report_plots.py` | Matriks korelasi Pearson (`1.000.000` baris) yang menampilkan klaster 4 fitur psikologis kolinear terhadap `cart_abandonment_rate`: `impulse_buying_score` ($r = +0,9615$), `stress_from_financial_decisions` ($r = +0,9404$), `overall_stress_level` ($r = +0,8635$), dan `impulse_purchases_per_month` ($r = +0,8501$). |
| **3** | **`customer_persona_comparison.png`** | `src/export_report_plots.py` | Komparasi *head-to-head* profil **High Abandonment (`505.469` user)** vs **Low Abandonment (`494.531` user)**: skor impulsif (`7,65` vs `2,29`) dan stres finansial (`7,62` vs `2,32`) berbeda tajam, sedangkan pengeluaran bulanan (`$2.499,68` vs `$2.497,48`) dan durasi sesi (`60,01` vs `60,04` menit) identik. |
| **4** | **`decision_tree_feature_importance.png`** | `src/extract_rules_insights.py` | Peringkat fitur teratas pada `DT_Champion_d10`: `impulse_buying_score` (**`95,00%`**) dan `stress_from_financial_decisions` (**`4,72%`**) menyumbang **`99,72%`** penurunan impuritas Gini pada pohon tunggal. |
| **5** | **`eda_device_vs_abandonment.png`** | `src/eda_profiling.py` | Distribusi pembatalan keranjang lintas perangkat (`Mobile 60,02%`, `Desktop 29,98%`, `Tablet 9,99%`) yang seragam di kisaran `~40,2%`, membuktikan friksi bukan disebabkan oleh kendala teknis *platform*. |
| **6** | **`eda_stress_vs_abandonment.png`** | `src/eda_profiling.py` | Kurva lonjakan non-linear laju pembatalan keranjang terhadap `stress_from_financial_decisions` (melonjak dari **`4,71%`** pada level stres `1` menjadi **`77,03%`** pada level stres `10`). |
| **7** | **`model_roc_pr_curves.png`** | `src/compare_models.py` | Kurva ROC dan Precision-Recall komparatif 6 model pada `200.277` data uji (`LR_Tuned 0,9906`, `GBT_Tuned 0,9906`, `DT_Champion_d10 0,9904`, `RF_Tuned 0,9894`, `DT_Default_d7 0,9773`, `DT_Shallow_d3 0,9651`). |
| **8** | **`dt_pruning_depth_curve.png`** | `src/tune_tree.py` | Kurva validasi 72 konfigurasi Decision Tree (`maxDepth 2..12`) yang memperlihatkan plateau performa pada `maxDepth 7..10` dan degradasi *overfitting* serta ledakan `3.431` node pada `maxDepth=12, minInst=1`. |
| **9** | **`pyspark_scaling_benchmark.png`** | `src/compare_models.py` | Benchmark skalabilitas waktu pelatihan PySpark terhadap ukuran baris (`100k`–`1M` baris) dan jumlah *executor threads* (`local[1]` `20,68s` $\rightarrow$ `local[2]` `9,01s`, speedup **`2,30x`**). |
| **10** | **`kmeans_persona_clusters.png`** | `src/clustering.py` | Evaluasi Silhouette Score & WSSSE K-Means (`K=2..5`, Silhouette terbaik `0,3921` pada `K=2`) beserta profil 3 Persona Pelanggan operasional (`K=3`). |
| **11** | **`shap_summary_plot.png`** | `src/explainability_shap.py` | Distribusi nilai **TreeSHAP** dan Mean $|\text{SHAP}|$ yang membuktikan arah dorongan positif `impulse_buying_score` dan `stress_from_financial_decisions` serta kontribusi `overall_stress_level` dan `impulse_purchases_per_month` pada ensemble pohon. |
| **12** | **`permutation_vs_gini_importance.png`** | `src/explainability_shap.py` | Komparasi 4 metode atribusi fitur (*Single-Tree Gini*, *Ensemble RF Gini*, *Standardized LR Weight*, dan *Permutation AUC Drop*) yang membongkar fenomena *Greedy Collinear Feature Masking*. |

---

## 📁 2. Direktori Metrik Terstruktur (`reports/metrics/`)

Seluruh angka evaluasi disimpan secara otomatis dalam format JSON dan CSV agar 100% dapat diaudit dan direproduksi:
- **`preprocessing_summary.json`**: Statistik ukuran CSV (`233,60 MB`) vs Snappy Parquet (`37,96 MB`, penghematan `83,75%`), benchmark I/O (`15,39x`), dan distribusi target biner.
- **`eda_summary.json` & `schema_and_correlation_audit.csv`**: Audit skema 60 kolom mentah, korelasi Pearson terhadap target, dan statistik per kuartil impulsif.
- **`feature_metadata.json`**: Daftar `57` fitur prediktor (`43` numerik + `14` kategorikal), dimensi `features_lr` (`132`), dan jumlah baris split (`799.723` Train / `200.277` Test).
- **`baseline_dt_metrics.json`**: Metrik model baseline `maxDepth=5` (`11` node, Test ROC-AUC `0,9318`, Akurasi `94,09%`).
- **`dt_tuning_grid.csv` & `dt_champion_summary.json`**: Hasil 72 konfigurasi grid validasi, tabel ablasi 7 model (`M0..M4`), dan metrik Champion `M2` (`maxDepth=10, minInst=20, gini`).
- **`rules_and_fidelity.json` & `decision_tree_rules.txt`**: Pohon keputusan yang telah diterjemahkan ke nama fitur asli (`impulse_buying_score`, `stress_from_financial_decisions`, dst.) beserta analisis **Surrogate Fidelity `98,28%`** (`DT_Shallow_d3` 4 daun vs `DT_Champion_d10` 473 node).
- **`model_comparison_summary.json`, `model_comparison.csv`, `mcnemar_tests.csv`, `bootstrap_auc_deltas.csv`**: Benchmark 6 model, 5-fold CV, uji McNemar berpasangan, dan selang kepercayaan Bootstrap 95%.
- **`sensitivity_and_negative_control.csv`**: Uji sensitivitas threshold (`35%`, `40%`, `45%`) dan uji kontrol negatif `checkout_abandonments_per_month >= 5` ($\text{AUC} \approx 0,50$).
- **`scaling_rows.csv` & `scaling_cores.csv`**: Hasil benchmark skalabilitas baris dan paralelisme CPU PySpark.
- **`kmeans_segmentation.json`**: Metrik Silhouette/WSSSE `K=2..5` dan profil segmen persona pelanggan.
- **`explainability_shap_permutation.json` & `permutation_importance.csv`**: Tabel atribusi lintas-metode, TreeSHAP, dan *joint cluster permutation*.

---

## 🗄️ 3. Direktori Data Mart BI (`reports/looker_data/`)

Berisi 5 file CSV agregasi terstruktur untuk **Google Looker Studio**, **Tableau**, atau **Power BI**:
1. **`looker_scorecard_global.csv`** (`1` baris): KPI makro global (`1.000.000` user, laju pembatalan `40,21%`, proporsi High Risk `50,55%`, rata-rata sesi `60,02` menit, rata-rata pengeluaran `$2.498,59`).
2. **`looker_demografi.csv`** (`120` baris): Agregasi multi-dimensi berdasarkan `country`, `device_type`, dan `age_group`.
3. **`looker_metode_bayar.csv`** (`48` baris): Analisis silang `preferred_payment_method` $\times$ `income_bracket`.
4. **`looker_stres_impulse.csv`** (`11` baris): Kurva `stress_from_financial_decisions` (skala `0–10`) terhadap skor impulsif dan laju pembatalan.
5. **`looker_model_feature_importance.csv`** (`10` baris): Bobot kepentingan fitur Champion Decision Tree dalam skala persentase.

---

## 🔄 4. Cara Mereproduksi Seluruh Artefak dari Awal

```bash
# Jalankan seluruh tahapan pipeline secara berurutan:
.venv/bin/python src/preprocessing.py
.venv/bin/python src/eda_profiling.py
.venv/bin/python src/feature_engineering.py
.venv/bin/python src/train_classifier.py
.venv/bin/python src/tune_tree.py
.venv/bin/python src/extract_rules_insights.py
.venv/bin/python src/clustering.py
.venv/bin/python src/explainability_shap.py
.venv/bin/python src/compare_models.py
.venv/bin/python src/export_looker_data.py
.venv/bin/python src/export_report_plots.py
.venv/bin/python src/generate_showcase_notebook.py

# Verifikasi integritas pipeline dengan unit test otomatis:
.venv/bin/python -m unittest discover -s tests -v
```
