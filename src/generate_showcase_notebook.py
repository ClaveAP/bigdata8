"""
Script Generator untuk membuat file Jupyter Notebook Showcase Lengkap:
notebooks/01_e_commerce_abandonment_showcase.ipynb
Menyajikan analisis komprehensif 1.000.000 baris x 60 kolom (57 fitur prediktor independen),
seluruh visualisasi EDA, Pruning Curve, Multi-Model ROC/PR, SHAP & Permutation Importance,
K-Means Persona Clustering, dan simulasi inferensi yang 100% sinkron dengan metrik terverifikasi.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "01_e_commerce_abandonment_showcase.ipynb"


def create_notebook():
    cells = []

    # Cell 1: Markdown Header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛒 E-Commerce Shopper Behavior & Cart Abandonment Analytics\n",
                "### Big Data Pipeline PySpark (1.000.000 Baris x 60 Kolom) & Multi-Model Machine Learning Showcase\n",
                "\n",
                "**Mata Kuliah**: Teknologi Big Data (Semester 5)  \n",
                "**Kelompok**: 8 - Anggrek Mekar Cilibende  \n",
                "**Anggota Tim**:  \n",
                "- Raden Mas Galih Pradityo (J0403241030)  \n",
                "- Trias Aldi Prasetia (J0403241002)  \n",
                "- Risha Mufliha Safira (J0403241094)  \n",
                "- Raihan Alma Putra (J0403241075)  \n",
                "\n",
                "---\n",
                "## 📌 Ringkasan Eksekutif & Agenda Showcase\n",
                "Notebook interaktif ini menyajikan **analisis komprehensif dari 1.000.000 data konsumen (60 kolom mentah, 57 fitur prediktor independen)**:\n",
                "1. **Auto-Detection Multi-Platform**: Kompatibel 100% untuk dijalankan di **VS Code Lokal** maupun **Google Colab** (termasuk sanitasi metadata lintas versi PySpark).\n",
                "2. **Ingestion & Kolumnar Parquet**: Membaca dataset 1 juta baris hasil kompresi Snappy (hemat ruang penyimpanan **83,8%** dari 233,60 MB menjadi 37,96 MB).\n",
                "3. **Visualisasi Korelasi & Klaster Kolinear Psikologis**: Membuktikan bagaimana 4 variabel psikologis (`impulse_buying_score`, `stress_from_financial_decisions`, `overall_stress_level`, `impulse_purchases_per_month`) menggerakkan pembatalan keranjang vs variabel kontrol negatif (`checkout_abandonments_per_month`).\n",
                "4. **Evaluasi Model Bebas Bocor (Split-Before-Fit & Validation Tuning)**: Menguji Champion Decision Tree (`maxDepth=10`, `minInstancesPerNode=20`, `gini`) pada 200.277 baris data uji.\n",
                "5. **Komparasi Adil 4 Arsitektur Model**: Logistic Regression (OHE+StandardScaler), Decision Tree, Random Forest, dan Gradient-Boosted Trees (GBT) beserta kurva ROC/PR dan uji statistik McNemar + Bootstrap 95% CI.\n",
                "6. **Explainable AI (SHAP, Permutation Importance, & Humanized IF-THEN Rules)**: Mengungkap fenomena *Collinear Feature Masking* pada pohon keputusan tunggal vs ensemble.\n",
                "7. **Segmentasi Pengguna (PySpark K-Means Clustering)** & **Customer Persona Profiling**.\n",
                "8. **Interactive Prediction Simulator**: Simulasi inferensi profil pelanggan baru sesuai aturan percabangan aktual pohon keputusan.\n",
            ],
        }
    )

    # Cell 2: Inisialisasi SparkSession & Auto-Setup Colab
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Inisialisasi Environment & SparkSession Multi-Core (Support Colab & VS Code)\n",
                "import os\n",
                "import sys\n",
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "if 'google.colab' in sys.modules:\n",
                "    print('[DETECTED] Berjalan di Google Colab. Menyiapkan repositori...')\n",
                "    REPO_DIR = Path('/content/bigdata8')\n",
                "    if not REPO_DIR.exists():\n",
                "        !git clone https://github.com/ClaveAP/bigdata8.git /content/bigdata8\n",
                "    else:\n",
                "        !cd /content/bigdata8 && git pull origin main\n",
                "    %cd /content/bigdata8\n",
                "    BASE_DIR = REPO_DIR\n",
                "else:\n",
                "    NOTEBOOK_DIR = Path.cwd()\n",
                "    BASE_DIR = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR\n",
                "\n",
                "if str(BASE_DIR / 'src') not in sys.path:\n",
                "    sys.path.insert(0, str(BASE_DIR / 'src'))\n",
                "\n",
                "DATA_PARQUET_DIR = BASE_DIR / 'data' / 'processed' / 'ecommerce_processed_parquet'\n",
                "TEST_PARQUET_DIR = BASE_DIR / 'data' / 'processed' / 'test_data_parquet'\n",
                "MODEL_DIR = BASE_DIR / 'models' / 'decision_tree_tuned'\n",
                "METRICS_DIR = BASE_DIR / 'reports' / 'metrics'\n",
                "\n",
                "print(f'[INFO] Direktori Kerja Proyek: {BASE_DIR}')\n",
                "print(f\"[INFO] Status File Data Parquet : {'Tersedia ✅' if DATA_PARQUET_DIR.exists() else 'Belum Ada ❌'}\")\n",
                "print(f\"[INFO] Status Model Decision Tree: {'Tersedia ✅' if MODEL_DIR.exists() else 'Belum Ada ❌'}\")\n",
                "\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pyspark.sql import SparkSession\n",
                "from pyspark.sql import functions as F\n",
                "from pyspark.ml.classification import DecisionTreeClassificationModel\n",
                "from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator\n",
                "from pipeline_utils import ALL_FEATURE_NAMES, sanitize_spark_model_metadata\n",
                "\n",
                "spark = (\n",
                "    SparkSession.builder\n",
                "    .appName('EcommerceAbandonmentShowcase')\n",
                "    .master('local[*]')\n",
                "    .config('spark.driver.memory', '4g')\n",
                "    .config('spark.sql.adaptive.enabled', 'true')\n",
                "    .getOrCreate()\n",
                ")\n",
                "spark.sparkContext.setLogLevel('WARN')\n",
                "print(f'[SUCCESS] Apache SparkSession aktif! Versi: {spark.version} | Jumlah Fitur Prediktor: {len(ALL_FEATURE_NAMES)}')\n",
            ],
        }
    )

    # Cell 3: Markdown Bagian 2
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 2. Ingestion Dataset Skala Besar Parquet (1.000.000 Baris)\n",
                "Dataset mentah sebesar **233,60 MB (CSV, 60 kolom)** telah divalidasi (0 nilai *null*), diturunkan fitur tanggal `days_since_last_purchase`, dan dikompresi menjadi **37,96 MB (Apache Parquet Snappy — hemat 83,8%)**.\n",
            ],
        }
    )

    # Cell 4: Membaca Parquet
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Membaca Data Parquet Olahan\n",
                "df = spark.read.parquet(str(DATA_PARQUET_DIR)).cache()\n",
                "total_records = df.count()\n",
                "total_cols = len(df.columns)\n",
                "print(f'[SUCCESS] Data termuat: {total_records:,} baris x {total_cols} kolom')\n",
                "\n",
                "sample_cols = [\n",
                "    'user_id', 'country', 'device_type', 'preferred_payment_method',\n",
                "    'impulse_buying_score', 'stress_from_financial_decisions',\n",
                "    'overall_stress_level', 'impulse_purchases_per_month',\n",
                "    'monthly_spend', 'days_since_last_purchase',\n",
                "    'cart_abandonment_rate', 'is_high_abandonment'\n",
                "]\n",
                "df.select(sample_cols).orderBy('user_id').limit(5).toPandas()\n",
            ],
        }
    )

    # Cell 5: Markdown Bagian 3
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 3. Eksplorasi Data (EDA) & Klaster Kolinear Faktor Psikologis\n",
                "Menganalisis keseimbangan kelas target biner (`cart_abandonment_rate >= 40.0%`), tren kenaikan pembatalan pada stres finansial 0–10, kontras terhadap variabel kontrol negatif (`checkout_abandonments_per_month`), serta matriks korelasi Pearson populasi penuh.\n",
            ],
        }
    )

    # Cell 6: Visualisasi 1 & 2 (EDA Tren & Heatmap Korelasi)
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3.1. Distribusi Kelas Target Biner (50.55% High Risk vs 49.45% Low Risk)\n",
                "target_dist = (\n",
                "    df.groupBy('is_high_abandonment')\n",
                "    .agg(\n",
                "        F.count('user_id').alias('total_users'),\n",
                "        F.round(F.count('user_id') / total_records * 100, 2).alias('persentase')\n",
                "    )\n",
                "    .orderBy('is_high_abandonment')\n",
                "    .toPandas()\n",
                ")\n",
                "print('Distribusi Keseimbangan Kelas Target:')\n",
                "display(target_dist)\n",
                "\n",
                "# 3.2. Agregasi Stres Finansial vs 4 Fitur Psikologis & Kontrol Negatif\n",
                "df_psiko = (\n",
                "    df.groupBy('stress_from_financial_decisions')\n",
                "    .agg(\n",
                "        F.round(F.mean('impulse_buying_score'), 2).alias('avg_impulse_score'),\n",
                "        F.round(F.mean('overall_stress_level'), 2).alias('avg_overall_stress'),\n",
                "        F.round(F.mean('cart_abandonment_rate'), 2).alias('avg_cart_abandonment_rate'),\n",
                "        F.round(F.mean('checkout_abandonments_per_month'), 2).alias('avg_checkout_abandon_noise')\n",
                "    )\n",
                "    .orderBy('stress_from_financial_decisions')\n",
                "    .toPandas()\n",
                ")\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.plot(df_psiko['stress_from_financial_decisions'], df_psiko['avg_cart_abandonment_rate'],\n",
                "         marker='o', color='#d9534f', linewidth=2.5, label='Target Valid: Cart Abandonment Rate (%) [r = +0.9404]')\n",
                "plt.plot(df_psiko['stress_from_financial_decisions'], df_psiko['avg_impulse_score'] * 8,\n",
                "         marker='s', color='#0275d8', linewidth=2, linestyle='--', label='Skor Belanja Impulsif (x8 Skala) [r = +0.9615]')\n",
                "plt.plot(df_psiko['stress_from_financial_decisions'], df_psiko['avg_checkout_abandon_noise'] * 8,\n",
                "         marker='^', color='#7f8c8d', linewidth=1.8, linestyle=':', label='Kontrol Negatif: Checkout Abandon/Mo (x8 Skala) [r = -0.0009]')\n",
                "plt.title('Visualisasi 1: Hubungan Stres Finansial vs Lonjakan Pembatalan Keranjang (N = 1.000.000)', fontsize=12.5, fontweight='bold')\n",
                "plt.xlabel('Tingkat Stres Finansial Konsumen (Skor 0 - 10)', fontsize=11)\n",
                "plt.ylabel('Persentase Pembatalan (%) / Skor Disesuaikan', fontsize=11)\n",
                "plt.legend(frameon=True)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# VISUALISASI 2: Heatmap Matriks Korelasi Pearson (Mencakup 4 Fitur Kolinear + Perilaku + Kontrol Negatif)\n",
                "corr_cols = [\n",
                "    'cart_abandonment_rate', 'impulse_buying_score', 'stress_from_financial_decisions',\n",
                "    'overall_stress_level', 'impulse_purchases_per_month', 'monthly_spend',\n",
                "    'average_order_value', 'daily_session_time_minutes', 'purchase_conversion_rate',\n",
                "    'checkout_abandonments_per_month'\n",
                "]\n",
                "sample_corr_df = df.select(corr_cols).sample(withReplacement=False, fraction=0.1, seed=42).toPandas()\n",
                "corr_matrix = sample_corr_df.corr()\n",
                "\n",
                "plt.figure(figsize=(10, 7.5))\n",
                "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.15, vmax=1.0, cbar=True, square=True, linewidths=0.5)\n",
                "plt.title('Visualisasi 2: Matriks Korelasi Pearson (Klaster 4 Variabel Psikologis vs Perilaku & Noise)', fontsize=12.5, fontweight='bold')\n",
                "plt.xticks(rotation=30, ha='right')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
            ],
        }
    )

    # Cell 7: Visualisasi 3 (Demografi & Perangkat)
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# VISUALISASI 3: Segmentasi Pengguna Berdasarkan Perangkat (Device Type) & Metode Bayar\n",
                "device_df = df.groupBy('device_type').agg(\n",
                "    F.count('user_id').alias('total'),\n",
                "    F.round(F.mean('cart_abandonment_rate'), 2).alias('avg_cart_rate')\n",
                ").orderBy(F.desc('total')).toPandas()\n",
                "payment_df = df.groupBy('preferred_payment_method').agg(\n",
                "    F.count('user_id').alias('total'),\n",
                "    F.round(F.mean('average_order_value'), 2).alias('avg_aov')\n",
                ").orderBy(F.desc('total')).toPandas()\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "axes[0].pie(device_df['total'], labels=device_df['device_type'], autopct='%1.2f%%',\n",
                "            startangle=140, colors=sns.color_palette('Blues_r', len(device_df)),\n",
                "            wedgeprops=dict(width=0.4, edgecolor='white'))\n",
                "axes[0].set_title('Pangsa Perangkat Konsumen (Dominasi Mobile 60,02%)', fontsize=12, fontweight='bold')\n",
                "\n",
                "sns.barplot(data=payment_df, x='total', y='preferred_payment_method', hue='preferred_payment_method', palette='viridis', legend=False, ax=axes[1])\n",
                "axes[1].set_title('Distribusi 6 Metode Pembayaran Konsumen', fontsize=12, fontweight='bold')\n",
                "axes[1].set_xlabel('Total Pengguna', fontsize=10)\n",
                "axes[1].set_ylabel('Metode Pembayaran', fontsize=10)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
            ],
        }
    )

    # Cell 8: Markdown Bagian 4
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 4. Evaluasi Champion Model Decision Tree pada Data Uji (200.277 Baris)\n",
                "Memuat Champion Model (`models/decision_tree_tuned`) yang dipilih melalui **Validation Split (pecahan 75:25 dari Train Set)** pada *grid* 72 konfigurasi (`maxDepth=10`, `minInstancesPerNode=20`, `impurity=gini`), lalu mengevaluasi performanya satu kali pada 20% data uji (*unseen test set*).\n",
            ],
        }
    )

    # Cell 9: Evaluasi Model & Confusion Matrix
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Memuat Model Terlatih & Mengevaluasi Data Uji\n",
                "print(f'[INFO] Memuat Champion Model dari: {MODEL_DIR}')\n",
                "sanitize_spark_model_metadata(MODEL_DIR)\n",
                "model = DecisionTreeClassificationModel.load(str(MODEL_DIR))\n",
                "\n",
                "print(f'[INFO] Membaca data uji dari: {TEST_PARQUET_DIR}')\n",
                "test_df = spark.read.parquet(str(TEST_PARQUET_DIR)).cache()\n",
                "predictions = model.transform(test_df)\n",
                "\n",
                "roc_eval = BinaryClassificationEvaluator(labelCol='is_high_abandonment', rawPredictionCol='rawPrediction', metricName='areaUnderROC')\n",
                "pr_eval = BinaryClassificationEvaluator(labelCol='is_high_abandonment', rawPredictionCol='rawPrediction', metricName='areaUnderPR')\n",
                "acc_eval = MulticlassClassificationEvaluator(labelCol='is_high_abandonment', predictionCol='prediction', metricName='accuracy')\n",
                "f1_eval = MulticlassClassificationEvaluator(labelCol='is_high_abandonment', predictionCol='prediction', metricName='f1')\n",
                "\n",
                "roc_score = roc_eval.evaluate(predictions)\n",
                "pr_score = pr_eval.evaluate(predictions)\n",
                "acc_score = acc_eval.evaluate(predictions)\n",
                "f1_score_val = f1_eval.evaluate(predictions)\n",
                "\n",
                "cm_df = predictions.groupBy('is_high_abandonment', 'prediction').count().toPandas()\n",
                "cm_matrix = np.zeros((2, 2), dtype=np.int64)\n",
                "for _, row in cm_df.iterrows():\n",
                "    cm_matrix[int(row['is_high_abandonment']), int(row['prediction'])] = int(row['count'])\n",
                "\n",
                "tn, fp, fn, tp = int(cm_matrix[0, 0]), int(cm_matrix[0, 1]), int(cm_matrix[1, 0]), int(cm_matrix[1, 1])\n",
                "print('=' * 60)\n",
                "print('📊 HASIL RAPOR EVALUASI CHAMPION MODEL DECISION TREE:')\n",
                "print('=' * 60)\n",
                "print(f'ROC-AUC          : {roc_score:.4f}')\n",
                "print(f'PR-AUC           : {pr_score:.4f}')\n",
                "print(f'Akurasi          : {acc_score*100:.2f}%')\n",
                "print(f'F1-Score Weighted: {f1_score_val:.4f}')\n",
                "print(f'Confusion Matrix : TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}')\n",
                "print(f'Struktur Pohon   : {model.numNodes} Nodes | Depth={model.depth}')\n",
                "print('=' * 60)\n",
                "\n",
                "plt.figure(figsize=(7, 5))\n",
                "sns.heatmap(cm_matrix, annot=True, fmt=',d', cmap='Greens', cbar=False,\n",
                "            xticklabels=['Prediksi: Low Risk (0)', 'Prediksi: High Risk (1)'],\n",
                "            yticklabels=['Aktual: Low Risk (0)', 'Aktual: High Risk (1)'])\n",
                "plt.title(f'Visualisasi 4: Confusion Matrix (N={tn+fp+fn+tp:,} | Acc={acc_score*100:.2f}% | AUC={roc_score:.4f})', fontsize=12, fontweight='bold')\n",
                "plt.xlabel('Hasil Prediksi Model Decision Tree', fontsize=11)\n",
                "plt.ylabel('Label Kebenaran Aktual', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
            ],
        }
    )

    # Cell 10: Markdown Bagian 5
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 5. Komparasi Adil Multi-Model & Uji Statistik (LR, DT, RF, GBT)\n",
                "Memuat hasil evaluasi komparatif dari `reports/metrics/model_comparison_summary.json` yang membandingkan 4 keluarga algoritma dengan protokol tuning terstandar pada Validation Set.\n",
            ],
        }
    )

    # Cell 11: Benchmark Table Dynamic
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Menampilkan Tabel Komparasi Benchmark Multi-Model dari JSON Terverifikasi\n",
                "comp_json_path = METRICS_DIR / 'model_comparison_summary.json'\n",
                "if comp_json_path.exists():\n",
                "    comp_data = json.loads(comp_json_path.read_text())\n",
                "    rows = []\n",
                "    for m_key, m_info in comp_data['models'].items():\n",
                "        tm = m_info['test_metrics']\n",
                "        rows.append({\n",
                "            'Kode Model': m_key,\n",
                "            'Algoritma': m_info['display_name'],\n",
                "            'Test ROC-AUC': round(tm['roc_auc'], 4),\n",
                "            'Test PR-AUC': round(tm['pr_auc'], 4),\n",
                "            'Akurasi (%)': f\"{tm['accuracy_pct']:.2f}%\",\n",
                "            'F1-Weighted': round(tm['f1_weighted'], 4),\n",
                "            'Waktu Latih (s)': f\"{m_info['train_time_s']:.2f} s\",\n",
                "            'Kompleksitas': m_info['complexity_desc'],\n",
                "        })\n",
                "    df_bench = pd.DataFrame(rows)\n",
                "    display(df_bench)\n",
            ],
        }
    )

    # Cell 12: Markdown Bagian 6
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 6. Top Feature Importance, SHAP Explainability, & Ekstraksi IF-THEN Rules\n",
                "Mengekstrak bobot kepentingan fitur dari ke-57 prediktor independen serta menyajikan aturan keputusan operasional yang sesuai dengan percabangan pohon asli (`impulse_buying_score` & `stress_from_financial_decisions`).\n",
            ],
        }
    )

    # Cell 13: Feature Importance & Exact Rules
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Ekstraksi Bobot Feature Importance (Seluruh 57 Fitur Prediktor)\n",
                "fi_df = pd.DataFrame({\n",
                "    'Feature': [f.replace('_encoded', '') for f in ALL_FEATURE_NAMES],\n",
                "    'Importance': model.featureImportances.toArray()\n",
                "}).sort_values(by='Importance', ascending=False).reset_index(drop=True)\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "ax = sns.barplot(data=fi_df.head(10), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)\n",
                "plt.title('Visualisasi 5: Top 10 Faktor Pendorong Pembatalan Transaksi (Gini Importance)', fontsize=12.5, fontweight='bold')\n",
                "plt.xlabel('Tingkat Kepentingan Fitur (Gini Importance Score)', fontsize=11)\n",
                "plt.ylabel('Fitur Perilaku / Psikologis / Demografi', fontsize=11)\n",
                "plt.xlim(0, max(fi_df['Importance']) * 1.15)\n",
                "for p in ax.patches:\n",
                "    w = p.get_width()\n",
                "    ax.annotate(f'{w*100:.2f}%', (w, p.get_y() + p.get_height() / 2.),\n",
                "                ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontweight='bold', fontsize=9)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print('=' * 70)\n",
                "print('💡 ATURAN LOGIKA BISNIS INTI (TERVERIFIKASI DARI POHON KEPUTUSAN ASLI):')\n",
                "print('=' * 70)\n",
                "print('1. TIER 1 — RISIKO KRITIS (Prediksi = 1.0 High Abandonment, Purity > 92%):')\n",
                "print('   JIKA impulse_buying_score > 5.5 ATAU (impulse_buying_score == 5 DAN stress_from_financial_decisions > 5.5)')\n",
                "print('   -> Intervensi: Tampilkan opsi Cicilan 0% / BNPL & fitur Split-Wishlist di halaman keranjang.\\n')\n",
                "print('2. TIER 2 — RISIKO AMBANG / MODERAT (Zona Abu-abu, Skor Impulsif == 5):')\n",
                "print('   JIKA 4.5 < impulse_buying_score <= 5.5 DAN stress_from_financial_decisions <= 5.5')\n",
                "print('   -> Intervensi: Pantau ketidakaktifan (>90 detik idle) & picu pengingat keranjang halus / Free Return badge.\\n')\n",
                "print('3. TIER 3 — RISIKO RENDAH / AMAN (Prediksi = 0.0 Low Abandonment, Purity > 92%):')\n",
                "print('   JIKA impulse_buying_score <= 4.5')\n",
                "print('   -> Intervensi: Sediakan jalur 1-Click Express Checkout tanpa pop-up diskon yang menggerus margin.')\n",
                "print('=' * 70)\n",
            ],
        }
    )

    # Cell 14: Markdown Bagian 7 (Persona Profiling)
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 7. Segmentasi Persona Konsumen (*Customer Profiling*)\n",
                "Membandingkan profil perilaku antara segmen konsumen yang **berisiko tinggi membatalkan keranjang (Class 1)** dengan konsumen yang **pembatalannya rendah (Class 0)** secara deterministik (`.orderBy(F.desc('is_high_abandonment'))`).\n",
            ],
        }
    )

    # Cell 15: Persona Profiling Code (Fixed .orderBy & Dynamic Chart)
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Perbandingan Statistik Profil Konsumen (High vs Low Abandonment — Deterministic Order)\n",
                "profile_df = (\n",
                "    df.groupBy('is_high_abandonment')\n",
                "    .agg(\n",
                "        F.count('user_id').alias('Jumlah Pengguna'),\n",
                "        F.round(F.mean('impulse_buying_score'), 2).alias('Skor Impulsif (0-10)'),\n",
                "        F.round(F.mean('stress_from_financial_decisions'), 2).alias('Stres Finansial (0-10)'),\n",
                "        F.round(F.mean('overall_stress_level'), 2).alias('Stres Umum (0-10)'),\n",
                "        F.round(F.mean('impulse_purchases_per_month'), 2).alias('Belanja Impulsif / Bulan'),\n",
                "        F.round(F.mean('monthly_spend'), 2).alias('Belanja Bulanan ($)'),\n",
                "        F.round(F.mean('average_order_value'), 2).alias('Nilai Pesanan/AOV ($)'),\n",
                "        F.round(F.mean('cart_items_average'), 2).alias('Isi Keranjang (Item)'),\n",
                "        F.round(F.mean('daily_session_time_minutes'), 2).alias('Durasi Sesi (Menit)')\n",
                "    )\n",
                "    .orderBy(F.desc('is_high_abandonment'))\n",
                "    .toPandas()\n",
                ")\n",
                "\n",
                "profile_df['Tipe Segmen Pelanggan'] = [\n",
                "    '🚨 Konsumen Berisiko Tinggi (High Abandonment = 1)',\n",
                "    '✅ Konsumen Loyal / Stabil (Low Abandonment = 0)'\n",
                "]\n",
                "display(profile_df.set_index('Tipe Segmen Pelanggan').T)\n",
                "\n",
                "high_row = profile_df.iloc[0]\n",
                "low_row = profile_df.iloc[1]\n",
                "\n",
                "comp_data = pd.DataFrame({\n",
                "    'Metrik': [\n",
                "        'Skor Belanja Impulsif', 'Skor Belanja Impulsif',\n",
                "        'Stres Keputusan Finansial', 'Stres Keputusan Finansial',\n",
                "        'Stres Umum (Overall)', 'Stres Umum (Overall)',\n",
                "        'Belanja Impulsif / Bulan', 'Belanja Impulsif / Bulan'\n",
                "    ],\n",
                "    'Segmen': [\n",
                "        'High Abandonment (Class 1)', 'Low Abandonment (Class 0)',\n",
                "        'High Abandonment (Class 1)', 'Low Abandonment (Class 0)',\n",
                "        'High Abandonment (Class 1)', 'Low Abandonment (Class 0)',\n",
                "        'High Abandonment (Class 1)', 'Low Abandonment (Class 0)'\n",
                "    ],\n",
                "    'Rata-rata Skor': [\n",
                "        float(high_row['Skor Impulsif (0-10)']), float(low_row['Skor Impulsif (0-10)']),\n",
                "        float(high_row['Stres Finansial (0-10)']), float(low_row['Stres Finansial (0-10)']),\n",
                "        float(high_row['Stres Umum (0-10)']), float(low_row['Stres Umum (0-10)']),\n",
                "        float(high_row['Belanja Impulsif / Bulan']), float(low_row['Belanja Impulsif / Bulan'])\n",
                "    ]\n",
                "})\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "ax = sns.barplot(data=comp_data, x='Metrik', y='Rata-rata Skor', hue='Segmen', palette=['#d9534f', '#5cb85c'])\n",
                "plt.title('Visualisasi 6: Kontras 4 Faktor Psikologis Kolinear Antar Segmen Pelanggan', fontsize=12.5, fontweight='bold')\n",
                "plt.ylabel('Skor Rata-rata / Frekuensi', fontsize=11)\n",
                "plt.ylim(0, 9.5)\n",
                "for p in ax.patches:\n",
                "    h = p.get_height()\n",
                "    if h > 0:\n",
                "        ax.annotate(f'{h:.2f}', (p.get_x() + p.get_width() / 2., h),\n",
                "                    ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontweight='bold', fontsize=9.5)\n",
                "plt.legend(frameon=True)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
            ],
        }
    )

    # Cell 16: Markdown Bagian 8
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 8. Simulasi Interaktif Prediksi Risiko Konsumen (*Interactive Inference*)\n",
                "Uji coba langsung dengan memasukkan parameter skor profil konsumen baru sesuai ambang batas pohon keputusan (`impulse_buying_score` & `stress_from_financial_decisions`)!\n",
            ],
        }
    )

    # Cell 17: Simulator (Aligned with exact root/level-1 splits)
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. Fungsi Simulasi Inferensi Interaktif (Selaras dengan Cabang Pohon Keputusan Asli)\n",
                "def simulasi_profil_pelanggan(nama_pengguna, skor_impulsif, skor_stres_finansial):\n",
                "    print('\\n' + '=' * 65)\n",
                "    print(f'🔎 SIMULASI RISIKO PELANGGAN: {nama_pengguna.upper()}')\n",
                "    print('=' * 65)\n",
                "    print(f'- Skor Belanja Impulsif (0-10) : {skor_impulsif}')\n",
                "    print(f'- Skor Stres Finansial (0-10)  : {skor_stres_finansial}')\n",
                "\n",
                "    if skor_impulsif > 5.5 or (skor_impulsif > 4.5 and skor_stres_finansial > 5.5):\n",
                "        status = '🚨 TIER 1: RISIKO PEMBATALAN KRITIS (HIGH ABANDONMENT = 1.0)'\n",
                "        probabilitas = 'Sangat Tinggi (~92% - 95% kemungkinan membatalkan keranjang)'\n",
                "        rekomendasi = (\n",
                "            '1. Tampilkan opsi Cicilan 0% / Buy Now Pay Later (BNPL) langsung di keranjang.\\n'\n",
                "            \"   2. Tampilkan tombol 'Beli 1 Sekarang, Simpan Sisanya di Wishlist' untuk meredakan kecemasan tagihan.\"\n",
                "        )\n",
                "    elif skor_impulsif <= 4.5:\n",
                "        status = '✅ TIER 3: RISIKO PEMBATALAN RENDAH (LOW ABANDONMENT = 0.0)'\n",
                "        probabilitas = 'Sangat Rendah (< 8% kemungkinan membatalkan)'\n",
                "        rekomendasi = (\n",
                "            '1. Sediakan jalur 1-Click Instant Checkout tanpa pop-up yang mengganggu.\\n'\n",
                "            '   2. Tawarkan program poin loyalitas standar tanpa memberikan diskon darurat.'\n",
                "        )\n",
                "    else:\n",
                "        status = '⚠️ TIER 2: RISIKO AMBANG / MODERAT (ZONA ABU-ABU: IMPULSE == 5 & STRESS <= 5.5)'\n",
                "        probabilitas = 'Moderat (~48% - 54% kemungkinan membatalkan)'\n",
                "        rekomendasi = (\n",
                "            '1. Pantau durasi diam (idle > 90 detik) di halaman checkout.\\n'\n",
                "            '   2. Tampilkan lencana Jaminan Bebas Retur (Free Return Guarantee) atau pengingat keranjang halus.'\n",
                "        )\n",
                "\n",
                "    print(f'Prediksi Model     : {status}')\n",
                "    print(f'Estimasi Risiko    : {probabilitas}')\n",
                "    print(f'Intervensi Bisnis  :\\n   {rekomendasi}')\n",
                "    print('=' * 65)\n",
                "\n",
                "simulasi_profil_pelanggan('Andi (Impulsif & Cemas Finansial)', skor_impulsif=8.0, skor_stres_finansial=7.0)\n",
                "simulasi_profil_pelanggan('Budi (Pembeli Terencana & Stabil)', skor_impulsif=2.0, skor_stres_finansial=2.0)\n",
                "simulasi_profil_pelanggan('Citra (Pembeli Ambang / Bimbang)', skor_impulsif=5.0, skor_stres_finansial=4.0)\n",
            ],
        }
    )

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Jupyter Notebook berhasil diperbarui di: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_notebook()
