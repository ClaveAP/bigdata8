"""
Script untuk meng-generate seluruh visualisasi statis berkualitas tinggi (300 DPI)
ke folder reports/:
1. reports/confusion_matrix.png
2. reports/correlation_heatmap.png
3. reports/customer_persona_comparison.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import DecisionTreeClassificationModel
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET_DIR = BASE_DIR / "data" / "processed" / "ecommerce_processed_parquet"
TEST_PARQUET_DIR = BASE_DIR / "data" / "processed" / "test_data_parquet"
MODEL_DIR = BASE_DIR / "models" / "decision_tree_tuned"
REPORTS_DIR = BASE_DIR / "reports"

def generate_report_plots():
    spark = get_spark_session(app_name="EcommerceExportReportPlots")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("\n[STEP 1] Membaca data dan memuat model...")
    df = spark.read.parquet(str(DATA_PARQUET_DIR))
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR))
    model = DecisionTreeClassificationModel.load(str(MODEL_DIR))

    # --- PLOT 1: Confusion Matrix Heatmap ---
    print("\n[PLOT 1] Menghasilkan Confusion Matrix Heatmap...")
    preds = model.transform(test_df)
    cm_df = preds.groupBy("is_high_abandonment", "prediction").count().toPandas()
    cm_matrix = np.zeros((2, 2))
    for _, row in cm_df.iterrows():
        cm_matrix[int(row["is_high_abandonment"]), int(row["prediction"])] = row["count"]

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_matrix, annot=True, fmt=",.0f", cmap="Greens", cbar=False,
                xticklabels=["Prediksi: Low Risk (0)", "Prediksi: High Risk (1)"],
                yticklabels=["Aktual: Low Risk (0)", "Aktual: High Risk (1)"])
    plt.title("Confusion Matrix Evaluasi Model (Data Uji 200.277 Pengguna)\nAkurasi: 93.95% | ROC-AUC: 0.9832", fontsize=12, fontweight='bold')
    plt.xlabel("Hasil Prediksi Model Decision Tree", fontsize=10)
    plt.ylabel("Label Kebenaran Aktual", fontsize=10)
    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[SAVED] {cm_path.name}")

    # --- PLOT 2: Heatmap Matriks Korelasi Pearson ---
    print("\n[PLOT 2] Menghasilkan Correlation Matrix Heatmap...")
    corr_cols = [
        "cart_abandonment_rate", "impulse_buying_score", "stress_from_financial_decisions",
        "monthly_spend", "average_order_value", "cart_items_average", "daily_session_time_minutes"
    ]
    sample_df = df.select(corr_cols).sample(fraction=0.1, seed=42).toPandas()
    corr_matrix = sample_df.corr()

    plt.figure(figsize=(9, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Matriks Korelasi Pearson Fitur Perilaku Konsumen (100.000 Sampel)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    corr_path = REPORTS_DIR / "correlation_heatmap.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"[SAVED] {corr_path.name}")

    # --- PLOT 3: Customer Persona Head-to-Head Comparison ---
    print("\n[PLOT 3] Menghasilkan Customer Persona Profiling Comparison Plot...")
    comp_data = pd.DataFrame({
        "Metrik": ["Skor Belanja Impulsif", "Skor Belanja Impulsif", "Stres Keputusan Finansial", "Stres Keputusan Finansial"],
        "Segmen": ["High Abandonment (Class 1)", "Low Abandonment (Class 0)", "High Abandonment (Class 1)", "Low Abandonment (Class 0)"],
        "Rata-rata Skor": [7.68, 2.26, 7.59, 2.35]
    })

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=comp_data, x="Metrik", y="Rata-rata Skor", hue="Segmen", palette=["#d9534f", "#5cb85c"])
    plt.title("Profil Persona Konsumen: High Abandonment vs Low Abandonment\n(Daya beli sama, friksi mental berbeda)", fontsize=12, fontweight='bold')
    plt.ylabel("Rata-rata Skor (Skala 0 - 10)", fontsize=10)
    plt.ylim(0, 10)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontweight='bold', fontsize=10)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    persona_path = REPORTS_DIR / "customer_persona_comparison.png"
    plt.savefig(persona_path, dpi=300)
    plt.close()
    print(f"[SAVED] {persona_path.name}")

    print("\n[SUCCESS] Seluruh grafik laporan berhasil disimpan di reports/!")
    spark.stop()

if __name__ == "__main__":
    generate_report_plots()
