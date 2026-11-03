"""
Modul Eksplorasi Data Awal (EDA) Skala Besar (1.000.000 Baris)
Menganalisis distribusi, missing values, dan korelasi fitur sesi/perilaku
terhadap checkout_abandonments_per_month & cart_abandonment_rate menggunakan PySpark.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "e_commerce_shopper_behaviour_and_lifestyle.csv"
REPORTS_DIR = BASE_DIR / "reports"

def run_eda():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark_session(app_name="EcommerceEDAProfiling")
    
    print(f"\n[INFO] Membaca dataset dari: {RAW_DATA_PATH}")
    df = spark.read.csv(str(RAW_DATA_PATH), header=True, inferSchema=True)
    
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"[SUCCESS] Dataset termuat: {total_rows:,} baris x {total_cols} kolom")
    
    # 1. Pengecekan Missing Values (isNull aman untuk String & Numeric)
    print("\n[ANALISIS 1] Menghitung Missing Values / Null per Kolom...")
    null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns[:20]]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    total_nulls = sum(null_counts.values())
    if total_nulls == 0:
        print("[SUCCESS] 100% Bersih! Tidak ditemukan Missing Values / Null pada 20 fitur awal.")
    else:
        print(f"[WARN] Ditemukan missing values: {null_counts}")
        
    # 2. Statistik Deskriptif Target Pembatalan
    print("\n[ANALISIS 2] Statistik Target Pembatalan Transaksi:")
    df.select(
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.stddev("checkout_abandonments_per_month"), 2).alias("std_checkout_abandon"),
        F.min("checkout_abandonments_per_month").alias("min_checkout_abandon"),
        F.max("checkout_abandonments_per_month").alias("max_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_rate"),
        F.round(F.stddev("cart_abandonment_rate"), 2).alias("std_cart_rate")
    ).show()

    # 3. Agregasi Hubungan Device Type vs Rata-rata Pembatalan
    print("\n[ANALISIS 3] Rata-rata Pembatalan berdasarkan Device Type:")
    device_summary = df.groupBy("device_type").agg(
        F.count("user_id").alias("user_count"),
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_rate"),
        F.round(F.mean("daily_session_time_minutes"), 2).alias("avg_session_min")
    ).toPandas()
    print(device_summary)
    
    # 4. Agregasi Hubungan Stres Finansial vs Pembatalan
    print("\n[ANALISIS 4] Pengaruh Stress from Financial Decisions (0 - 5):")
    stress_summary = df.groupBy("stress_from_financial_decisions").agg(
        F.count("user_id").alias("user_count"),
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_rate"),
        F.round(F.mean("impulse_buying_score"), 2).alias("avg_impulse_score")
    ).orderBy("stress_from_financial_decisions").toPandas()
    print(stress_summary)

    # 5. Visualisasi & Plot Export ke reports/
    print("\n[INFO] Menghasilkan Visualisasi EDA ke folder reports/...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Device Type vs Checkout Abandonment
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=device_summary, x="device_type", y="avg_checkout_abandon", palette="Blues_d")
    plt.title("Rata-rata Checkout Abandonment per Bulan berdasarkan Device Type", fontsize=12, fontweight="bold")
    plt.xlabel("Device Type", fontsize=10)
    plt.ylabel("Rata-rata Pembatalan / Bulan", fontsize=10)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plot1_path = REPORTS_DIR / "eda_device_vs_abandonment.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot1_path.name}")

    # Plot 2: Financial Stress vs Abandonment Rate
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(data=stress_summary, x="stress_from_financial_decisions", y="avg_checkout_abandon", marker="o", color="crimson", linewidth=2.5)
    plt.title("Tren Pembatalan Checkout terhadap Tingkat Stres Finansial", fontsize=12, fontweight="bold")
    plt.xlabel("Tingkat Stres Finansial (Skor)", fontsize=10)
    plt.ylabel("Rata-rata Pembatalan / Bulan", fontsize=10)
    plt.tight_layout()
    plot2_path = REPORTS_DIR / "eda_stress_vs_abandonment.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot2_path.name}")

    spark.stop()
    print("\n[SUCCESS] Analisis EDA Skala Besar Selesai!")

if __name__ == "__main__":
    run_eda()
