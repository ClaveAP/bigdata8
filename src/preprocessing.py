"""
Modul Pembersihan Data Skala Besar & Rekayasa Fitur Awal (PySpark)
Membaca CSV mentah 1.000.000 baris, memvalidasi missing values,
membuat label target biner (is_high_abandonment), dan mengekspor ke format Apache Parquet.
"""

import os
import shutil
from pathlib import Path
from pyspark.sql import functions as F
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "e_commerce_shopper_behaviour_and_lifestyle.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "ecommerce_processed_parquet"

def run_preprocessing():
    spark = get_spark_session(app_name="EcommerceDataPreprocessing")
    print(f"\n[INFO] Membaca dataset mentah: {RAW_DATA_PATH}")
    
    df = spark.read.csv(str(RAW_DATA_PATH), header=True, inferSchema=True)
    initial_count = df.count()
    print(f"[SUCCESS] Data termuat: {initial_count:,} baris x {len(df.columns)} kolom")
    
    # 1. Handling Missing Values (Drop baris yang memiliki nilai null pada kolom krusial)
    print("\n[STEP 1] Validasi & Pembersihan Missing Values...")
    crucial_cols = [
        "user_id", "checkout_abandonments_per_month", "cart_abandonment_rate",
        "daily_session_time_minutes", "cart_items_average", "device_type",
        "stress_from_financial_decisions", "monthly_spend"
    ]
    df_clean = df.dropna(subset=crucial_cols)
    clean_count = df_clean.count()
    print(f"[INFO] Baris setelah validasi null: {clean_count:,} baris (Dihapus: {initial_count - clean_count})")
    
    # 2. Pembuatan Fitur Target Biner (is_high_abandonment)
    # Threshold: >= 5 checkout abandonments/month = 1 (Tinggi), < 5 = 0 (Rendah)
    print("\n[STEP 2] Rekayasa Variabel Target (is_high_abandonment)...")
    df_transformed = df_clean.withColumn(
        "is_high_abandonment",
        F.when(F.col("checkout_abandonments_per_month") >= 5, 1).otherwise(0)
    )
    
    # Cek distribusi kelas target
    print("[INFO] Distribusi Kelas Target (is_high_abandonment):")
    df_transformed.groupBy("is_high_abandonment").agg(
        F.count("user_id").alias("total_users"),
        F.round(F.count("user_id") / clean_count * 100, 2).alias("percentage")
    ).show()
    
    # 3. Ekspor ke Apache Parquet (Standar Industri Big Data dengan Kompresi Snappy)
    print(f"\n[STEP 3] Menyimpan data terproses ke format Apache Parquet di: {PROCESSED_DATA_DIR}")
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
        
    (
        df_transformed.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(str(PROCESSED_DATA_DIR))
    )
    
    # Hitung ukuran storage Parquet vs CSV mentah
    raw_size_mb = RAW_DATA_PATH.stat().st_size / (1024 * 1024)
    parquet_size_mb = sum(f.stat().st_size for f in PROCESSED_DATA_DIR.glob("**/*.parquet")) / (1024 * 1024)
    compression_ratio = (1 - (parquet_size_mb / raw_size_mb)) * 100
    
    print(f"[SUCCESS] Ekspor Parquet Selesai!")
    print(f"  - Ukuran CSV Mentah  : {raw_size_mb:.2f} MB")
    print(f"  - Ukuran Parquet     : {parquet_size_mb:.2f} MB")
    print(f"  - Efisiensi Storage  : Lebih hemat {compression_ratio:.1f}%")
    
    spark.stop()

if __name__ == "__main__":
    run_preprocessing()
