"""
Modul Pembersihan Data Skala Besar & Rekayasa Fitur Awal (PySpark)
Membaca CSV mentah 1.000.000 baris, memvalidasi missing values,
membuat label target biner (is_high_abandonment) berbasis cart_abandonment_rate (>= 40.0%),
dan mengekspor ke format Apache Parquet.
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
    
    # 1. Handling Missing Values pada kolom krusial
    print("\n[STEP 1] Validasi & Pembersihan Missing Values...")
    crucial_cols = [
        "user_id", "cart_abandonment_rate", "stress_from_financial_decisions",
        "impulse_buying_score", "monthly_spend", "daily_session_time_minutes", "device_type"
    ]
    df_clean = df.dropna(subset=crucial_cols)
    clean_count = df_clean.count()
    print(f"[INFO] Baris setelah validasi: {clean_count:,} baris")
    
    # 2. Pembuatan Fitur Target Biner (is_high_abandonment)
    # Rata-rata cart_abandonment_rate di dataset adalah 40.21%.
    # Threshold: >= 40.0% = 1 (High Abandonment Risk), < 40.0% = 0 (Low Abandonment Risk)
    print("\n[STEP 2] Rekayasa Variabel Target (is_high_abandonment berbasis cart_abandonment_rate >= 40%)...")
    df_transformed = df_clean.withColumn(
        "is_high_abandonment",
        F.when(F.col("cart_abandonment_rate") >= 40.0, 1).otherwise(0)
    )
    
    # Cek distribusi kelas target
    print("[INFO] Distribusi Kelas Target (is_high_abandonment):")
    df_transformed.groupBy("is_high_abandonment").agg(
        F.count("user_id").alias("total_users"),
        F.round(F.count("user_id") / clean_count * 100, 2).alias("percentage")
    ).show()
    
    # 3. Ekspor ke Apache Parquet (Snappy Compressed)
    print(f"\n[STEP 3] Menyimpan data terproses ke format Apache Parquet di: {PROCESSED_DATA_DIR}")
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
        
    (
        df_transformed.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(str(PROCESSED_DATA_DIR))
    )
    
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
