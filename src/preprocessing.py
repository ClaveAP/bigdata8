"""
Modul Pembersihan Data Skala Besar & Rekayasa Fitur Awal (PySpark)
Membaca CSV mentah 1.000.000 baris x 60 kolom, memvalidasi missing values pada seluruh 60 kolom,
menurunkan fitur durasi hari sejak transaksi terakhir (days_since_last_purchase),
membuat label target biner (is_high_abandonment berbasis cart_abandonment_rate >= 40.0%),
dan mengekspor ke format Apache Parquet Snappy + menyimpan laporan JSON ke reports/metrics/.
"""

import shutil
from pyspark.sql import functions as F
from spark_session import get_spark_session
from pipeline_utils import (
    DERIVED_DATE_COL,
    PROCESSED_PARQUET_DIR,
    RAW_DATA_PATH,
    RAW_DATE_COL,
    REFERENCE_MAX_DATE,
    TARGET_COL,
    TARGET_SOURCE_COL,
    save_json_metrics,
)


def run_preprocessing():
    spark = get_spark_session(app_name="EcommerceDataPreprocessing")
    print(f"\n[INFO] Membaca dataset mentah: {RAW_DATA_PATH}")

    df = spark.read.csv(str(RAW_DATA_PATH), header=True, inferSchema=True)
    initial_count = df.count()
    raw_cols = df.columns
    print(f"[SUCCESS] Data termuat: {initial_count:,} baris x {len(raw_cols)} kolom")

    # 1. Validasi Missing Values pada SELURUH 60 kolom
    print("\n[STEP 1] Validasi Missing Values pada seluruh 60 kolom...")
    null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in raw_cols]
    null_dict = df.select(null_exprs).collect()[0].asDict()
    total_nulls = int(sum(null_dict.values()))
    print(f"[INFO] Total nilai Null di seluruh 60 kolom: {total_nulls}")

    df_clean = df.dropna() if total_nulls > 0 else df
    clean_count = df_clean.count()

    # 2. Rekayasa Fitur Tanggal & Target Biner (is_high_abandonment)
    print(f"\n[STEP 2] Menurunkan '{DERIVED_DATE_COL}' & target biner '{TARGET_COL}' ({TARGET_SOURCE_COL} >= 40.0%)...")
    df_transformed = (
        df_clean.withColumn(
            DERIVED_DATE_COL,
            F.datediff(F.lit(REFERENCE_MAX_DATE), F.to_date(F.col(RAW_DATE_COL))).cast("int"),
        )
        .withColumn(
            TARGET_COL,
            F.when(F.col(TARGET_SOURCE_COL) >= 40.0, 1).otherwise(0),
        )
    )

    # Distribusi Kelas Target
    dist_rows = (
        df_transformed.groupBy(TARGET_COL)
        .agg(
            F.count("user_id").alias("total_users"),
            F.round(F.count("user_id") / clean_count * 100, 2).alias("percentage"),
        )
        .orderBy(TARGET_COL)
        .collect()
    )
    class_dist = {
        str(int(r[TARGET_COL])): {
            "count": int(r["total_users"]),
            "percentage": float(r["percentage"]),
        }
        for r in dist_rows
    }
    print("[INFO] Distribusi Kelas Target:", class_dist)

    # 3. Ekspor ke Apache Parquet (Snappy Compressed)
    print(f"\n[STEP 3] Menyimpan data terproses ke Apache Parquet di: {PROCESSED_PARQUET_DIR}")
    if PROCESSED_PARQUET_DIR.exists():
        shutil.rmtree(PROCESSED_PARQUET_DIR)

    (
        df_transformed.write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(str(PROCESSED_PARQUET_DIR))
    )

    raw_size_mb = RAW_DATA_PATH.stat().st_size / (1024 * 1024)
    parquet_size_mb = sum(
        f.stat().st_size for f in PROCESSED_PARQUET_DIR.glob("**/*.parquet")
    ) / (1024 * 1024)
    compression_ratio = (1.0 - (parquet_size_mb / raw_size_mb)) * 100.0

    summary = {
        "raw_rows": int(initial_count),
        "clean_rows": int(clean_count),
        "raw_columns_count": len(raw_cols),
        "processed_columns_count": len(df_transformed.columns),
        "total_nulls_across_60_cols": total_nulls,
        "raw_csv_size_mb": round(raw_size_mb, 2),
        "parquet_size_mb": round(parquet_size_mb, 2),
        "storage_reduction_pct": round(compression_ratio, 2),
        "target_threshold_pct": 40.0,
        "class_distribution": class_dist,
    }
    out_json = save_json_metrics("preprocessing_summary.json", summary)
    print(f"[SAVED] Ringkasan preprocessing disimpan ke: {out_json}")
    print(
        f"[SUCCESS] CSV {raw_size_mb:.2f} MB -> Parquet {parquet_size_mb:.2f} MB (Hemat {compression_ratio:.1f}%)"
    )

    spark.stop()
    return summary


if __name__ == "__main__":
    run_preprocessing()
