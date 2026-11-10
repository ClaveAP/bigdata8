"""
Modul Feature Engineering & ML Preparation (PySpark) — Bebas Kebocoran (Split-Before-Fit)
Membaca dataset Parquet (1.000.000 baris), membagi data menjadi Train (80%) dan Test (20%)
SEBELUM melakukan fitting pipeline transformasi fitur:
1. `features`: 43 fitur numerik + 14 StringIndexer (57 dimensi) dengan metadata kategorikal
   untuk model berbasis pohon (Decision Tree & Random Forest).
2. `features_lr`: 43 fitur numerik + 14 OneHotEncoder yang distandarisasi via StandardScaler
   untuk perbandingan adil pada Logistic Regression (bebas bias urutan kategori semu).
"""

import shutil
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from spark_session import get_spark_session
from pipeline_utils import (
    ALL_FEATURE_NAMES,
    CATEGORICAL_COLS,
    ID_COL,
    INDEXED_CATEGORICAL_COLS,
    NEGATIVE_CONTROL_COL,
    NUMERIC_COLS,
    OHE_CATEGORICAL_COLS,
    PROCESSED_PARQUET_DIR,
    TARGET_COL,
    TARGET_SOURCE_COL,
    TEST_PARQUET_DIR,
    TRAIN_PARQUET_DIR,
    save_json_metrics,
)


def run_feature_engineering():
    spark = get_spark_session(app_name="EcommerceFeatureEngineering")
    print(f"\n[INFO] Membaca dataset Parquet dari: {PROCESSED_PARQUET_DIR}")

    df = spark.read.parquet(str(PROCESSED_PARQUET_DIR))
    total_records = df.count()
    print(f"[SUCCESS] Data termuat: {total_records:,} baris x {len(df.columns)} kolom")

    # 1. Pembagian Dataset TERLEBIH DAHULU (Strict Split-Before-Fit)
    print("\n[STEP 1] Membagi Dataset menjadi Train Set (80%) dan Test Set (20%) sebelum fitting fitur...")
    train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)
    train_raw = train_raw.cache()
    test_raw = test_raw.cache()

    train_count = train_raw.count()
    test_count = test_raw.count()
    print(f"[INFO] Train Set : {train_count:,} baris ({train_count / total_records * 100:.2f}%)")
    print(f"[INFO] Test Set  : {test_count:,} baris ({test_count / total_records * 100:.2f}%)")

    # 2. Bangun Pipeline Transformasi Fitur (Fit HANYA pada Train Set)
    print("\n[STEP 2] Menyiapkan Pipeline StringIndexer (Tree) + OneHotEncoder & StandardScaler (LR)...")
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_COLS
    ]

    tree_assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + INDEXED_CATEGORICAL_COLS,
        outputCol="features",
    )

    ohe_encoder = OneHotEncoder(
        inputCols=INDEXED_CATEGORICAL_COLS,
        outputCols=OHE_CATEGORICAL_COLS,
        dropLast=True,
        handleInvalid="keep",
    )

    lr_raw_assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + OHE_CATEGORICAL_COLS,
        outputCol="features_lr_raw",
    )

    lr_scaler = StandardScaler(
        inputCol="features_lr_raw",
        outputCol="features_lr",
        withMean=False,
        withStd=True,
    )

    pipeline = Pipeline(
        stages=indexers + [tree_assembler, ohe_encoder, lr_raw_assembler, lr_scaler]
    )

    print("[INFO] Melakukan fitting Pipeline HANYA pada Train Set...")
    pipeline_model = pipeline.fit(train_raw)

    train_transformed = pipeline_model.transform(train_raw)
    test_transformed = pipeline_model.transform(test_raw)

    # Cek dimensi vektor OHE + Numerik
    sample_row = train_transformed.select("features", "features_lr").first()
    tree_dim = int(sample_row["features"].size)
    lr_dim = int(sample_row["features_lr"].size)
    print(f"[INFO] Dimensi vektor fitur Tree (`features`)   : {tree_dim} fitur")
    print(f"[INFO] Dimensi vektor fitur LR (`features_lr`) : {lr_dim} fitur (OHE + Scaled)")

    # 3. Simpan Train & Test Parquet
    # Sertakan kolom target & kontrol negatif di luar vektor fitur untuk analisis sensitivitas & kontrol negatif
    cols_to_save = [
        ID_COL,
        "features",
        "features_lr",
        TARGET_COL,
        TARGET_SOURCE_COL,
        NEGATIVE_CONTROL_COL,
    ]

    print("\n[STEP 3] Menyimpan Train & Test data ke Parquet...")
    for target_dir in (TRAIN_PARQUET_DIR, TEST_PARQUET_DIR):
        if target_dir.exists():
            shutil.rmtree(target_dir)

    train_transformed.select(*cols_to_save).write.mode("overwrite").option(
        "compression", "snappy"
    ).parquet(str(TRAIN_PARQUET_DIR))
    test_transformed.select(*cols_to_save).write.mode("overwrite").option(
        "compression", "snappy"
    ).parquet(str(TEST_PARQUET_DIR))

    # Distribusi kelas di Train & Test
    train_pos = train_raw.filter(F.col(TARGET_COL) == 1).count()
    test_pos = test_raw.filter(F.col(TARGET_COL) == 1).count()

    meta = {
        "total_records": int(total_records),
        "train_count": int(train_count),
        "test_count": int(test_count),
        "train_positive_count": int(train_pos),
        "train_negative_count": int(train_count - train_pos),
        "test_positive_count": int(test_pos),
        "test_negative_count": int(test_count - test_pos),
        "split_seed": 42,
        "split_before_fit": True,
        "numeric_feature_count": len(NUMERIC_COLS),
        "categorical_feature_count": len(CATEGORICAL_COLS),
        "tree_feature_vector_dim": tree_dim,
        "lr_ohe_scaled_vector_dim": lr_dim,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "all_feature_names": ALL_FEATURE_NAMES,
        "excluded_non_predictor_cols": [
            ID_COL,
            TARGET_SOURCE_COL,
            NEGATIVE_CONTROL_COL,
            "last_purchase_date (diturunkan menjadi days_since_last_purchase)",
        ],
    }
    out_meta = save_json_metrics("feature_metadata.json", meta)
    print(f"[SAVED] Metadata fitur tersimpan di: {out_meta}")
    print("[SUCCESS] Feature Engineering (Split-Before-Fit) Selesai 100%!")

    spark.stop()
    return meta


if __name__ == "__main__":
    run_feature_engineering()
