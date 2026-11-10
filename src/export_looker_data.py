"""
Modul Ekspor Data Mart untuk Google Looker Studio (Optimized & Deterministic)
Membaca dataset Parquet hasil olahan PySpark (1.000.000 baris) dan Champion Model,
melakukan agregasi data mart sesuai 5 tabel Looker Studio:
1. looker_demografi.csv
2. looker_metode_bayar.csv
3. looker_stres_impulse.csv
4. looker_scorecard_global.csv
5. looker_model_feature_importance.csv
"""

import pandas as pd
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.sql import functions as F
from spark_session import get_spark_session
from pipeline_utils import (
    ALL_FEATURE_NAMES,
    LOOKER_DIR,
    MODELS_DIR,
    PROCESSED_PARQUET_DIR,
)

BEST_MODEL_DIR = MODELS_DIR / "decision_tree_tuned"


def export_looker_tables():
    spark = get_spark_session(app_name="EcommerceExportLooker")
    LOOKER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Membaca data Parquet dari: {PROCESSED_PARQUET_DIR}")
    df = spark.read.parquet(str(PROCESSED_PARQUET_DIR)).cache()

    # 1. Tabel 1: Agregasi Demografi & Perangkat (Negara x Device x Gender = 10 x 3 x 4 = 120 baris)
    print("\n[STEP 1] Membuat Data Mart Demografi & Perangkat Ringkas...")
    df_demografi = (
        df.groupBy("country", "device_type", "gender")
        .agg(
            F.count("user_id").alias("total_users"),
            F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
            F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
            F.round(F.mean("monthly_spend"), 2).alias("avg_monthly_spend"),
            F.round(F.mean("daily_session_time_minutes"), 2).alias("avg_session_minutes"),
        )
        .orderBy("country", "device_type", "gender")
        .toPandas()
    )
    csv_demografi_path = LOOKER_DIR / "looker_demografi.csv"
    df_demografi.to_csv(csv_demografi_path, index=False)
    print(f"[SAVED] {csv_demografi_path.name} ({len(df_demografi)} baris)")

    # 2. Tabel 2: Agregasi Metode Pembayaran & Kategori Produk (6 x 8 = 48 baris)
    print("\n[STEP 2] Membuat Data Mart Metode Pembayaran & Kategori...")
    df_pembayaran = (
        df.groupBy("preferred_payment_method", "product_category_preference")
        .agg(
            F.count("user_id").alias("total_transactions"),
            F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
            F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
            F.round(F.mean("average_order_value"), 2).alias("avg_order_value"),
            F.round(F.mean("return_rate"), 2).alias("avg_return_rate"),
        )
        .orderBy("preferred_payment_method", "product_category_preference")
        .toPandas()
    )
    csv_pembayaran_path = LOOKER_DIR / "looker_metode_bayar.csv"
    df_pembayaran.to_csv(csv_pembayaran_path, index=False)
    print(f"[SAVED] {csv_pembayaran_path.name} ({len(df_pembayaran)} baris)")

    # 3. Tabel 3: Agregasi Stres Finansial & Belanja Impulsif (0..10 = 11 baris)
    print("\n[STEP 3] Membuat Data Mart Stres Finansial & Impulsivitas...")
    df_stres = (
        df.groupBy("stress_from_financial_decisions")
        .agg(
            F.count("user_id").alias("total_users"),
            F.round(F.mean("impulse_buying_score"), 2).alias("avg_impulse_score"),
            F.round(F.mean("cart_items_average"), 2).alias("avg_cart_items"),
            F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
            F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
        )
        .orderBy("stress_from_financial_decisions")
        .toPandas()
    )
    csv_stres_path = LOOKER_DIR / "looker_stres_impulse.csv"
    df_stres.to_csv(csv_stres_path, index=False)
    print(f"[SAVED] {csv_stres_path.name} ({len(df_stres)} baris)")

    # 4. Tabel 4: Ringkasan Global Scorecard (1 Baris Ringkas)
    df_global = df.agg(
        F.count("user_id").alias("total_users"),
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
        F.round(F.mean("monthly_spend"), 2).alias("avg_monthly_spend"),
        F.round(F.mean("average_order_value"), 2).alias("avg_order_value"),
    ).toPandas()
    csv_global_path = LOOKER_DIR / "looker_scorecard_global.csv"
    df_global.to_csv(csv_global_path, index=False)
    print(f"[SAVED] {csv_global_path.name} (Data Scorecard Utama)")

    # 5. Tabel 5: Top 10 Feature Importance dari Champion Model Decision Tree
    if BEST_MODEL_DIR.exists():
        print("\n[STEP 5] Mengekspor Feature Importance Champion Model untuk Looker Studio...")
        model = DecisionTreeClassificationModel.load(str(BEST_MODEL_DIR))
        importances = model.featureImportances.toArray()
        fi_df = (
            pd.DataFrame(
                {
                    "feature_name": [f.replace("_encoded", "") for f in ALL_FEATURE_NAMES],
                    "importance_score": [float(x) for x in importances],
                    "percentage_contribution": [round(float(x) * 100.0, 2) for x in importances],
                }
            )
            .sort_values(by="importance_score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        csv_fi_path = LOOKER_DIR / "looker_model_feature_importance.csv"
        fi_df.to_csv(csv_fi_path, index=False, float_format="%.6f")
        print(f"[SAVED] {csv_fi_path.name} (10 baris)")

    print("\n[SUCCESS] Seluruh 5 file Data Mart siap dipakai di Looker Studio!")
    spark.stop()


if __name__ == "__main__":
    export_looker_tables()
