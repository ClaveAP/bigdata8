"""
Modul Ekspor Data Mart untuk Google Looker Studio (Optimized Compact)
Membaca dataset Parquet hasil olahan PySpark (1.000.000 baris),
melakukan agregasi data mart sesuai dimensi laporan (Demografi, Pembayaran, Psikologis),
dan menyimpannya ke format CSV ringkas di folder reports/looker_data/
"""

from pathlib import Path
from pyspark.sql import functions as F
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_DATA_PATH = BASE_DIR / "data" / "processed" / "ecommerce_processed_parquet"
LOOKER_DIR = BASE_DIR / "reports" / "looker_data"

def export_looker_tables():
    spark = get_spark_session(app_name="EcommerceExportLooker")
    LOOKER_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Membaca data Parquet dari: {PARQUET_DATA_PATH}")
    df = spark.read.parquet(str(PARQUET_DATA_PATH))

    # 1. Tabel 1: Agregasi Demografi & Perangkat (Ringkas: Negara x Device x Gender)
    print("\n[STEP 1] Membuat Data Mart Demografi & Perangkat Ringkas...")
    df_demografi = df.groupBy("country", "device_type", "gender").agg(
        F.count("user_id").alias("total_users"),
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
        F.round(F.mean("monthly_spend"), 2).alias("avg_monthly_spend"),
        F.round(F.mean("daily_session_time_minutes"), 2).alias("avg_session_minutes")
    ).toPandas()
    
    csv_demografi_path = LOOKER_DIR / "looker_demografi.csv"
    df_demografi.to_csv(csv_demografi_path, index=False)
    print(f"[SAVED] {csv_demografi_path.name} ({len(df_demografi)} baris agregasi, super cepat di Looker Studio)")

    # 2. Tabel 2: Agregasi Metode Pembayaran & Kategori Produk
    print("\n[STEP 2] Membuat Data Mart Metode Pembayaran & Kategori...")
    df_pembayaran = df.groupBy("preferred_payment_method", "product_category_preference").agg(
        F.count("user_id").alias("total_transactions"),
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
        F.round(F.mean("average_order_value"), 2).alias("avg_order_value"),
        F.round(F.mean("return_rate"), 2).alias("avg_return_rate")
    ).toPandas()
    
    csv_pembayaran_path = LOOKER_DIR / "looker_metode_bayar.csv"
    df_pembayaran.to_csv(csv_pembayaran_path, index=False)
    print(f"[SAVED] {csv_pembayaran_path.name} ({len(df_pembayaran)} baris)")

    # 3. Tabel 3: Agregasi Stres Finansial & Belanja Impulsif
    print("\n[STEP 3] Membuat Data Mart Stres Finansial & Impulsivitas...")
    df_stres = df.groupBy("stress_from_financial_decisions").agg(
        F.count("user_id").alias("total_users"),
        F.round(F.mean("impulse_buying_score"), 2).alias("avg_impulse_score"),
        F.round(F.mean("cart_items_average"), 2).alias("avg_cart_items"),
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate")
    ).orderBy("stress_from_financial_decisions").toPandas()
    
    csv_stres_path = LOOKER_DIR / "looker_stres_impulse.csv"
    df_stres.to_csv(csv_stres_path, index=False)
    print(f"[SAVED] {csv_stres_path.name} ({len(df_stres)} baris)")

    # 4. Tabel 4: Ringkasan Global Scorecard (1 Baris Ringkas)
    df_global = df.agg(
        F.count("user_id").alias("total_users"),
        F.round(F.mean("checkout_abandonments_per_month"), 2).alias("avg_checkout_abandon"),
        F.round(F.mean("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
        F.round(F.mean("monthly_spend"), 2).alias("avg_monthly_spend"),
        F.round(F.mean("average_order_value"), 2).alias("avg_order_value")
    ).toPandas()
    csv_global_path = LOOKER_DIR / "looker_scorecard_global.csv"
    df_global.to_csv(csv_global_path, index=False)
    print(f"[SAVED] {csv_global_path.name} (Data Scorecard Utama)")

    print(f"\n[SUCCESS] Seluruh file Data Mart siap dipakai di Looker Studio!")
    spark.stop()

if __name__ == "__main__":
    export_looker_tables()
