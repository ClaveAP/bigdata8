"""
Modul Feature Engineering & ML Preparation (PySpark)
Membaca dataset Parquet, melakukan encoding variabel kategorikal via StringIndexer,
menggabungkan fitur dengan VectorAssembler, membagi data (Train 80% / Test 20%),
dan menyimpan fitur siap latih ke Parquet.
"""

from pathlib import Path
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PARQUET_DIR = BASE_DIR / "data" / "processed" / "ecommerce_processed_parquet"
TRAIN_PARQUET_DIR = BASE_DIR / "data" / "processed" / "train_data_parquet"
TEST_PARQUET_DIR = BASE_DIR / "data" / "processed" / "test_data_parquet"

def run_feature_engineering():
    spark = get_spark_session(app_name="EcommerceFeatureEngineering")
    print(f"\n[INFO] Membaca dataset Parquet dari: {INPUT_PARQUET_DIR}")
    
    df = spark.read.parquet(str(INPUT_PARQUET_DIR))
    total_records = df.count()
    print(f"[SUCCESS] Data termuat: {total_records:,} baris")

    # 1. Identifikasi Fitur Kategorikal untuk StringIndexer
    categorical_cols = [
        "gender", "country", "urban_rural", "employment_status",
        "education_level", "relationship_status", "occupation",
        "ethnicity", "language_preference", "device_type",
        "preferred_payment_method", "product_category_preference",
        "shopping_time_of_day", "budgeting_style"
    ]
    
    # Fitur Numerik Utama (Demografi, Perilaku, Finansial, Sesi)
    numeric_cols = [
        "age", "income_level", "household_size", "has_children",
        "weekly_purchases", "monthly_spend", "average_order_value",
        "coupon_usage_frequency", "loyalty_program_member", "referral_count",
        "weekend_shopper", "impulse_purchases_per_month", "browse_to_buy_ratio",
        "return_frequency", "brand_loyalty_score", "impulse_buying_score",
        "environmental_consciousness", "health_conscious_shopping", "travel_frequency",
        "hobby_count", "social_media_influence_score", "exercise_frequency",
        "stress_from_financial_decisions", "overall_stress_level", "sleep_quality",
        "physical_activity_level", "mental_health_score", "daily_session_time_minutes",
        "product_views_per_day", "ad_views_per_day", "ad_clicks_per_day",
        "wishlist_items_count", "cart_items_average", "purchase_conversion_rate",
        "notification_response_rate", "account_age_months", "social_sharing_frequency",
        "premium_subscription", "return_rate"
    ]

    print("\n[STEP 1] Menyiapkan Pipeline StringIndexer untuk Fitur Kategorikal...")
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep")
        for col_name in categorical_cols
    ]
    
    # 2. VectorAssembler: Menggabungkan seluruh fitur ke dalam 1 vektor fitur 'features'
    indexed_categorical_cols = [f"{col_name}_idx" for col_name in categorical_cols]
    feature_columns = numeric_cols + indexed_categorical_cols
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Eksekusi Pipeline Transformasi Fitur
    pipeline = Pipeline(stages=indexers + [assembler])
    pipeline_model = pipeline.fit(df)
    df_assembled = pipeline_model.transform(df)

    # 3. Pembagian Dataset (Train-Test Split 80:20)
    print("\n[STEP 2] Membagi Dataset menjadi Train Set (80%) dan Test Set (20%)...")
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
    
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"[INFO] Train Set : {train_count:,} baris")
    print(f"[INFO] Test Set  : {test_count:,} baris")

    # 4. Simpan Train & Test ke Parquet agar siap dilatih oleh Decision Tree
    print("\n[STEP 3] Menyimpan Train & Test data ke Parquet...")
    train_df.select("features", "is_high_abandonment").write.mode("overwrite").parquet(str(TRAIN_PARQUET_DIR))
    test_df.select("features", "is_high_abandonment").write.mode("overwrite").parquet(str(TEST_PARQUET_DIR))
    
    print("[SUCCESS] Feature Engineering & Dataset Split Selesai 100%!")
    spark.stop()

if __name__ == "__main__":
    run_feature_engineering()
