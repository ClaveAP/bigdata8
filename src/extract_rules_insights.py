"""
Modul Ekstraksi Feature Importance, IF-THEN Rules, & Analisis Bisnis
Memuat Champion Model Decision Tree dari disk, memetakan bobot fitur ke nama kolom asli,
menghasilkan plot Top-15 Feature Importance, dan mengekstrak aturan bisnis operasional.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.classification import DecisionTreeClassificationModel
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
BEST_MODEL_DIR = BASE_DIR / "models" / "decision_tree_tuned"
REPORTS_DIR = BASE_DIR / "reports"

def run_extraction():
    spark = get_spark_session(app_name="EcommerceExtractRules")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Memuat Champion Model dari: {BEST_MODEL_DIR}")
    model = DecisionTreeClassificationModel.load(str(BEST_MODEL_DIR))

    # Daftar nama fitur asli sesuai urutan VectorAssembler di feature_engineering.py
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
    categorical_cols = [
        "gender", "country", "urban_rural", "employment_status",
        "education_level", "relationship_status", "occupation",
        "ethnicity", "language_preference", "device_type",
        "preferred_payment_method", "product_category_preference",
        "shopping_time_of_day", "budgeting_style"
    ]
    all_feature_names = numeric_cols + [f"{c}_encoded" for c in categorical_cols]

    # 1. Feature Importance Mapping
    importances = model.featureImportances.toArray()
    fi_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\n" + "="*50)
    print("TOP 10 FITUR PALING BERPENGARUH (FEATURE IMPORTANCE)")
    print("="*50)
    print(fi_df.head(10).to_string(index=False))
    print("="*50)

    # 2. Visualisasi Feature Importance
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    top_fi = fi_df.head(12)
    ax = sns.barplot(data=top_fi, x="Importance", y="Feature", palette="viridis")
    plt.title("Top 12 Faktor Pendorong Pembatalan Transaksi (Decision Tree)", fontsize=13, fontweight="bold")
    plt.xlabel("Tingkat Kepentingan Fitur (Feature Importance Score)", fontsize=11)
    plt.ylabel("Fitur Perilaku / Demografi", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.4f}", (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plot_path = REPORTS_DIR / "decision_tree_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[SAVED] Plot tersimpan: {plot_path.name}")

    # 3. Ekstraksi IF-THEN Rules
    tree_rules = model.toDebugString
    rules_path = REPORTS_DIR / "decision_tree_rules.txt"
    with open(rules_path, "w") as f:
        f.write(tree_rules)
    print(f"[SAVED] Aturan keputusan lengkap tersimpan di: {rules_path.name}")

    print("\n[SUCCESS] Ekstraksi Feature Importance & IF-THEN Rules Selesai!")
    spark.stop()

if __name__ == "__main__":
    run_extraction()
