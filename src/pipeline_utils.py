"""
Utilitas Terpusat Pipeline Big Data & Single Source of Truth (SSOT)
Menyimpan definisi kolom resmi (60 kolom mentah -> 57 fitur prediktor independen),
fungsi sanitasi kompatibilitas model PySpark lintas versi (Spark 4.3 lokal vs Spark 4.0 Colab),
dan helper pembacaan/penulisan metrik JSON terstruktur di reports/metrics/.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "e_commerce_shopper_behaviour_and_lifestyle.csv"
PROCESSED_PARQUET_DIR = BASE_DIR / "data" / "processed" / "ecommerce_processed_parquet"
TRAIN_PARQUET_DIR = BASE_DIR / "data" / "processed" / "train_data_parquet"
TEST_PARQUET_DIR = BASE_DIR / "data" / "processed" / "test_data_parquet"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
LOOKER_DIR = REPORTS_DIR / "looker_data"

# Identitas & kolom target/kontrol (DILARANG masuk ke VectorAssembler prediktor)
ID_COL = "user_id"
TARGET_SOURCE_COL = "cart_abandonment_rate"
NEGATIVE_CONTROL_COL = "checkout_abandonments_per_month"
TARGET_COL = "is_high_abandonment"
RAW_DATE_COL = "last_purchase_date"
DERIVED_DATE_COL = "days_since_last_purchase"
REFERENCE_MAX_DATE = "2027-01-01"

# 43 Fitur Numerik Independen (termasuk 3 fitur numerik yang sebelumnya terlewat + 1 fitur durasi dari tanggal)
NUMERIC_COLS: List[str] = [
    "age",
    "income_level",
    "household_size",
    "has_children",
    "weekly_purchases",
    "monthly_spend",
    "average_order_value",
    "coupon_usage_frequency",
    "loyalty_program_member",
    "referral_count",
    "weekend_shopper",
    "impulse_purchases_per_month",
    "browse_to_buy_ratio",
    "return_frequency",
    "brand_loyalty_score",
    "impulse_buying_score",
    "environmental_consciousness",
    "health_conscious_shopping",
    "travel_frequency",
    "hobby_count",
    "social_media_influence_score",
    "exercise_frequency",
    "stress_from_financial_decisions",
    "overall_stress_level",
    "sleep_quality",
    "physical_activity_level",
    "mental_health_score",
    "daily_session_time_minutes",
    "product_views_per_day",
    "ad_views_per_day",
    "ad_clicks_per_day",
    "wishlist_items_count",
    "cart_items_average",
    "purchase_conversion_rate",
    "notification_response_rate",
    "account_age_months",
    "social_sharing_frequency",
    "premium_subscription",
    "return_rate",
    "review_writing_frequency",
    "reading_habits",
    "app_usage_frequency",
    DERIVED_DATE_COL,
]

# 14 Fitur Kategorikal Nominal/Ordinal
CATEGORICAL_COLS: List[str] = [
    "gender",
    "country",
    "urban_rural",
    "employment_status",
    "education_level",
    "relationship_status",
    "occupation",
    "ethnicity",
    "language_preference",
    "device_type",
    "preferred_payment_method",
    "product_category_preference",
    "shopping_time_of_day",
    "budgeting_style",
]

INDEXED_CATEGORICAL_COLS: List[str] = [f"{c}_idx" for c in CATEGORICAL_COLS]
OHE_CATEGORICAL_COLS: List[str] = [f"{c}_ohe" for c in CATEGORICAL_COLS]

# Urutan nama fitur di dalam vektor `features` (Total = 43 + 14 = 57 fitur)
ALL_FEATURE_NAMES: List[str] = NUMERIC_COLS + [f"{c}_encoded" for c in CATEGORICAL_COLS]


def ensure_dirs() -> None:
    """Memastikan direktori output tersedia."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOOKER_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_json_metrics(filename: str, payload: Dict[str, Any]) -> Path:
    """Menyimpan dictionary metrik ke reports/metrics/<filename>."""
    ensure_dirs()
    out_path = METRICS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def load_json_metrics(filename: str) -> Dict[str, Any]:
    """Membaca dictionary metrik dari reports/metrics/<filename>."""
    in_path = METRICS_DIR / filename
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_spark_model_metadata(model_dir: Path) -> int:
    """
    Menghapus parameter 'pruneTree' dari metadata JSON model PySpark 4.3+
    agar model dapat dimuat tanpa NoSuchElementException pada PySpark 4.0.x (Google Colab),
    serta membersihkan file checksum .crc terkait.
    """
    meta_dir = Path(model_dir) / "metadata"
    if not meta_dir.exists():
        return 0

    cleaned_count = 0
    for meta_file in meta_dir.glob("part-*"):
        try:
            raw_json = json.loads(meta_file.read_text(encoding="utf-8"))
            changed = False
            for section in ("paramMap", "defaultParamMap"):
                if section in raw_json and "pruneTree" in raw_json[section]:
                    del raw_json[section]["pruneTree"]
                    changed = True
            if changed:
                meta_file.write_text(json.dumps(raw_json), encoding="utf-8")
                cleaned_count += 1
        except Exception:
            continue

    if cleaned_count > 0:
        for crc_file in meta_dir.glob(".*crc"):
            try:
                crc_file.unlink()
            except OSError:
                pass
    return cleaned_count


def humanize_tree_debug_string(debug_str: str, feature_names: List[str] = ALL_FEATURE_NAMES) -> str:
    """
    Mengganti token 'feature N' pada DecisionTreeClassificationModel.toDebugString
    dengan nama kolom asli agar aturan IF-THEN langsung terbaca manusia.
    """
    def _replace(match: re.Match) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(feature_names):
            return f"{feature_names[idx]} [f{idx}]"
        return match.group(0)

    return re.sub(r"\bfeature (\d+)\b", _replace, debug_str)
