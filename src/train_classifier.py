"""
Modul Pelatihan Model Decision Tree Classifier Baseline (PySpark MLlib)
Melatih model Decision Tree Baseline (maxDepth=5) pada data Train (80%),
mengevaluasi metrik lengkap (ROC-AUC, PR-AUC, Akurasi, F1, Precision, Recall, Confusion Matrix)
pada data Test (20%), melakukan sanitasi metadata untuk kompatibilitas Google Colab,
dan menyimpan metrik ke reports/metrics/baseline_dt_metrics.json.
"""

import time
from pyspark.ml.classification import DecisionTreeClassifier
from compare_models import evaluate_spark_predictions_single_pass
from spark_session import get_spark_session
from pipeline_utils import (
    MODELS_DIR,
    TARGET_COL,
    TEST_PARQUET_DIR,
    TRAIN_PARQUET_DIR,
    humanize_tree_debug_string,
    sanitize_spark_model_metadata,
    save_json_metrics,
)


def train_baseline_tree():
    spark = get_spark_session(app_name="EcommerceDecisionTreeBaseline")
    baseline_dir = MODELS_DIR / "decision_tree_baseline"
    baseline_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Membaca data Train dari: {TRAIN_PARQUET_DIR}")
    train_df = spark.read.parquet(str(TRAIN_PARQUET_DIR)).select("features", TARGET_COL).cache()
    print(f"[INFO] Membaca data Test dari: {TEST_PARQUET_DIR}")
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR)).select("features", TARGET_COL).cache()

    train_n = train_df.count()
    test_n = test_df.count()
    print(f"[INFO] Train: {train_n:,} baris | Test: {test_n:,} baris")

    print("\n[STEP 1] Melatih Decision Tree Baseline (maxDepth=5, minInstancesPerNode=20, impurity=gini, seed=42)...")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol=TARGET_COL,
        maxDepth=5,
        minInstancesPerNode=20,
        impurity="gini",
        seed=42,
    )

    t0 = time.perf_counter()
    model = dt.fit(train_df)
    train_time_s = time.perf_counter() - t0
    print(f"[SUCCESS] Model Baseline dilatih dalam {train_time_s:.2f} detik (Nodes={model.numNodes}, Depth={model.depth})")

    print("\n[STEP 2] Evaluasi pada Test Set...")
    predictions = model.transform(test_df)
    eval_m, _, _, _ = evaluate_spark_predictions_single_pass(predictions, TARGET_COL)

    print("\n" + "=" * 55)
    print("HASIL EVALUASI DECISION TREE CLASSIFIER (BASELINE depth=5)")
    print("=" * 55)
    print(f"ROC-AUC            : {eval_m['roc_auc']:.4f}")
    print(f"PR-AUC             : {eval_m['pr_auc']:.4f}")
    print(f"Accuracy           : {eval_m['accuracy_pct']:.2f}%")
    print(f"F1-Score (Weighted): {eval_m['f1_weighted']:.4f}")
    print(f"Confusion Matrix   : TN={eval_m['tn']:,}, FP={eval_m['fp']:,}, FN={eval_m['fn']:,}, TP={eval_m['tp']:,}")
    print("=" * 55)

    print(f"\n[STEP 3] Menyimpan model terlatih ke: {baseline_dir}")
    model.write().overwrite().save(str(baseline_dir))
    cleaned = sanitize_spark_model_metadata(baseline_dir)
    print(f"[INFO] Sanitasi metadata Colab (`pruneTree`): {cleaned} file dibersihkan.")

    payload = {
        "model_name": "DecisionTree_Baseline_d5",
        "params": {"maxDepth": 5, "minInstancesPerNode": 20, "impurity": "gini", "seed": 42},
        "num_nodes": int(model.numNodes),
        "effective_depth": int(model.depth),
        "train_rows": int(train_n),
        "test_rows": int(test_n),
        "train_time_s": round(train_time_s, 3),
        **eval_m,
        "rules_preview": humanize_tree_debug_string(model.toDebugString).splitlines()[:25],
    }
    save_json_metrics("baseline_dt_metrics.json", payload)
    spark.stop()
    return payload


if __name__ == "__main__":
    train_baseline_tree()
