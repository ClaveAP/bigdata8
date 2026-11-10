"""
Modul Pelatihan Model Decision Tree Classifier Baseline (PySpark MLlib)
Melatih model Decision Tree Baseline (maxDepth=5) pada data Train (80%),
mengevaluasi metrik lengkap (ROC-AUC, PR-AUC, Akurasi, F1, Precision, Recall, Confusion Matrix)
pada data Test (20%), melakukan sanitasi metadata untuk kompatibilitas Google Colab,
dan menyimpan metrik ke reports/metrics/baseline_dt_metrics.json.
"""

import time
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
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


def compute_confusion_and_multiclass_metrics(preds_df, label_col: str = TARGET_COL):
    """Menghitung TN, FP, FN, TP, Accuracy, Precision, Recall, F1 dalam satu agregasi Spark."""
    rows = (
        preds_df.groupBy(label_col, "prediction")
        .count()
        .collect()
    )
    counts = {(int(r[label_col]), int(r["prediction"])): int(r["count"]) for r in rows}
    tn = counts.get((0, 0), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)
    tp = counts.get((1, 1), 0)
    total = tn + fp + fn + tp

    acc = (tn + tp) / total if total > 0 else 0.0
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0

    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0

    n_pos = tp + fn
    n_neg = tn + fp
    weighted_prec = (prec_pos * n_pos + prec_neg * n_neg) / total if total > 0 else 0.0
    weighted_rec = (rec_pos * n_pos + rec_neg * n_neg) / total if total > 0 else 0.0
    weighted_f1 = (f1_pos * n_pos + f1_neg * n_neg) / total if total > 0 else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": round(acc, 6),
        "accuracy_pct": round(acc * 100.0, 2),
        "f1_weighted": round(weighted_f1, 6),
        "f1_positive": round(f1_pos, 6),
        "precision_weighted": round(weighted_prec, 6),
        "recall_weighted": round(weighted_rec, 6),
        "precision_positive": round(prec_pos, 6),
        "recall_positive": round(rec_pos, 6),
    }


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

    roc_eval = BinaryClassificationEvaluator(
        labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    roc_auc = float(roc_eval.evaluate(predictions))
    pr_auc = float(pr_eval.evaluate(predictions))
    cm_metrics = compute_confusion_and_multiclass_metrics(predictions, TARGET_COL)

    print("\n" + "=" * 55)
    print("HASIL EVALUASI DECISION TREE CLASSIFIER (BASELINE depth=5)")
    print("=" * 55)
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"PR-AUC             : {pr_auc:.4f}")
    print(f"Accuracy           : {cm_metrics['accuracy_pct']:.2f}%")
    print(f"F1-Score (Weighted): {cm_metrics['f1_weighted']:.4f}")
    print(f"Confusion Matrix   : TN={cm_metrics['tn']:,}, FP={cm_metrics['fp']:,}, FN={cm_metrics['fn']:,}, TP={cm_metrics['tp']:,}")
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
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        **cm_metrics,
        "rules_preview": humanize_tree_debug_string(model.toDebugString).splitlines()[:25],
    }
    save_json_metrics("baseline_dt_metrics.json", payload)
    spark.stop()
    return payload


if __name__ == "__main__":
    train_baseline_tree()
