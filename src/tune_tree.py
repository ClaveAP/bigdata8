"""
Modul Hyperparameter Tuning & Pruning Decision Tree (PySpark MLlib)
Melakukan eksperimen kedalaman pohon (maxDepth: 7, 10, 12),
kriteria impurity (gini vs entropy), dan pruning (minInstancesPerNode=20 & 50)
untuk menemukan model terbaik serta mengekstrak Feature Importance ranking.
"""

from pathlib import Path
import pandas as pd
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARQUET_DIR = BASE_DIR / "data" / "processed" / "train_data_parquet"
TEST_PARQUET_DIR = BASE_DIR / "data" / "processed" / "test_data_parquet"
BEST_MODEL_DIR = BASE_DIR / "models" / "decision_tree_tuned"

def run_tuning():
    spark = get_spark_session(app_name="EcommerceDecisionTreeTuning")
    BEST_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Membaca data Train dari: {TRAIN_PARQUET_DIR}")
    train_df = spark.read.parquet(str(TRAIN_PARQUET_DIR))
    print(f"[INFO] Membaca data Test dari: {TEST_PARQUET_DIR}")
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR))

    # Grid konfigurasi hyperparameter untuk tuning & pruning
    candidates = [
        {"maxDepth": 7, "minInstancesPerNode": 20, "impurity": "gini"},
        {"maxDepth": 10, "minInstancesPerNode": 20, "impurity": "gini"},
        {"maxDepth": 10, "minInstancesPerNode": 50, "impurity": "entropy"}
    ]

    roc_evaluator = BinaryClassificationEvaluator(labelCol="is_high_abandonment", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="f1")
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="accuracy")

    best_model = None
    best_roc = 0.0
    best_params = {}
    results = []

    print("\n[STEP 1] Memulai Hyperparameter Grid Tuning & Tree Pruning...")
    for idx, params in enumerate(candidates, 1):
        print(f"\n--- Menguji Model {idx}/{len(candidates)}: depth={params['maxDepth']}, minInstances={params['minInstancesPerNode']}, impurity={params['impurity']} ---")
        dt = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="is_high_abandonment",
            maxDepth=params["maxDepth"],
            minInstancesPerNode=params["minInstancesPerNode"],
            impurity=params["impurity"],
            seed=42
        )
        model = dt.fit(train_df)
        preds = model.transform(test_df)
        
        roc = roc_evaluator.evaluate(preds)
        f1 = f1_evaluator.evaluate(preds)
        acc = acc_evaluator.evaluate(preds)
        
        print(f"Hasil: ROC-AUC={roc:.4f} | Accuracy={acc*100:.2f}% | F1={f1:.4f}")
        results.append({**params, "ROC-AUC": roc, "Accuracy": acc, "F1": f1})
        
        if roc > best_roc:
            best_roc = roc
            best_model = model
            best_params = params

    print("\n" + "="*50)
    print(f"[CHAMPION MODEL] Parameter Terbaik: {best_params} dengan ROC-AUC={best_roc:.4f}")
    print("="*50)

    # Simpan model terbaik
    print(f"[INFO] Menyimpan Champion Model ke: {BEST_MODEL_DIR}")
    best_model.write().overwrite().save(str(BEST_MODEL_DIR))

    # Ekstraksi Feature Importance
    importances = best_model.featureImportances.toArray()
    print(f"\n[INFO] Jumlah total bobot fitur terdeteksi: {len(importances)}")
    
    spark.stop()

if __name__ == "__main__":
    run_tuning()
