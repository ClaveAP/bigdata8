"""
Modul Model Benchmarking & Comparison (PySpark MLlib)
Melatih dan membandingkan 3 algoritma klasifikasi pada Big Data (800k Train, 200k Test):
1. Logistic Regression (Linear Baseline)
2. Decision Tree Classifier (Interpretable White-box)
3. Random Forest Classifier (Non-linear Ensemble)

Mengevaluasi ROC-AUC, Akurasi, F1-Score, Training Time, dan kesesuaian rekomendasi bisnis.
"""

import time
from pathlib import Path
import pandas as pd
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARQUET_DIR = BASE_DIR / "data" / "processed" / "train_data_parquet"
TEST_PARQUET_DIR = BASE_DIR / "data" / "processed" / "test_data_parquet"

def run_model_comparison():
    spark = get_spark_session(app_name="EcommerceModelComparison")
    print(f"\n[INFO] Membaca data Train dari: {TRAIN_PARQUET_DIR}")
    train_df = spark.read.parquet(str(TRAIN_PARQUET_DIR))
    print(f"[INFO] Membaca data Test dari: {TEST_PARQUET_DIR}")
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR))

    # Evaluator standar industri
    roc_evaluator = BinaryClassificationEvaluator(labelCol="is_high_abandonment", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="f1")

    models_to_test = [
        {
            "name": "Logistic Regression (Linear Baseline)",
            "estimator": LogisticRegression(featuresCol="features", labelCol="is_high_abandonment", maxIter=20)
        },
        {
            "name": "Decision Tree Classifier (Champion Interpretable)",
            "estimator": DecisionTreeClassifier(featuresCol="features", labelCol="is_high_abandonment", maxDepth=7, minInstancesPerNode=20, seed=42)
        },
        {
            "name": "Random Forest Classifier (Ensemble)",
            "estimator": RandomForestClassifier(featuresCol="features", labelCol="is_high_abandonment", numTrees=20, maxDepth=7, seed=42)
        }
    ]

    benchmark_results = []

    print("\n" + "="*65)
    print("MEMULAI BENCHMARK KOMPARASI 3 MODEL MACHINE LEARNING (PYSPARK MLLIB)")
    print("="*65)

    for item in models_to_test:
        model_name = item["name"]
        clf = item["estimator"]
        print(f"\n[TRAINING] Melatih {model_name}...")
        
        start_time = time.time()
        model_fit = clf.fit(train_df)
        train_duration = time.time() - start_time
        
        print(f"[PREDICTING] Evaluasi pada Test Set (200.277 baris)...")
        preds = model_fit.transform(test_df)
        
        roc_auc = roc_evaluator.evaluate(preds)
        accuracy = acc_evaluator.evaluate(preds)
        f1 = f1_evaluator.evaluate(preds)
        
        print(f"--> Hasil {model_name}:")
        print(f"    Waktu Training : {train_duration:.2f} detik")
        print(f"    ROC-AUC        : {roc_auc:.4f}")
        print(f"    Accuracy       : {accuracy * 100:.2f}%")
        print(f"    F1-Score       : {f1:.4f}")

        benchmark_results.append({
            "Algorithm": model_name,
            "ROC-AUC": round(roc_auc, 4),
            "Accuracy": f"{accuracy * 100:.2f}%",
            "F1-Score": round(f1, 4),
            "Training Time (s)": round(train_duration, 2)
        })

    # Cetak Tabel Perbandingan
    comparison_df = pd.DataFrame(benchmark_results)
    print("\n" + "="*65)
    print("TABEL RINGKASAN BENCHMARK KOMPARASI MODEL")
    print("="*65)
    print(comparison_df.to_string(index=False))
    print("="*65)

    spark.stop()
    return comparison_df

if __name__ == "__main__":
    run_model_comparison()
