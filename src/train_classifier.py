"""
Modul Pelatihan Model Decision Tree Classifier (PySpark MLlib)
Melatih model Decision Tree Baseline pada data Train (799.723 baris),
mengevaluasi metrik (ROC-AUC, Akurasi, F1-Score) pada data Test (200.277 baris),
dan mengekstrak Feature Importance serta pohon keputusan.
"""

from pathlib import Path
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from spark_session import get_spark_session

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARQUET_DIR = BASE_DIR / "data" / "processed" / "train_data_parquet"
TEST_PARQUET_DIR = BASE_DIR / "data" / "processed" / "test_data_parquet"
MODELS_DIR = BASE_DIR / "models" / "decision_tree_baseline"

def train_baseline_tree():
    spark = get_spark_session(app_name="EcommerceDecisionTreeBaseline")
    MODELS_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Membaca data Train dari: {TRAIN_PARQUET_DIR}")
    train_df = spark.read.parquet(str(TRAIN_PARQUET_DIR))
    print(f"[INFO] Membaca data Test dari: {TEST_PARQUET_DIR}")
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR))

    # 1. Inisialisasi Decision Tree Classifier Baseline
    print("\n[STEP 1] Menginisialisasi Decision Tree Classifier (maxDepth=5, seed=42)...")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="is_high_abandonment",
        maxDepth=5,
        seed=42
    )

    # 2. Pelatihan Model (Training di Spark MLlib)
    print("[STEP 2] Memulai proses training pada 799.723 baris...")
    model = dt.fit(train_df)
    print("[SUCCESS] Model Decision Tree berhasil dilatih!")

    # 3. Prediksi pada Test Set (200.277 baris)
    print("\n[STEP 3] Melakukan prediksi pada Test Set...")
    predictions = model.transform(test_df)
    
    # 4. Evaluasi Metrik
    roc_evaluator = BinaryClassificationEvaluator(labelCol="is_high_abandonment", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="f1")
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="accuracy")
    prec_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="weightedPrecision")
    rec_evaluator = MulticlassClassificationEvaluator(labelCol="is_high_abandonment", predictionCol="prediction", metricName="weightedRecall")

    roc_auc = roc_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    acc = acc_evaluator.evaluate(predictions)
    prec = prec_evaluator.evaluate(predictions)
    rec = rec_evaluator.evaluate(predictions)

    print("\n" + "="*50)
    print("HASIL EVALUASI DECISION TREE CLASSIFIER (BASELINE)")
    print("="*50)
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"Accuracy           : {acc * 100:.2f}%")
    print(f"F1-Score           : {f1:.4f}")
    print(f"Weighted Precision : {prec:.4f}")
    print(f"Weighted Recall    : {rec:.4f}")
    print("="*50)

    # 5. Struktur Pohon Keputusan (IF-THEN Rules Ringkas)
    print("\n[STEP 4] Ekstraksi Representasi Pohon Keputusan:")
    print(model.toDebugString[:800] + "\n... [truncated for readability]")

    # 6. Simpan Model Artefak
    print(f"\n[STEP 5] Menyimpan model terlatih ke: {MODELS_DIR}")
    model.write().overwrite().save(str(MODELS_DIR))
    print("[SUCCESS] Artefak model tersimpan!")

    spark.stop()

if __name__ == "__main__":
    train_baseline_tree()
