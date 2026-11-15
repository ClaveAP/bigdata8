"""
Modul Hyperparameter Tuning & Pruning Decision Tree (PySpark MLlib) — Bebas Kebocoran Uji (E2)
Melakukan eksperimen faktorial penuh pada Validation Split (pecahan 75:25 dari Train Set):
- maxDepth: [2, 3, 4, 5, 6, 7, 8, 10, 12]
- minInstancesPerNode: [1, 20, 50, 100]
- impurity: ["gini", "entropy"]
Total = 9 x 4 x 2 = 72 konfigurasi untuk memetakan kurva Pruning, mengisolasi efek Gini vs Entropy,
dan memilih Champion Model murni dari Validation Set sebelum dilatih ulang pada 100% Full Train Set (799.723 baris)
dan dievaluasi satu kali pada 100% Test Set (200.277 baris) menggunakan PySpark MLlib.
"""

import time
import pandas as pd
from pyspark.ml.classification import DecisionTreeClassifier
from spark_session import get_spark_session
from compare_models import evaluate_spark_predictions_single_pass
from pipeline_utils import (
    ALL_FEATURE_NAMES,
    METRICS_DIR,
    MODELS_DIR,
    TARGET_COL,
    TEST_PARQUET_DIR,
    TRAIN_PARQUET_DIR,
    sanitize_spark_model_metadata,
    save_json_metrics,
)

BEST_MODEL_DIR = MODELS_DIR / "decision_tree_tuned"


def run_tuning():
    spark = get_spark_session(app_name="EcommerceDecisionTreeTuning")
    BEST_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Membaca data Train dari: {TRAIN_PARQUET_DIR}")
    full_train_df = spark.read.parquet(str(TRAIN_PARQUET_DIR)).select("features", TARGET_COL).cache()
    print(f"[INFO] Membaca data Test dari: {TEST_PARQUET_DIR}")
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR)).select("features", TARGET_COL).cache()

    # Pecah Train Set (80%) menjadi Train-Sub (75% dari Train = ~60% total) & Validation (25% dari Train = ~20% total)
    print("\n[STEP 1] Membagi Train Set menjadi Train-Sub (75%) dan Validation Set (25%) untuk Tuning Bebas Bocor...")
    train_sub_df, val_df = full_train_df.randomSplit([0.75, 0.25], seed=42)

    n_full_train = full_train_df.count()
    n_train_sub = train_sub_df.count()
    n_val = val_df.count()
    n_test = test_df.count()
    print(
        f"[INFO] Train Penuh: {n_full_train:,} | Train-Sub: {n_train_sub:,} | Validation: {n_val:,} | Test (Untouched): {n_test:,}"
    )

    # Evaluasi seluruh 72 konfigurasi faktorial 100% menggunakan PySpark DecisionTreeClassifier murni
    # pada sub-sampel konsisten dari Train-Sub (25%) dan Validation Split (50%) agar seluruh kandidat adil & bebas bias mesin.
    sub_tr_spark = train_sub_df.sample(withReplacement=False, fraction=0.25, seed=42).cache()
    sub_va_spark = val_df.sample(withReplacement=False, fraction=0.50, seed=42).cache()
    n_sub_tr = sub_tr_spark.count()
    n_sub_va = sub_va_spark.count()
    print(f"[INFO] Sub-sampel Validasi PySpark Murni: Train-Sub={n_sub_tr:,} | Validation={n_sub_va:,}")

    depths = [2, 3, 4, 5, 6, 7, 8, 10, 12]
    min_instances_list = [1, 20, 50, 100]
    impurities = ["gini", "entropy"]

    grid_rows = []
    total_configs = len(depths) * len(min_instances_list) * len(impurities)
    print(f"\n[STEP 2] Menjalankan Eksperimen Faktorial Penuh ({total_configs} Konfigurasi) 100% PySpark MLlib pada Validation Set...")

    config_idx = 0
    for depth in depths:
        for min_inst in min_instances_list:
            for imp in impurities:
                config_idx += 1
                dt_sv = DecisionTreeClassifier(
                    featuresCol="features",
                    labelCol=TARGET_COL,
                    maxDepth=depth,
                    minInstancesPerNode=min_inst,
                    impurity=imp,
                    seed=42,
                )
                t0 = time.perf_counter()
                m_sv = dt_sv.fit(sub_tr_spark)
                fit_time_s = time.perf_counter() - t0
                em_sv, _, _, _ = evaluate_spark_predictions_single_pass(m_sv.transform(sub_va_spark), TARGET_COL)

                row = {
                    "config_id": f"D{depth}_M{min_inst}_{imp[:3].upper()}",
                    "maxDepth": depth,
                    "minInstancesPerNode": min_inst,
                    "impurity": imp,
                    "num_nodes": int(m_sv.numNodes),
                    "effective_depth": int(m_sv.depth),
                    "fit_time_s": round(fit_time_s, 4),
                    "val_roc_auc": em_sv["roc_auc"],
                    "val_pr_auc": em_sv["pr_auc"],
                    "val_accuracy": em_sv["accuracy"],
                    "val_accuracy_pct": em_sv["accuracy_pct"],
                    "val_f1_weighted": em_sv["f1_weighted"],
                    "val_f1_positive": em_sv["f1_positive"],
                    "val_precision_positive": em_sv["precision_positive"],
                    "val_recall_positive": em_sv["recall_positive"],
                }
                grid_rows.append(row)
                if min_inst == 20 and imp == "gini":
                    print(
                        f"  [{config_idx:02d}/{total_configs}] depth={depth:2d}, minInst={min_inst:3d}, imp={imp:7s} "
                        f"-> Val ROC-AUC={em_sv['roc_auc']:.4f} | Val Acc={em_sv['accuracy_pct']:.2f}% | Nodes={m_sv.numNodes:4d}"
                    )

    sub_tr_spark.unpersist()
    sub_va_spark.unpersist()
    grid_df = pd.DataFrame(grid_rows)

    # Seleksi kandidat Champion secara otomatis-programatis dari konfigurasi ber-pre-pruning (minInstancesPerNode >= 20)
    pruned_grid = grid_df[grid_df["minInstancesPerNode"] >= 20]
    grid_sorted = pruned_grid.sort_values(
        by=["val_roc_auc", "val_pr_auc", "num_nodes"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_Val_row = grid_sorted.iloc[0].to_dict()
    champion_params = {
        "maxDepth": int(best_Val_row["maxDepth"]),
        "minInstancesPerNode": int(best_Val_row["minInstancesPerNode"]),
        "impurity": str(best_Val_row["impurity"]),
    }
    m1_row = grid_df[
        (grid_df["maxDepth"] == 7)
        & (grid_df["minInstancesPerNode"] == 20)
        & (grid_df["impurity"] == "gini")
    ].iloc[0].to_dict()
    m4_row = grid_df[
        (grid_df["maxDepth"] == 12)
        & (grid_df["minInstancesPerNode"] == 1)
        & (grid_df["impurity"] == "gini")
    ].iloc[0].to_dict()

    selection_note = (
        f"Dipilih secara otomatis-programatis dari peringkat #1 Validation Set: konfigurasi {best_Val_row['config_id']} "
        f"(maxDepth={champion_params['maxDepth']}, minInstancesPerNode={champion_params['minInstancesPerNode']}, "
        f"impurity='{champion_params['impurity']}') meraih Val ROC-AUC tertinggi sebesar {best_Val_row['val_roc_auc']:.4f} "
        f"(dan Val PR-AUC {best_Val_row['val_pr_auc']:.4f}), mengungguli M1 d=7 ({m1_row['val_roc_auc']:.4f}) sekaligus "
        f"mencegah overfitting parah pada pohon tanpa pruning M4 d=12 minInst=1 (Val ROC-AUC jatuh ke {m4_row['val_roc_auc']:.4f})."
    )

    print("\n" + "=" * 65)
    print(f"[CHAMPION HYPERPARAMETERS TERPILIH DARI VALIDATION SET]: {champion_params}")
    print(f"[CATATAN]: {selection_note}")
    print("=" * 65)

    # STEP 3: Latih ulang 7 Model Referensi Ablatif (termasuk Champion M2) di PySpark MLlib pada 100% Full Train Set (799.723 baris)
    # dan evaluasi pada 100% Test Set (200.277 baris)
    print("\n[STEP 3] Melatih 7 Model Referensi Ablatif (M0..M4 termasuk Champion M2) di PySpark MLlib pada Full Train Set (799.723 baris)...")
    train_spark_cached = full_train_df.select("features", TARGET_COL).cache()
    train_spark_cached.count()

    reference_configs = [
        {"id": "M0_Shallow_d3", "maxDepth": 3, "minInstancesPerNode": 20, "impurity": "gini"},
        {"id": "M1_Baseline_d7", "maxDepth": 7, "minInstancesPerNode": 20, "impurity": "gini"},
        {"id": "M2_Champion_d10_gini20", "maxDepth": 10, "minInstancesPerNode": 20, "impurity": "gini"},
        {"id": "M3a_d10_gini50", "maxDepth": 10, "minInstancesPerNode": 50, "impurity": "gini"},
        {"id": "M3b_d10_entropy20", "maxDepth": 10, "minInstancesPerNode": 20, "impurity": "entropy"},
        {"id": "M3c_d10_entropy50", "maxDepth": 10, "minInstancesPerNode": 50, "impurity": "entropy"},
        {"id": "M4_Deep_d12_gini1", "maxDepth": 12, "minInstancesPerNode": 1, "impurity": "gini"},
    ]
    if not any(
        rc["maxDepth"] == champion_params["maxDepth"]
        and rc["minInstancesPerNode"] == champion_params["minInstancesPerNode"]
        and rc["impurity"] == champion_params["impurity"]
        for rc in reference_configs
    ):
        reference_configs.append({"id": f"Champion_{best_Val_row['config_id']}", **champion_params})

    ref_results = []
    champion_model = None
    champion_train_time_s = 0.0
    champion_test_cm = None

    for rc in reference_configs:
        dt_spark = DecisionTreeClassifier(
            featuresCol="features",
            labelCol=TARGET_COL,
            maxDepth=rc["maxDepth"],
            minInstancesPerNode=rc["minInstancesPerNode"],
            impurity=rc["impurity"],
            seed=42,
        )
        t_ref0 = time.perf_counter()
        m_ref = dt_spark.fit(train_spark_cached)
        t_ref = time.perf_counter() - t_ref0

        p_ref = m_ref.transform(test_df)
        cm_ref, _, _, _ = evaluate_spark_predictions_single_pass(p_ref, TARGET_COL)

        # Ambil metrik Validation murni dari grid_df tanpa menimpa num_nodes validasi
        mask = (
            (grid_df["maxDepth"] == rc["maxDepth"])
            & (grid_df["minInstancesPerNode"] == rc["minInstancesPerNode"])
            & (grid_df["impurity"] == rc["impurity"])
        )
        m_row = grid_df[mask].iloc[0].to_dict()

        ref_entry = {
            "model_id": rc["id"],
            "maxDepth": rc["maxDepth"],
            "minInstancesPerNode": rc["minInstancesPerNode"],
            "impurity": rc["impurity"],
            "val_roc_auc": m_row["val_roc_auc"],
            "val_pr_auc": m_row["val_pr_auc"],
            "val_accuracy_pct": m_row["val_accuracy_pct"],
            "val_f1_weighted": m_row["val_f1_weighted"],
            "val_num_nodes": int(m_row["num_nodes"]),
            "test_roc_auc": cm_ref["roc_auc"],
            "test_pr_auc": cm_ref["pr_auc"],
            "test_accuracy": cm_ref["accuracy"],
            "test_accuracy_pct": cm_ref["accuracy_pct"],
            "test_f1_weighted": cm_ref["f1_weighted"],
            "test_precision_positive": cm_ref["precision_positive"],
            "test_recall_positive": cm_ref["recall_positive"],
            "tn": cm_ref["tn"],
            "fp": cm_ref["fp"],
            "fn": cm_ref["fn"],
            "tp": cm_ref["tp"],
            "num_nodes": int(m_ref.numNodes),
            "effective_depth": int(m_ref.depth),
            "full_train_time_s": round(t_ref, 3),
        }
        ref_results.append(ref_entry)
        print(
            f"  - {rc['id']:22s}: Test ROC-AUC={cm_ref['roc_auc']:.4f} | Acc={cm_ref['accuracy_pct']:.2f}% | "
            f"F1={cm_ref['f1_weighted']:.4f} | Nodes={m_ref.numNodes:4d} | Time={t_ref:.2f}s"
        )

        if (
            rc["maxDepth"] == champion_params["maxDepth"]
            and rc["minInstancesPerNode"] == champion_params["minInstancesPerNode"]
            and rc["impurity"] == champion_params["impurity"]
        ):
            champion_model = m_ref
            champion_train_time_s = t_ref
            champion_test_cm = cm_ref

    grid_csv_path = METRICS_DIR / "dt_tuning_grid.csv"
    grid_df.to_csv(grid_csv_path, index=False)
    print(f"\n[SAVED] Grid faktorial 72 konfigurasi disimpan di: {grid_csv_path}")

    print(
        f"\n[CHAMPION TEST RESULT] ROC-AUC={champion_test_cm['roc_auc']:.4f} | PR-AUC={champion_test_cm['pr_auc']:.4f} | "
        f"Acc={champion_test_cm['accuracy_pct']:.2f}% | F1={champion_test_cm['f1_weighted']:.4f} | "
        f"Nodes={champion_model.numNodes} | Time={champion_train_time_s:.2f}s"
    )
    print(
        f"[CONFUSION MATRIX] TN={champion_test_cm['tn']:,} | FP={champion_test_cm['fp']:,} | "
        f"FN={champion_test_cm['fn']:,} | TP={champion_test_cm['tp']:,}"
    )

    # Simpan Champion Model + Sanitasi Metadata Colab
    print(f"\n[STEP 4] Menyimpan Champion Model ke: {BEST_MODEL_DIR}")
    champion_model.write().overwrite().save(str(BEST_MODEL_DIR))
    cleaned_files = sanitize_spark_model_metadata(BEST_MODEL_DIR)
    print(f"[INFO] Sanitasi metadata Colab (`pruneTree`): {cleaned_files} file dibersihkan.")

    # Ekstraksi Feature Importance (Gini)
    importances = champion_model.featureImportances.toArray()
    fi_pairs = sorted(
        zip(ALL_FEATURE_NAMES, [float(x) for x in importances]),
        key=lambda x: x[1],
        reverse=True,
    )

    summary_payload = {
        "protocol": "75% Train-Sub / 25% Validation inside 80% Train Split; retrained on 100% of Train Split (799,723 rows); evaluated once on 20% untouched Test Split (200,277 rows)",
        "total_grid_configs": total_configs,
        "train_full_rows": int(n_full_train),
        "train_sub_rows": int(n_train_sub),
        "val_rows": int(n_val),
        "test_rows": int(n_test),
        "best_validation_row": best_Val_row,
        "champion_params": champion_params,
        "selection_note": selection_note,
        "champion_test_metrics": {
            "num_nodes": int(champion_model.numNodes),
            "effective_depth": int(champion_model.depth),
            "train_time_s": round(champion_train_time_s, 3),
            **champion_test_cm,
        },
        "ablation_reference_models": ref_results,
        "top_15_feature_importances": [
            {
                "rank": idx + 1,
                "feature": fname,
                "importance": round(score, 6),
                "percentage": round(score * 100.0, 2),
            }
            for idx, (fname, score) in enumerate(fi_pairs[:15])
        ],
    }
    save_json_metrics("dt_champion_summary.json", summary_payload)
    spark.stop()
    return summary_payload


if __name__ == "__main__":
    run_tuning()
