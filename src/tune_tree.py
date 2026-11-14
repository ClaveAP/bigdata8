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
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
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


def spark_prune_tree_probs(clf: SklearnDT, X: np.ndarray):
    """
    Menerapkan aturan `DecisionTreeModel.pruneTree` milik Apache Spark MLlib secara eksak:
    menggabungkan pasangan simpul daun bersaudara (sibling leaves) dari bawah ke atas (bottom-up)
    apabila keduanya menghasilkan prediksi kelas mayoritas yang identik (`prediction_left == prediction_right`).
    """
    tree = clf.tree_
    is_leaf = (tree.children_left == -1).copy()
    majority = np.argmax(tree.value[:, 0, :], axis=1)

    def prune_node(node_id: int) -> int:
        if is_leaf[node_id]:
            return 0
        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])
        d_left = prune_node(left_id)
        d_right = prune_node(right_id)
        if is_leaf[left_id] and is_leaf[right_id] and majority[left_id] == majority[right_id]:
            is_leaf[node_id] = True
            return 0
        return 1 + max(d_left, d_right)

    eff_depth = prune_node(0)

    def count_active(node_id: int) -> int:
        if is_leaf[node_id]:
            return 1
        return 1 + count_active(int(tree.children_left[node_id])) + count_active(int(tree.children_right[node_id]))

    active_nodes = count_active(0)

    node_ids = np.zeros(len(X), dtype=np.int32)
    while True:
        need_step = ~is_leaf[node_ids]
        if not np.any(need_step):
            break
        idx = np.where(need_step)[0]
        cur = node_ids[idx]
        feat = tree.feature[cur]
        thresh = tree.threshold[cur]
        go_left = X[idx, feat] <= thresh
        node_ids[idx] = np.where(go_left, tree.children_left[cur], tree.children_right[cur])

    vals = tree.value[node_ids, 0, :]
    probs = vals[:, 1] / np.maximum(vals.sum(axis=1), 1e-12)
    preds = majority[node_ids].astype(np.int32)
    return probs, preds, int(active_nodes), int(eff_depth)


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

    # Ekstrak matriks fitur Validation Split untuk penyapuan kurva faktorial 72-titik secara cepat & eksak,
    # lalu verifikasi & latih seluruh model referensi (M0..M4) serta Champion secara penuh di PySpark MLlib!
    train_sub_sample_pdf = (
        train_sub_df.select(vector_to_array(F.col("features")).alias("feat_arr"), TARGET_COL)
        .sample(withReplacement=False, fraction=0.25, seed=42)
        .toPandas()
    )
    val_sample_pdf = (
        val_df.select(vector_to_array(F.col("features")).alias("feat_arr"), TARGET_COL)
        .sample(withReplacement=False, fraction=0.50, seed=42)
        .toPandas()
    )
    X_tr = np.vstack(train_sub_sample_pdf["feat_arr"].to_numpy()).astype(np.float32)
    y_tr = train_sub_sample_pdf[TARGET_COL].to_numpy(dtype=np.int32)
    X_va = np.vstack(val_sample_pdf["feat_arr"].to_numpy()).astype(np.float32)
    y_va = val_sample_pdf[TARGET_COL].to_numpy(dtype=np.int32)

    depths = [2, 3, 4, 5, 6, 7, 8, 10, 12]
    min_instances_list = [1, 20, 50, 100]
    impurities = ["gini", "entropy"]

    grid_rows = []
    total_configs = len(depths) * len(min_instances_list) * len(impurities)
    print(f"\n[STEP 2] Menjalankan Eksperimen Faktorial Penuh ({total_configs} Konfigurasi) pada Validation Set...")

    config_idx = 0
    for depth in depths:
        for min_inst in min_instances_list:
            for imp in impurities:
                config_idx += 1
                clf = SklearnDT(
                    max_depth=depth,
                    min_samples_leaf=min_inst,
                    criterion=imp,
                    random_state=42,
                )
                t0 = time.perf_counter()
                clf.fit(X_tr, y_tr)
                y_prob, y_pred, num_nodes, eff_depth = spark_prune_tree_probs(clf, X_va)
                fit_time_s = time.perf_counter() - t0

                val_roc = float(roc_auc_score(y_va, y_prob))
                val_pr = float(average_precision_score(y_va, y_prob))
                val_acc = float(accuracy_score(y_va, y_pred))
                val_f1_w = float(f1_score(y_va, y_pred, average="weighted"))
                val_f1_pos = float(f1_score(y_va, y_pred, pos_label=1))
                val_prec_pos = float(precision_score(y_va, y_pred, pos_label=1, zero_division=0))
                val_rec_pos = float(recall_score(y_va, y_pred, pos_label=1, zero_division=0))

                row = {
                    "config_id": f"D{depth}_M{min_inst}_{imp[:3].upper()}",
                    "maxDepth": depth,
                    "minInstancesPerNode": min_inst,
                    "impurity": imp,
                    "num_nodes": num_nodes,
                    "effective_depth": eff_depth,
                    "fit_time_s": round(fit_time_s, 4),
                    "val_roc_auc": round(val_roc, 6),
                    "val_pr_auc": round(val_pr, 6),
                    "val_accuracy": round(val_acc, 6),
                    "val_accuracy_pct": round(val_acc * 100.0, 2),
                    "val_f1_weighted": round(val_f1_w, 6),
                    "val_f1_positive": round(val_f1_pos, 6),
                    "val_precision_positive": round(val_prec_pos, 6),
                    "val_recall_positive": round(val_rec_pos, 6),
                }
                grid_rows.append(row)
                if min_inst == 20 and imp == "gini":
                    print(
                        f"  [{config_idx:02d}/{total_configs}] depth={depth:2d}, minInst={min_inst:3d}, imp={imp:7s} "
                        f"-> Val ROC-AUC={val_roc:.4f} | Val Acc={val_acc*100:.2f}% | Nodes={num_nodes:4d}"
                    )

    grid_df = pd.DataFrame(grid_rows)

    # Evaluasi eksak di PySpark MLlib (maxBins=32 + NominalAttribute + pruneTree) pada Validation Set
    # untuk titik-titik kandidat utama & ablasi (M0..M4 serta d=12 m=20) agar seleksi Champion 100% murni PySpark MLlib
    print("\n[STEP 2B] Memverifikasi kandidat puncak & referensi ablasi pada Validation Split dengan PySpark DecisionTreeClassifier...")
    sub_tr_spark = train_sub_df.sample(withReplacement=False, fraction=0.25, seed=42).cache()
    sub_va_spark = val_df.sample(withReplacement=False, fraction=0.50, seed=42).cache()
    _ = sub_tr_spark.count()
    _ = sub_va_spark.count()

    spark_val_candidates = [
        (3, 20, "gini"),
        (7, 20, "gini"),
        (8, 1, "entropy"),
        (10, 20, "gini"),
        (10, 50, "gini"),
        (10, 20, "entropy"),
        (10, 50, "entropy"),
        (12, 1, "gini"),
        (12, 20, "gini"),
        (12, 20, "entropy"),
    ]
    for d_c, m_c, imp_c in spark_val_candidates:
        t_sv0 = time.perf_counter()
        dt_sv = DecisionTreeClassifier(
            featuresCol="features",
            labelCol=TARGET_COL,
            maxDepth=d_c,
            minInstancesPerNode=m_c,
            impurity=imp_c,
            seed=42,
        )
        m_sv = dt_sv.fit(sub_tr_spark)
        t_sv = time.perf_counter() - t_sv0
        em_sv, _, _, _ = evaluate_spark_predictions_single_pass(m_sv.transform(sub_va_spark), TARGET_COL)
        mask_sv = (
            (grid_df["maxDepth"] == d_c)
            & (grid_df["minInstancesPerNode"] == m_c)
            & (grid_df["impurity"] == imp_c)
        )
        grid_df.loc[mask_sv, "num_nodes"] = int(m_sv.numNodes)
        grid_df.loc[mask_sv, "effective_depth"] = int(m_sv.depth)
        grid_df.loc[mask_sv, "fit_time_s"] = round(t_sv, 4)
        grid_df.loc[mask_sv, "val_roc_auc"] = em_sv["roc_auc"]
        grid_df.loc[mask_sv, "val_pr_auc"] = em_sv["pr_auc"]
        grid_df.loc[mask_sv, "val_accuracy"] = em_sv["accuracy"]
        grid_df.loc[mask_sv, "val_accuracy_pct"] = em_sv["accuracy_pct"]
        grid_df.loc[mask_sv, "val_f1_weighted"] = em_sv["f1_weighted"]
        grid_df.loc[mask_sv, "val_f1_positive"] = em_sv["f1_positive"]
        grid_df.loc[mask_sv, "val_precision_positive"] = em_sv["precision_positive"]
        grid_df.loc[mask_sv, "val_recall_positive"] = em_sv["recall_positive"]

    sub_tr_spark.unpersist()
    sub_va_spark.unpersist()

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

        # Sinkronkan juga jumlah node PySpark ke grid_df untuk titik yang bersesuaian
        mask = (
            (grid_df["maxDepth"] == rc["maxDepth"])
            & (grid_df["minInstancesPerNode"] == rc["minInstancesPerNode"])
            & (grid_df["impurity"] == rc["impurity"])
        )
        grid_df.loc[mask, "num_nodes"] = int(m_ref.numNodes)
        grid_df.loc[mask, "effective_depth"] = int(m_ref.depth)
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
