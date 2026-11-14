"""
Modul Model Benchmarking, Komparasi Adil, Uji Statistik, & Skalabilitas Big Data (PySpark MLlib)
Mencakup seluruh eksperimen ilmiah proyek dan naskah jurnal JIKO (E0, E1, E3, E4, E5, E7, E9):
1. E1/E9: Tuning & komparasi adil 4 arsitektur model pada Validation Split, lalu evaluasi 1x di Test Set (200.277 baris):
   - Logistic Regression (menggunakan `features_lr`: One-Hot Encoded + StandardScaler)
   - Decision Tree Classifier (menggunakan `features`: Champion White-Box d=10 + varian d=3 & d=7)
   - Random Forest Classifier (menggunakan `features`: Ensemble Bagging)
   - Gradient-Boosted Trees / GBTClassifier (menggunakan `features`: Ensemble Boosting)
2. E5: Uji statistik berpasangan pada 200.277 data uji:
   - Uji McNemar (Chi-Square df=1 & exact binomial p-value)
   - Bootstrap (95% Confidence Interval untuk ROC-AUC, PR-AUC, F1, serta selisih Delta-AUC)
   - 5-Fold Cross-Validation pada Train Set (Mean +/- SD)
3. E3/E4: Benchmark Skalabilitas Komputasi PySpark:
   - Skalabilitas volume baris data (20%, 40%, 60%, 80%, 100% Train Set)
   - Skalabilitas core CPU (local[1], local[2], local[4] -> Speedup)
   - Efisiensi I/O Storage (CSV mentah 233,60 MB vs Apache Parquet Snappy 37,96 MB)
4. E7: Kontrol Negatif (`checkout_abandonments_per_month >= 5` -> membuktikan AUC ~0.50)
   & Analisis Sensitivitas Ambang Target (`cart_abandonment_rate` >= 35%, 40%, 45%).
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from pyspark.ml.classification import (
    DecisionTreeClassificationModel,
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from spark_session import get_spark_session
from pipeline_utils import (
    METRICS_DIR,
    MODELS_DIR,
    NEGATIVE_CONTROL_COL,
    PROCESSED_PARQUET_DIR,
    RAW_DATA_PATH,
    TARGET_COL,
    TARGET_SOURCE_COL,
    TEST_PARQUET_DIR,
    TRAIN_PARQUET_DIR,
    load_json_metrics,
    save_json_metrics,
)


def evaluate_spark_predictions_single_pass(preds_df, label_col: str = TARGET_COL):
    """
    Mengekstrak label aktual, probabilitas kelas positif, dan prediksi kelas dari DataFrame PySpark
    dalam SATU kali eksekusi job Spark, lalu menghitung metrik evaluasi presisi tinggi.
    """
    pdf = (
        preds_df.select(
            F.col(label_col).cast("int").alias("y_true"),
            vector_to_array(F.col("probability"))[1].cast("double").alias("y_prob"),
            F.col("prediction").cast("int").alias("y_pred"),
        )
        .toPandas()
    )
    y_true = pdf["y_true"].to_numpy()
    y_prob = pdf["y_prob"].to_numpy()
    y_pred = pdf["y_pred"].to_numpy()

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()]
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0.0
    f1_w = float(f1_score(y_true, y_pred, average="weighted"))
    f1_pos = float(f1_score(y_true, y_pred, pos_label=1))
    prec_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_pos = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_pos = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    metrics = {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "accuracy": round(acc, 6),
        "accuracy_pct": round(acc * 100.0, 2),
        "f1_weighted": round(f1_w, 6),
        "f1_positive": round(f1_pos, 6),
        "precision_weighted": round(prec_w, 6),
        "recall_weighted": round(rec_w, 6),
        "precision_positive": round(prec_pos, 6),
        "recall_positive": round(rec_pos, 6),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_test": total,
    }
    return metrics, y_true, y_prob, y_pred


def mcnemar_test_paired(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Menghitung Uji McNemar berpasangan (df=1) antara dua model pada test set yang sama."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & (~correct_b)))
    c = int(np.sum((~correct_a) & correct_b))
    n_disc = b + c
    if n_disc == 0:
        chi2_stat = 0.0
        chi2_cc = 0.0
        p_chi2 = 1.0
        p_exact = 1.0
    else:
        chi2_stat = float(((b - c) ** 2) / n_disc)
        chi2_cc = float(((abs(b - c) - 1.0) ** 2) / n_disc)
        p_chi2 = float(chi2.sf(chi2_cc, df=1))
        p_exact = float(binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided").pvalue)

    return {
        "a_correct_b_wrong_b": b,
        "a_wrong_b_correct_c": c,
        "discordant_total": n_disc,
        "chi2_uncorrected": round(chi2_stat, 4),
        "chi2_continuity_corrected": round(chi2_cc, 4),
        "df": 1,
        "p_value_chi2_cc": p_chi2,
        "p_value_exact_binomial": p_exact,
    }


def bootstrap_ci_and_deltas(
    y_true: np.ndarray,
    model_probs: dict,
    model_preds: dict,
    n_boot: int = 300,
    seed: int = 42,
):
    """Menghitung 95% Bootstrap Confidence Interval pada sampel uji terstratifikasi."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    sub_n = min(n, 40000)
    base_idx = rng.choice(n, size=sub_n, replace=False)
    y_sub = y_true[base_idx]
    probs_sub = {k: v[base_idx] for k, v in model_probs.items()}
    preds_sub = {k: v[base_idx] for k, v in model_preds.items()}

    auc_samples = {k: [] for k in model_probs}
    pr_samples = {k: [] for k in model_probs}
    f1_samples = {k: [] for k in model_probs}

    for _ in range(n_boot):
        idx = rng.integers(0, sub_n, size=sub_n)
        yb = y_sub[idx]
        for k in model_probs:
            auc_samples[k].append(roc_auc_score(yb, probs_sub[k][idx]))
            pr_samples[k].append(average_precision_score(yb, probs_sub[k][idx]))
            f1_samples[k].append(f1_score(yb, preds_sub[k][idx], average="weighted"))

    ci_per_model = {}
    for k in model_probs:
        auc_samples[k] = np.array(auc_samples[k])
        pr_samples[k] = np.array(pr_samples[k])
        f1_samples[k] = np.array(f1_samples[k])
        ci_per_model[k] = {
            "roc_auc_ci95_low": round(float(np.percentile(auc_samples[k], 2.5)), 6),
            "roc_auc_ci95_high": round(float(np.percentile(auc_samples[k], 97.5)), 6),
            "roc_auc_boot_sd": round(float(np.std(auc_samples[k], ddof=1)), 6),
            "pr_auc_ci95_low": round(float(np.percentile(pr_samples[k], 2.5)), 6),
            "pr_auc_ci95_high": round(float(np.percentile(pr_samples[k], 97.5)), 6),
            "f1_ci95_low": round(float(np.percentile(f1_samples[k], 2.5)), 6),
            "f1_ci95_high": round(float(np.percentile(f1_samples[k], 97.5)), 6),
        }

    pairs = [
        ("LR_Tuned", "DT_Champion_d10"),
        ("RF_Tuned", "DT_Champion_d10"),
        ("GBT_Tuned", "DT_Champion_d10"),
        ("LR_Tuned", "RF_Tuned"),
        ("DT_Champion_d10", "DT_Shallow_d3"),
    ]
    delta_results = []
    for m_a, m_b in pairs:
        if m_a in auc_samples and m_b in auc_samples:
            diff = auc_samples[m_a] - auc_samples[m_b]
            p_two_sided = float(2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
            delta_results.append(
                {
                    "model_a": m_a,
                    "model_b": m_b,
                    "delta_roc_auc_mean": round(float(np.mean(diff)), 6),
                    "delta_roc_auc_ci95_low": round(float(np.percentile(diff, 2.5)), 6),
                    "delta_roc_auc_ci95_high": round(float(np.percentile(diff, 97.5)), 6),
                    "bootstrap_p_value": round(p_two_sided, 6),
                }
            )

    return ci_per_model, delta_results


def build_downsampled_curves(y_true: np.ndarray, model_probs: dict, max_points: int = 200):
    """Menyimpan koordinat kurva ROC dan Precision-Recall untuk visualisasi multi-model."""
    curves = {}
    for key, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        prec, rec, _ = precision_recall_curve(y_true, probs)

        idx_roc = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr)), dtype=int)
        idx_pr = np.linspace(0, len(prec) - 1, min(max_points, len(prec)), dtype=int)

        curves[key] = {
            "roc": {
                "fpr": [round(float(x), 5) for x in fpr[idx_roc]],
                "tpr": [round(float(x), 5) for x in tpr[idx_roc]],
            },
            "pr": {
                "recall": [round(float(x), 5) for x in rec[idx_pr]],
                "precision": [round(float(x), 5) for x in prec[idx_pr]],
            },
        }
    return curves


def run_model_comparison():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark_session(app_name="EcommerceModelComparison")

    print("\n[INFO] Membaca data Train & Test dari Parquet...")
    train_df = spark.read.parquet(str(TRAIN_PARQUET_DIR)).cache()
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR)).cache()
    n_train = train_df.count()
    n_test = test_df.count()
    print(f"[INFO] Train: {n_train:,} baris | Test: {n_test:,} baris")

    # Gunakan Validation Sample (20% dari Train Set) untuk pemilihan cepat hiperparameter LR & RF
    train_tune_df, val_tune_df = (
        train_df.sample(withReplacement=False, fraction=0.25, seed=42).randomSplit([0.75, 0.25], seed=42)
    )
    train_tune_df = train_tune_df.cache()
    val_tune_df = val_tune_df.cache()
    _ = train_tune_df.count()
    _ = val_tune_df.count()

    # 1. Tuning Logistic Regression pada `features_lr` (One-Hot Encoded + Standardized)
    print("\n[E1-A] Tuning Logistic Regression (OHE + StandardScaler) pada Validation Set...")
    lr_grid = []
    best_lr_reg = 0.0
    best_lr_val_auc = -1.0
    for reg in [0.0, 0.01, 0.1]:
        lr_cand = LogisticRegression(
            featuresCol="features_lr", labelCol=TARGET_COL, regParam=reg, maxIter=50
        )
        t0 = time.perf_counter()
        m_lr = lr_cand.fit(train_tune_df)
        t_fit = time.perf_counter() - t0
        val_m, _, _, _ = evaluate_spark_predictions_single_pass(m_lr.transform(val_tune_df), TARGET_COL)
        lr_grid.append({"regParam": reg, "fit_time_s": round(t_fit, 3), **val_m})
        print(f"  LR regParam={reg:.2f} -> Val ROC-AUC={val_m['roc_auc']:.4f} | Val F1={val_m['f1_weighted']:.4f}")
        if val_m["roc_auc"] > best_lr_val_auc:
            best_lr_val_auc = val_m["roc_auc"]
            best_lr_reg = reg

    # 2. Tuning Random Forest pada `features`
    print("\n[E1-B] Tuning Random Forest pada Validation Set...")
    rf_grid = []
    best_rf_params = {"numTrees": 40, "maxDepth": 10}
    best_rf_val_auc = -1.0
    for n_trees in [20, 40]:
        for d in [7, 10]:
            rf_cand = RandomForestClassifier(
                featuresCol="features",
                labelCol=TARGET_COL,
                numTrees=n_trees,
                maxDepth=d,
                minInstancesPerNode=20,
                seed=42,
            )
            t0 = time.perf_counter()
            m_rf = rf_cand.fit(train_tune_df)
            t_fit = time.perf_counter() - t0
            val_m, _, _, _ = evaluate_spark_predictions_single_pass(m_rf.transform(val_tune_df), TARGET_COL)
            rf_grid.append({"numTrees": n_trees, "maxDepth": d, "fit_time_s": round(t_fit, 3), **val_m})
            print(
                f"  RF numTrees={n_trees}, maxDepth={d} -> Val ROC-AUC={val_m['roc_auc']:.4f} | Val F1={val_m['f1_weighted']:.4f} ({t_fit:.2f}s)"
            )
            if val_m["roc_auc"] > best_rf_val_auc:
                best_rf_val_auc = val_m["roc_auc"]
                best_rf_params = {"numTrees": n_trees, "maxDepth": d}

    train_tune_df.unpersist()
    val_tune_df.unpersist()

    try:
        dt_champ_info = load_json_metrics("dt_champion_summary.json")
        dt_champ_params = dt_champ_info["champion_params"]
    except Exception:
        dt_champ_params = {"maxDepth": 10, "minInstancesPerNode": 20, "impurity": "gini"}

    models_spec = [
        {
            "key": "LR_Tuned",
            "display_name": "Logistic Regression (OHE+Scaled)",
            "architecture": "Linear (One-Hot + Standardized)",
            "interpretability": "Moderat (Bobot log-odds aditif global, tanpa aturan ambang bertingkat)",
            "estimator": LogisticRegression(
                featuresCol="features_lr", labelCol=TARGET_COL, regParam=best_lr_reg, maxIter=50
            ),
            "params_desc": f"regParam={best_lr_reg}, maxIter=50, OHE+StandardScaler (132 dims)",
        },
        {
            "key": "DT_Shallow_d3",
            "display_name": "Decision Tree (Shallow d=3)",
            "architecture": "Tree-based Ultra-Compact (White-Box)",
            "interpretability": "Sangat Tinggi (Hanya 7 node / 4 aturan daun, dapat dihafal langsung)",
            "estimator": DecisionTreeClassifier(
                featuresCol="features", labelCol=TARGET_COL, maxDepth=3, minInstancesPerNode=20, impurity="gini", seed=42
            ),
            "params_desc": "maxDepth=3, minInstancesPerNode=20, impurity=gini",
        },
        {
            "key": "DT_Default_d7",
            "display_name": "Decision Tree (Baseline d=7)",
            "architecture": "Tree-based Medium (White-Box)",
            "interpretability": "Tinggi (Pohon menengah)",
            "estimator": DecisionTreeClassifier(
                featuresCol="features", labelCol=TARGET_COL, maxDepth=7, minInstancesPerNode=20, impurity="gini", seed=42
            ),
            "params_desc": "maxDepth=7, minInstancesPerNode=20, impurity=gini",
        },
        {
            "key": "DT_Champion_d10",
            "display_name": "Decision Tree (Champion d=10)",
            "architecture": "Tree-based Tuned & Pruned (White-Box)",
            "interpretability": "Sangat Tinggi (Eksplisit IF-THEN Rules & threshold fitur)",
            "estimator": DecisionTreeClassifier(
                featuresCol="features",
                labelCol=TARGET_COL,
                maxDepth=dt_champ_params["maxDepth"],
                minInstancesPerNode=dt_champ_params["minInstancesPerNode"],
                impurity=dt_champ_params["impurity"],
                seed=42,
            ),
            "params_desc": f"maxDepth={dt_champ_params['maxDepth']}, minInstancesPerNode={dt_champ_params['minInstancesPerNode']}, impurity={dt_champ_params['impurity']}",
        },
        {
            "key": "RF_Tuned",
            "display_name": "Random Forest (Tuned Ensemble)",
            "architecture": "Ensemble Bagging (Black-Box)",
            "interpretability": "Rendah (Gabungan puluhan pohon acak, tanpa aturan tunggal)",
            "estimator": RandomForestClassifier(
                featuresCol="features",
                labelCol=TARGET_COL,
                numTrees=best_rf_params["numTrees"],
                maxDepth=best_rf_params["maxDepth"],
                minInstancesPerNode=20,
                seed=42,
            ),
            "params_desc": f"numTrees={best_rf_params['numTrees']}, maxDepth={best_rf_params['maxDepth']}, minInstancesPerNode=20",
        },
        {
            "key": "GBT_Tuned",
            "display_name": "Gradient-Boosted Trees (GBT)",
            "architecture": "Ensemble Boosting (Black-Box)",
            "interpretability": "Rendah (Sekuens pohon residu aditif)",
            "estimator": GBTClassifier(
                featuresCol="features", labelCol=TARGET_COL, maxIter=20, maxDepth=5, seed=42
            ),
            "params_desc": "maxIter=20, maxDepth=5",
        },
    ]

    print("\n" + "=" * 75)
    print("[E1] MELATIH & MENGEVALUASI 6 MODEL PADA FULL TRAIN (799.723) -> TEST (200.277)")
    print("=" * 75)

    comparison_rows = []
    models_dict = {}
    y_true_ref = None
    model_probs = {}
    model_preds = {}

    for spec in models_spec:
        m_key = spec["key"]
        m_name = spec["display_name"]
        est = spec["estimator"]
        print(f"\n[TRAINING] {m_name} pada Full Train Set (799.723 baris)...")
        t0 = time.perf_counter()
        if m_key == "DT_Champion_d10" and (MODELS_DIR / "decision_tree_tuned").exists():
            fitted = DecisionTreeClassificationModel.load(str(MODELS_DIR / "decision_tree_tuned"))
            t_fit = float(dt_champ_info["champion_test_metrics"]["train_time_s"])
        else:
            fitted = est.fit(train_df)
            t_fit = time.perf_counter() - t0

        preds_df = fitted.transform(test_df)
        eval_m, y_t, y_prob, y_pred = evaluate_spark_predictions_single_pass(preds_df, TARGET_COL)
        if y_true_ref is None:
            y_true_ref = y_t
        model_probs[m_key] = y_prob
        model_preds[m_key] = y_pred

        num_nodes = int(getattr(fitted, "numNodes", getattr(fitted, "totalNumNodes", 0)))
        complexity_desc = (
            f"{num_nodes} Tree Nodes" if num_nodes > 0 else f"{int(fitted.numFeatures)} Linear Weights"
        )
        row = {
            "model_key": m_key,
            "model_name": m_name,
            "architecture": spec["architecture"],
            "params": spec["params_desc"],
            "interpretability": spec["interpretability"],
            "complexity_desc": complexity_desc,
            "num_nodes_or_weights": num_nodes if num_nodes > 0 else int(fitted.numFeatures),
            "train_time_s": round(t_fit, 3),
            **eval_m,
        }
        comparison_rows.append(row)
        models_dict[m_key] = {
            "display_name": m_name,
            "architecture": spec["architecture"],
            "params": spec["params_desc"],
            "interpretability": spec["interpretability"],
            "complexity_desc": complexity_desc,
            "train_time_s": round(t_fit, 3),
            "test_metrics": eval_m,
        }
        print(
            f"  -> Test ROC-AUC={eval_m['roc_auc']:.4f} | PR-AUC={eval_m['pr_auc']:.4f} | "
            f"Acc={eval_m['accuracy_pct']:.2f}% | F1={eval_m['f1_weighted']:.4f} | "
            f"{complexity_desc} | Time={t_fit:.2f}s"
        )

    # 3. 5-Fold Cross-Validation pada sampel terstratifikasi Train Set (160.000 baris) untuk ke-6 model
    print("\n[E5-A] Menjalankan 5-Fold Cross-Validation untuk ke-6 model benchmark...")
    cv_sample_df = (
        train_df.sample(withReplacement=False, fraction=0.20, seed=42)
        .withColumn("fold_id", F.pmod(F.xxhash64(F.col("user_id")), F.lit(5)))
        .cache()
    )
    _ = cv_sample_df.count()
    cv_summary = {}
    cv_specs = models_spec
    for spec in cv_specs:
        m_key = spec["key"]
        m_name = spec["display_name"]
        est = spec["estimator"]
        fold_aucs, fold_f1s, fold_accs = [], [], []
        for k in range(5):
            tr_k = cv_sample_df.filter(F.col("fold_id") != k)
            va_k = cv_sample_df.filter(F.col("fold_id") == k)
            m_k = est.fit(tr_k)
            em_k, _, _, _ = evaluate_spark_predictions_single_pass(m_k.transform(va_k), TARGET_COL)
            fold_aucs.append(em_k["roc_auc"])
            fold_f1s.append(em_k["f1_weighted"])
            fold_accs.append(em_k["accuracy"])
        cv_summary[m_key] = {
            "cv5_roc_auc_mean": round(float(np.mean(fold_aucs)), 6),
            "cv5_roc_auc_sd": round(float(np.std(fold_aucs, ddof=1)), 6),
            "cv5_f1_mean": round(float(np.mean(fold_f1s)), 6),
            "cv5_f1_sd": round(float(np.std(fold_f1s, ddof=1)), 6),
            "cv5_acc_mean": round(float(np.mean(fold_accs)), 6),
            "cv5_acc_sd": round(float(np.std(fold_accs, ddof=1)), 6),
            "fold_roc_aucs": fold_aucs,
        }
        models_dict[m_key]["cv5_metrics"] = cv_summary[m_key]
        print(
            f"  [CV-5] {m_name}: ROC-AUC = {cv_summary[m_key]['cv5_roc_auc_mean']:.4f} +/- {cv_summary[m_key]['cv5_roc_auc_sd']:.4f} | "
            f"F1 = {cv_summary[m_key]['cv5_f1_mean']:.4f} +/- {cv_summary[m_key]['cv5_f1_sd']:.4f}"
        )
    cv_sample_df.unpersist()

    # 4. Uji Statistik Berpasangan (McNemar & Bootstrap 95% CI) pada 200.277 data uji
    print("\n[E5-B] Menghitung Uji McNemar Berpasangan & Bootstrap 95% CI...")
    ci_per_model, bootstrap_deltas = bootstrap_ci_and_deltas(y_true_ref, model_probs, model_preds, n_boot=300)

    for row in comparison_rows:
        m_key = row["model_key"]
        if m_key in ci_per_model:
            row.update(ci_per_model[m_key])
            models_dict[m_key]["bootstrap_ci95"] = ci_per_model[m_key]
        if m_key in cv_summary:
            row.update(cv_summary[m_key])

    mcnemar_pairs = [
        ("LR_Tuned", "DT_Champion_d10"),
        ("RF_Tuned", "DT_Champion_d10"),
        ("GBT_Tuned", "DT_Champion_d10"),
        ("LR_Tuned", "RF_Tuned"),
        ("DT_Champion_d10", "DT_Shallow_d3"),
    ]
    mcnemar_results = []
    for m_a, m_b in mcnemar_pairs:
        res = mcnemar_test_paired(y_true_ref, model_preds[m_a], model_preds[m_b])
        mcnemar_results.append(
            {
                "model_a": m_a,
                "model_a_name": models_dict[m_a]["display_name"],
                "model_b": m_b,
                "model_b_name": models_dict[m_b]["display_name"],
                **res,
            }
        )
        print(
            f"  [McNemar] {m_a} vs {m_b}: b={res['a_correct_b_wrong_b']}, c={res['a_wrong_b_correct_c']}, "
            f"chi2_cc={res['chi2_continuity_corrected']:.4f}, p_exact={res['p_value_exact_binomial']:.3e}"
        )

    # Simpan koordinat kurva ROC & PR
    curves_payload = build_downsampled_curves(y_true_ref, model_probs)
    save_json_metrics("roc_pr_curves.json", curves_payload)

    # 5. E7: Kontrol Negatif (`checkout_abandonments_per_month >= 5`) & Sensitivitas Ambang (35%, 40%, 45%)
    print("\n[E7] Menjalankan Eksperimen Kontrol Negatif & Sensitivitas Ambang Target...")
    sensitivity_rows = []
    exp_train_sub = train_df.sample(withReplacement=False, fraction=0.25, seed=42).cache()
    exp_test_sub = test_df.sample(withReplacement=False, fraction=0.50, seed=42).cache()
    n_exp_train = exp_train_sub.count()
    _ = exp_test_sub.count()

    # 5a. Kontrol Negatif
    tr_nc = exp_train_sub.withColumn("label_neg_ctrl", F.when(F.col(NEGATIVE_CONTROL_COL) >= 5, 1).otherwise(0))
    te_nc = exp_test_sub.withColumn("label_neg_ctrl", F.when(F.col(NEGATIVE_CONTROL_COL) >= 5, 1).otherwise(0))
    for m_key, m_label, est_nc in [
        (
            "LR_Tuned",
            "Logistic Regression (OHE+Scaled)",
            LogisticRegression(featuresCol="features_lr", labelCol="label_neg_ctrl", regParam=best_lr_reg, maxIter=30),
        ),
        (
            "DT_Champion_d10",
            "Decision Tree (Champion d=10)",
            DecisionTreeClassifier(featuresCol="features", labelCol="label_neg_ctrl", maxDepth=10, minInstancesPerNode=20, seed=42),
        ),
        (
            "RF_Tuned",
            "Random Forest (Tuned)",
            RandomForestClassifier(featuresCol="features", labelCol="label_neg_ctrl", numTrees=20, maxDepth=7, seed=42),
        ),
    ]:
        m_nc = est_nc.fit(tr_nc)
        em_nc, _, _, _ = evaluate_spark_predictions_single_pass(m_nc.transform(te_nc), "label_neg_ctrl")
        sensitivity_rows.append(
            {
                "experiment": "Negative_Control_Checkout_Abandonments_ge_5",
                "target_definition": f"{NEGATIVE_CONTROL_COL} >= 5",
                "model_key": m_key,
                "model_name": m_label,
                "roc_auc": em_nc["roc_auc"],
                "pr_auc": em_nc["pr_auc"],
                "accuracy_pct": em_nc["accuracy_pct"],
                "f1_weighted": em_nc["f1_weighted"],
            }
        )
        print(f"  [Negative Control] {m_label}: ROC-AUC={em_nc['roc_auc']:.4f} | Acc={em_nc['accuracy_pct']:.2f}%")

    # 5b. Sensitivitas Ambang (35%, 40%, 45%)
    for thresh in [35.0, 40.0, 45.0]:
        lbl_col = f"label_t{int(thresh)}"
        tr_t = exp_train_sub.withColumn(lbl_col, F.when(F.col(TARGET_SOURCE_COL) >= thresh, 1).otherwise(0))
        te_t = exp_test_sub.withColumn(lbl_col, F.when(F.col(TARGET_SOURCE_COL) >= thresh, 1).otherwise(0))
        pos_rate = tr_t.filter(F.col(lbl_col) == 1).count() / n_exp_train * 100.0
        for m_key, m_label, est_t in [
            (
                "LR_Tuned",
                "Logistic Regression (OHE+Scaled)",
                LogisticRegression(featuresCol="features_lr", labelCol=lbl_col, regParam=best_lr_reg, maxIter=40),
            ),
            (
                "DT_Champion_d10",
                "Decision Tree (Champion d=10)",
                DecisionTreeClassifier(featuresCol="features", labelCol=lbl_col, maxDepth=10, minInstancesPerNode=20, seed=42),
            ),
            (
                "RF_Tuned",
                "Random Forest (Tuned)",
                RandomForestClassifier(featuresCol="features", labelCol=lbl_col, numTrees=30, maxDepth=10, minInstancesPerNode=20, seed=42),
            ),
        ]:
            m_t = est_t.fit(tr_t)
            em_t, _, _, _ = evaluate_spark_predictions_single_pass(m_t.transform(te_t), lbl_col)
            sensitivity_rows.append(
                {
                    "experiment": f"Threshold_{int(thresh)}pct",
                    "target_definition": f"{TARGET_SOURCE_COL} >= {thresh:.1f}% (Pos={pos_rate:.2f}%)",
                    "model_key": m_key,
                    "model_name": m_label,
                    "roc_auc": em_t["roc_auc"],
                    "pr_auc": em_t["pr_auc"],
                    "accuracy_pct": em_t["accuracy_pct"],
                    "f1_weighted": em_t["f1_weighted"],
                }
            )
            print(
                f"  [Sensitivity >= {thresh:.0f}%] {m_label}: ROC-AUC={em_t['roc_auc']:.4f} | Acc={em_t['accuracy_pct']:.2f}% | F1={em_t['f1_weighted']:.4f}"
            )
    exp_train_sub.unpersist()
    exp_test_sub.unpersist()

    # 6. E3: Benchmark Skalabilitas Volume Baris Data (20%, 40%, 60%, 80%, 100% Train Set)
    print("\n[E3] Menjalankan Benchmark Skalabilitas Volume Baris Data (20% s.d. 100% Train Set)...")
    row_scaling_pivoted = []
    row_scaling_long = []
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    for frac in fractions:
        if frac < 1.0:
            sub_df = train_df.sample(withReplacement=False, fraction=frac, seed=42).cache()
            n_rows_sub = sub_df.count()
            t0 = time.perf_counter()
            _ = DecisionTreeClassifier(featuresCol="features", labelCol=TARGET_COL, maxDepth=10, minInstancesPerNode=20, seed=42).fit(sub_df)
            dt_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = LogisticRegression(featuresCol="features_lr", labelCol=TARGET_COL, regParam=best_lr_reg, maxIter=50).fit(sub_df)
            lr_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = RandomForestClassifier(featuresCol="features", labelCol=TARGET_COL, numTrees=best_rf_params["numTrees"], maxDepth=10, minInstancesPerNode=20, seed=42).fit(sub_df)
            rf_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = GBTClassifier(featuresCol="features", labelCol=TARGET_COL, maxIter=20, maxDepth=5, seed=42).fit(sub_df)
            gbt_s = time.perf_counter() - t0
            sub_df.unpersist()
        else:
            n_rows_sub = n_train
            dt_s = models_dict["DT_Champion_d10"]["train_time_s"]
            lr_s = models_dict["LR_Tuned"]["train_time_s"]
            rf_s = models_dict["RF_Tuned"]["train_time_s"]
            gbt_s = models_dict["GBT_Tuned"]["train_time_s"]

        piv_row = {
            "fraction": frac,
            "train_rows": int(n_rows_sub),
            "dt_median_s": round(float(dt_s), 3),
            "lr_median_s": round(float(lr_s), 3),
            "rf_median_s": round(float(rf_s), 3),
            "gbt_median_s": round(float(gbt_s), 3),
        }
        row_scaling_pivoted.append(piv_row)
        for m_k, val_s in [("DT_Champion_d10", dt_s), ("LR_Tuned", lr_s), ("RF_Tuned", rf_s), ("GBT_Tuned", gbt_s)]:
            row_scaling_long.append(
                {"fraction": frac, "train_rows": int(n_rows_sub), "model_key": m_k, "fit_time_s": round(float(val_s), 3)}
            )
        print(
            f"  [Row Scaling {int(frac*100):3d}% ({n_rows_sub:,} rows)] DT={dt_s:.2f}s | LR={lr_s:.2f}s | RF={rf_s:.2f}s | GBT={gbt_s:.2f}s"
        )

    # 7. Benchmark I/O Pembacaan & Agregasi CSV (233.60 MB) vs Parquet Snappy (37.96 MB)
    print("\n[E3-IO] Mengukur Latensi Baca + Agregasi Penuh: CSV Mentah vs Apache Parquet Snappy...")
    io_times_dict = {}
    for fmt_key, reader_fn in [
        ("csv", lambda: spark.read.csv(str(RAW_DATA_PATH), header=True, inferSchema=True)),
        ("parquet", lambda: spark.read.parquet(str(PROCESSED_PARQUET_DIR))),
    ]:
        _ = reader_fn().groupBy("device_type").agg(F.mean(TARGET_SOURCE_COL)).collect()
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = (
                reader_fn()
                .groupBy("device_type", "country")
                .agg(F.count("user_id"), F.mean(TARGET_SOURCE_COL), F.mean("impulse_buying_score"))
                .collect()
            )
            runs.append(time.perf_counter() - t0)
        io_times_dict[fmt_key] = float(np.median(runs))

    prep_summary = load_json_metrics("preprocessing_summary.json")
    csv_mb = float(prep_summary["raw_csv_size_mb"])
    pq_mb = float(prep_summary["parquet_size_mb"])
    red_pct = float(prep_summary.get("storage_reduction_pct", prep_summary.get("storage_saving_pct", 83.75)))
    storage_io_summary = {
        "csv_size_mb": csv_mb,
        "parquet_size_mb": pq_mb,
        "size_reduction_pct": red_pct,
        "csv_read_agg_median_s": round(io_times_dict["csv"], 3),
        "parquet_read_agg_median_s": round(io_times_dict["parquet"], 3),
        "io_speedup_factor": round(io_times_dict["csv"] / max(io_times_dict["parquet"], 1e-6), 2),
    }
    print(
        f"  [I/O Benchmark] CSV ({csv_mb:.2f} MB): {storage_io_summary['csv_read_agg_median_s']:.3f}s vs "
        f"Parquet ({pq_mb:.2f} MB): {storage_io_summary['parquet_read_agg_median_s']:.3f}s "
        f"-> Speedup {storage_io_summary['io_speedup_factor']:.2f}x"
    )

    spark.stop()

    # 8. E4: Benchmark Skalabilitas Core CPU (local[1], local[2], local[4]) untuk Decision Tree Champion
    print("\n[E4] Menjalankan Benchmark Skalabilitas Core CPU (local[1], local[2], local[4])...")
    core_scaling_rows = []
    t1_median = None
    for n_cores in [1, 2, 4]:
        sp_core = (
            SparkSession.builder.appName(f"CoreScaling_{n_cores}")
            .master(f"local[{n_cores}]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.shuffle.partitions", str(max(4, n_cores * 2)))
            .getOrCreate()
        )
        sp_core.sparkContext.setLogLevel("WARN")
        tr_c = sp_core.read.parquet(str(TRAIN_PARQUET_DIR)).select("features", TARGET_COL).cache()
        _ = tr_c.count()
        dt_c = DecisionTreeClassifier(
            featuresCol="features",
            labelCol=TARGET_COL,
            maxDepth=dt_champ_params["maxDepth"],
            minInstancesPerNode=dt_champ_params["minInstancesPerNode"],
            impurity=dt_champ_params["impurity"],
            seed=42,
        )
        # Warm-up 1x + 2 timed runs
        _ = dt_c.fit(tr_c)
        c_times = []
        for _ in range(2):
            t0 = time.perf_counter()
            _ = dt_c.fit(tr_c)
            c_times.append(time.perf_counter() - t0)
        med_s = float(np.median(c_times))
        if n_cores == 1:
            t1_median = med_s
        speedup = round(t1_median / med_s, 3) if t1_median and med_s > 0 else 1.0
        core_scaling_rows.append(
            {
                "cores": n_cores,
                "master": f"local[{n_cores}]",
                "dt_fit_median_s": round(med_s, 3),
                "runs_s": [round(float(x), 3) for x in c_times],
                "speedup_vs_1core": speedup,
            }
        )
        print(f"  [Core Scaling local[{n_cores}]] Median={med_s:.2f}s | Speedup={speedup:.2f}x")
        sp_core.stop()

    # Simpan seluruh tabel CSV & JSON ke reports/metrics/
    pd.DataFrame(comparison_rows).to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(mcnemar_results).to_csv(METRICS_DIR / "mcnemar_tests.csv", index=False)
    pd.DataFrame(bootstrap_deltas).to_csv(METRICS_DIR / "bootstrap_auc_deltas.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(METRICS_DIR / "sensitivity_and_negative_control.csv", index=False)
    pd.DataFrame(row_scaling_pivoted).to_csv(METRICS_DIR / "scaling_rows.csv", index=False)
    pd.DataFrame(core_scaling_rows).to_csv(METRICS_DIR / "scaling_cores.csv", index=False)

    master_payload = {
        "hardware": {
            "cpu": "Intel(R) Core(TM) i5-7267U CPU @ 3.10GHz (2 Physical Cores / 4 Logical Threads)",
            "ram_gb": 16,
            "os": "macOS",
            "spark_version": "4.3.0",
        },
        "dataset_split": {"train_rows": int(n_train), "test_rows": int(n_test)},
        "lr_validation_grid": lr_grid,
        "rf_validation_grid": rf_grid,
        "models": models_dict,
        "models_comparison": comparison_rows,
        "mcnemar_paired_tests": mcnemar_results,
        "bootstrap_paired_auc_deltas": bootstrap_deltas,
        "sensitivity_and_negative_control": sensitivity_rows,
        "scaling_benchmarks": {
            "row_scaling": row_scaling_pivoted,
            "row_scaling_long": row_scaling_long,
            "core_scaling": core_scaling_rows,
            "storage_io_benchmark": storage_io_summary,
        },
    }
    out_path = save_json_metrics("model_comparison_summary.json", master_payload)
    print(f"\n[SAVED] Ringkasan lengkap komparasi model & uji statistik disimpan di: {out_path}")
    return master_payload


if __name__ == "__main__":
    run_model_comparison()
