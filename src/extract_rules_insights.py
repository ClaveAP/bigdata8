"""
Modul Ekstraksi Feature Importance, IF-THEN Rules, & Evaluasi Fidelity Pohon Keputusan (E6)
Memuat Champion Model Decision Tree dari disk, memetakan seluruh 57 fitur ke nama kolom asli,
menghasilkan plot Top-12 Feature Importance, mengekstrak aturan bisnis operasional dengan nama
kolom manusiawi, serta mengukur tingkat kesepakatan (Fidelity) pohon dangkal (maxDepth=3)
terhadap Champion Decision Tree (maxDepth=10) pada 200.277 data uji.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.classification import (
    DecisionTreeClassificationModel,
    DecisionTreeClassifier,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from spark_session import get_spark_session
from compare_models import evaluate_spark_predictions_single_pass
from pipeline_utils import (
    ALL_FEATURE_NAMES,
    LOOKER_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TEST_PARQUET_DIR,
    TRAIN_PARQUET_DIR,
    humanize_tree_debug_string,
    save_json_metrics,
)

BEST_MODEL_DIR = MODELS_DIR / "decision_tree_tuned"


def extract_top_split_thresholds(debug_str: str, feature_names: list[str]) -> list[dict]:
    """Mengekstrak split cabang teratas (indentasi <= 6 spasi) dari toDebugString secara terprogram."""
    splits = []
    pattern = re.compile(r"^(\s*)If \(feature (\d+) (<=|>|in) ([^)]+)\)")
    for line in debug_str.splitlines():
        m = pattern.match(line)
        if m:
            indent_len = len(m.group(1))
            level = (indent_len // 2)
            if level <= 3:
                f_idx = int(m.group(2))
                f_name = feature_names[f_idx] if 0 <= f_idx < len(feature_names) else f"feature_{f_idx}"
                op = m.group(3)
                val_str = m.group(4).strip()
                splits.append(
                    {
                        "tree_level": level,
                        "feature_index": f_idx,
                        "feature_name": f_name,
                        "operator": op,
                        "threshold": val_str,
                        "condition": f"{f_name} {op} {val_str}",
                    }
                )
    return splits


def run_extraction():
    spark = get_spark_session(app_name="EcommerceExtractRulesAndFidelity")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOOKER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Memuat Champion Model dari: {BEST_MODEL_DIR}")
    champion_model = DecisionTreeClassificationModel.load(str(BEST_MODEL_DIR))

    # 1. Feature Importance Mapping (57 Fitur)
    importances = champion_model.featureImportances.toArray()
    fi_df = (
        pd.DataFrame(
            {
                "feature_name": [f.replace("_encoded", "") for f in ALL_FEATURE_NAMES],
                "feature_internal": ALL_FEATURE_NAMES,
                "importance_score": [float(x) for x in importances],
                "percentage_contribution": [round(float(x) * 100.0, 2) for x in importances],
            }
        )
        .sort_values(by="importance_score", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 65)
    print("TOP 10 FITUR PALING BERPENGARUH (GINI IMPORTANCE — CHAMPION DT)")
    print("=" * 65)
    print(fi_df[["feature_name", "importance_score", "percentage_contribution"]].head(10).to_string(index=False))
    print("=" * 65)

    # Simpan CSV untuk Looker Studio
    looker_fi_path = LOOKER_DIR / "looker_model_feature_importance.csv"
    fi_df[["feature_name", "importance_score", "percentage_contribution"]].head(10).to_csv(
        looker_fi_path,
        index=False,
        float_format="%.6f",
    )
    print(f"[SAVED] Looker Feature Importance CSV: {looker_fi_path}")

    # 2. Visualisasi Feature Importance (Decision Tree)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    top_fi = fi_df.head(12)
    ax = sns.barplot(
        data=top_fi,
        x="importance_score",
        y="feature_name",
        hue="feature_name",
        palette="viridis",
        legend=False,
    )
    plt.title(
        "Top 12 Faktor Pendorong Pembatalan Keranjang (Decision Tree Gini Importance)",
        fontsize=12.5,
        fontweight="bold",
    )
    plt.xlabel("Skor Kepentingan Fitur (Gini Importance Score)", fontsize=11)
    plt.ylabel("Fitur Perilaku / Psikologis / Demografi", fontsize=11)
    plt.xlim(0, max(top_fi["importance_score"]) * 1.15)
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(
            f"{w:.4f} ({w * 100:.2f}%)",
            (w, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9,
        )
    plt.tight_layout()
    plot_path = REPORTS_DIR / "decision_tree_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot tersimpan: {plot_path.name}")

    # 3. Ekstraksi IF-THEN Rules Lengkap & Humanized (Mengganti `feature N` dengan nama kolom)
    raw_debug_str = champion_model.toDebugString
    humanized_debug_str = humanize_tree_debug_string(raw_debug_str, ALL_FEATURE_NAMES)

    # Juga latih pohon ringkas (maxDepth=3) pada Train Set untuk evaluasi Fidelity & ekstrak 8 aturan daun murni
    print("\n[STEP 2] Melatih Pohon Dangkal (maxDepth=3) untuk Evaluasi Fidelity & Aturan Operasional (E6)...")
    train_df = spark.read.parquet(str(TRAIN_PARQUET_DIR)).cache()
    test_df = spark.read.parquet(str(TEST_PARQUET_DIR)).cache()

    dt_shallow = DecisionTreeClassifier(
        featuresCol="features",
        labelCol=TARGET_COL,
        maxDepth=3,
        minInstancesPerNode=20,
        impurity="gini",
        seed=42,
    )
    shallow_model = dt_shallow.fit(train_df)
    shallow_debug_humanized = humanize_tree_debug_string(shallow_model.toDebugString, ALL_FEATURE_NAMES)

    rules_header = (
        "# ============================================================================\n"
        "# DOKUMEN EKSTRAKSI ATURAN KEPUTUSAN (IF-THEN RULES) — PYSPARK DECISION TREE\n"
        "# Catatan: Seluruh indeks `feature N` telah dipetakan otomatis ke nama kolom asli.\n"
        "# ============================================================================\n\n"
        "## BAGIAN 1: POHON KEPUTUSAN RINGKAS (maxDepth=3, 15 Nodes, 8 Leaf Rules)\n"
        "----------------------------------------------------------------------------\n"
        f"{shallow_debug_humanized}\n"
        "## BAGIAN 2: CHAMPION MODEL PENUH (maxDepth=10, minInstancesPerNode=20, Gini)\n"
        "----------------------------------------------------------------------------\n"
        f"{humanized_debug_str}"
    )

    rules_path = REPORTS_DIR / "decision_tree_rules.txt"
    rules_path.write_text(rules_header, encoding="utf-8")
    print(f"[SAVED] Aturan keputusan lengkap (Humanized) tersimpan di: {rules_path.name}")

    # 4. Evaluasi Fidelity (Kesepakatan Prediksi maxDepth=3 vs Champion maxDepth=10 pada Test Set)
    champ_preds_df = champion_model.transform(test_df)
    shallow_preds_df = shallow_model.transform(test_df)

    champ_metrics, y_true, _, y_pred_champ = evaluate_spark_predictions_single_pass(champ_preds_df, TARGET_COL)
    shallow_metrics, _, _, y_pred_shallow = evaluate_spark_predictions_single_pass(shallow_preds_df, TARGET_COL)

    agreement_rate = float(np.mean(y_pred_shallow == y_pred_champ))
    disagreement_count = int(np.sum(y_pred_shallow != y_pred_champ))
    print(
        f"[FIDELITY E6] Shallow Tree (maxDepth=3, 8 aturan daun) mencapai ROC-AUC={shallow_metrics['roc_auc']:.4f}, "
        f"Akurasi={shallow_metrics['accuracy_pct']:.2f}%, dan Fidelity={agreement_rate * 100:.2f}% "
        f"terhadap Champion Tree (maxDepth=10, {champion_model.numNodes} node)."
    )

    # 5. Evaluasi Empiris 3 Tier Aturan Operasional pada Data Uji (200.277 baris)
    # Berdasarkan split aktual pohon keputusan pada impulse_buying_score & stress_from_financial_decisions
    imp_idx = ALL_FEATURE_NAMES.index("impulse_buying_score")
    str_idx = ALL_FEATURE_NAMES.index("stress_from_financial_decisions")
    test_with_cols = test_df.withColumn("_f_arr", vector_to_array(F.col("features"))).withColumn(
        "impulse_buying_score", F.col("_f_arr")[imp_idx]
    ).withColumn(
        "stress_from_financial_decisions", F.col("_f_arr")[str_idx]
    )

    tier_expr = (
        F.when(
            (F.col("impulse_buying_score") > 5.5)
            | ((F.col("impulse_buying_score") > 4.5) & (F.col("stress_from_financial_decisions") > 5.5)),
            F.lit("Tier 1: Risiko Kritis (Impulse > 5.5 atau [Impulse == 5 & Stress > 5.5])"),
        )
        .when(
            (F.col("impulse_buying_score") > 4.5) & (F.col("stress_from_financial_decisions") <= 5.5),
            F.lit("Tier 2: Risiko Ambang/Moderat (Impulse == 5 & Stress <= 5.5)"),
        )
        .otherwise(F.lit("Tier 3: Risiko Rendah/Aman (Impulse <= 4.5)"))
    )

    tier_df = (
        test_with_cols.withColumn("rule_tier", tier_expr)
        .groupBy("rule_tier")
        .agg(
            F.count("*").alias("test_users"),
            F.sum(F.col(TARGET_COL)).alias("high_abandon_users"),
            F.round(F.avg(F.col(TARGET_COL)) * 100, 2).alias("high_abandon_rate_pct"),
            F.round(F.avg("cart_abandonment_rate"), 2).alias("avg_cart_abandonment_rate"),
            F.round(F.avg("impulse_buying_score"), 2).alias("avg_impulse_score"),
            F.round(F.avg("stress_from_financial_decisions"), 2).alias("avg_financial_stress"),
        )
        .orderBy("rule_tier")
        .toPandas()
    )
    total_test_users = int(tier_df["test_users"].sum())
    tier_df["coverage_pct"] = (tier_df["test_users"] / total_test_users * 100.0).round(2)

    print("\n[TIER OPERASIONAL AKTUAL PADA DATA UJI (N = 200,277)]:")
    print(tier_df.to_string(index=False))

    top_splits = extract_top_split_thresholds(raw_debug_str, ALL_FEATURE_NAMES)

    payload = {
        "champion_num_nodes": int(champion_model.numNodes),
        "champion_depth": int(champion_model.depth),
        "shallow_d3_num_nodes": int(shallow_model.numNodes),
        "shallow_d3_depth": int(shallow_model.depth),
        "shallow_d3_leaf_rules": 8,
        "champion_test_metrics": champ_metrics,
        "shallow_d3_test_metrics": shallow_metrics,
        "fidelity_shallow_vs_champion": {
            "agreement_rate": round(agreement_rate, 6),
            "agreement_pct": round(agreement_rate * 100.0, 2),
            "disagreement_count": disagreement_count,
            "total_test_rows": len(y_true),
            "auc_retention_pct": round(shallow_metrics["roc_auc"] / champ_metrics["roc_auc"] * 100.0, 2),
            "accuracy_retention_pct": round(shallow_metrics["accuracy"] / champ_metrics["accuracy"] * 100.0, 2),
        },
        "top_tree_splits_level_0_to_3": top_splits[:15],
        "shallow_tree_humanized": shallow_debug_humanized,
        "operational_3_tier_rules_test_evaluation": tier_df.to_dict(orient="records"),
        "top_15_feature_importances": fi_df.head(15).to_dict(orient="records"),
    }
    save_json_metrics("rules_and_fidelity.json", payload)
    spark.stop()
    return payload


if __name__ == "__main__":
    run_extraction()
