"""
Modul Segmentasi Perilaku Pengguna E-Commerce (PySpark MLlib K-Means Clustering)
Memenuhi Rule 4 AGENTS.md & 00_PROJECT_CHARTER.md:
- Menstandarisasi fitur psikologis, impulsivitas, dan perilaku belanja pada 1.000.000 pengguna.
- Mengevaluasi jumlah klaster k in {2, 3, 4, 5} menggunakan Silhouette Score & Within-Set Sum of Squared Errors (WSSSE).
- Mengekstrak profil Persona Belanja E-Commerce dan memvisualisasikan distribusi risiko pembatalan keranjang per klaster.
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F
from spark_session import get_spark_session
from pipeline_utils import (
    PROCESSED_PARQUET_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TARGET_SOURCE_COL,
    save_json_metrics,
)

CLUSTER_FEATURES = [
    "impulse_buying_score",
    "stress_from_financial_decisions",
    "overall_stress_level",
    "impulse_purchases_per_month",
    "monthly_spend",
    "average_order_value",
    "daily_session_time_minutes",
    "purchase_conversion_rate",
    "coupon_usage_frequency",
    "return_rate",
]


def run_clustering():
    spark = get_spark_session(app_name="EcommerceKMeansSegmentation")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Membaca dataset Parquet 1M baris dari: {PROCESSED_PARQUET_DIR}")
    df = spark.read.parquet(str(PROCESSED_PARQUET_DIR)).select(
        *CLUSTER_FEATURES, TARGET_SOURCE_COL, TARGET_COL
    )

    print("[STEP 1] Merakit dan menstandarisasi (Z-score) fitur perilaku & psikologis...")
    assembler = VectorAssembler(inputCols=CLUSTER_FEATURES, outputCol="raw_cluster_features")
    assembled_df = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="raw_cluster_features",
        outputCol="scaled_cluster_features",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df).select(
        *CLUSTER_FEATURES, TARGET_SOURCE_COL, TARGET_COL, "scaled_cluster_features"
    ).cache()

    total_rows = scaled_df.count()
    print(f"[INFO] Data siap untuk K-Means Clustering: {total_rows:,} baris x {len(CLUSTER_FEATURES)} fitur.")

    # Gunakan sampel terstratifikasi 200.000 baris untuk evaluasi cepat Silhouette Score (O(N) dengan squaredEuclidean)
    eval_sample_df = scaled_df.sample(withReplacement=False, fraction=0.20, seed=42).cache()
    evaluator = ClusteringEvaluator(
        featuresCol="scaled_cluster_features",
        predictionCol="cluster_id",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    k_candidates = [2, 3, 4, 5]
    k_eval_rows = []
    trained_models = {}

    print("\n[STEP 2] Mengevaluasi K-Means untuk k in {2, 3, 4, 5}...")
    for k in k_candidates:
        t0 = time.perf_counter()
        kmeans = KMeans(
            featuresCol="scaled_cluster_features",
            predictionCol="cluster_id",
            k=k,
            seed=42,
            maxIter=30,
        )
        km_model = kmeans.fit(scaled_df)
        fit_time_s = time.perf_counter() - t0
        wssse = float(km_model.summary.trainingCost)

        sample_preds = km_model.transform(eval_sample_df)
        sil_score = float(evaluator.evaluate(sample_preds))
        trained_models[k] = km_model

        k_eval_rows.append(
            {
                "k": k,
                "silhouette_score": round(sil_score, 6),
                "wssse": round(wssse, 2),
                "fit_time_s": round(fit_time_s, 3),
            }
        )
        print(f"  - k={k}: Silhouette = {sil_score:.4f} | WSSSE = {wssse:,.1f} | Waktu = {fit_time_s:.2f}s")

    # Pilih k=3 untuk segmentasi 3-Tier Persona Operasional (Rendah, Ambang/Moderat, Kritis)
    # Sekaligus catat k dengan Silhouette tertinggi (biasanya k=2 karena struktur biner psikologis)
    best_sil_row = max(k_eval_rows, key=lambda r: r["silhouette_score"])
    selected_k = 3
    selected_model = trained_models[selected_k]
    clustered_df = selected_model.transform(scaled_df)

    print(f"\n[STEP 3] Menyusun Profil Persona Pengguna untuk k={selected_k} (3-Tier Behavioral Personas)...")
    profile_pdf = (
        clustered_df.groupBy("cluster_id")
        .agg(
            F.count("*").alias("user_count"),
            F.round(F.avg(TARGET_SOURCE_COL), 2).alias("avg_cart_abandonment_rate"),
            F.round(F.avg(F.col(TARGET_COL)) * 100.0, 2).alias("high_abandonment_pct"),
            F.round(F.avg("impulse_buying_score"), 2).alias("avg_impulse_score"),
            F.round(F.avg("stress_from_financial_decisions"), 2).alias("avg_financial_stress"),
            F.round(F.avg("overall_stress_level"), 2).alias("avg_overall_stress"),
            F.round(F.avg("impulse_purchases_per_month"), 2).alias("avg_impulse_purchases"),
            F.round(F.avg("monthly_spend"), 2).alias("avg_monthly_spend"),
            F.round(F.avg("average_order_value"), 2).alias("avg_order_value"),
            F.round(F.avg("daily_session_time_minutes"), 2).alias("avg_session_minutes"),
            F.round(F.avg("purchase_conversion_rate"), 2).alias("avg_conversion_rate"),
        )
        .orderBy("avg_cart_abandonment_rate")
        .toPandas()
    )

    # Beri label persona terurut dari tingkat pembatalan terendah ke tertinggi
    persona_labels = [
        "Persona A: Mindful & Low-Stress Shoppers (Risiko Rendah)",
        "Persona B: Moderate/Threshold Shoppers (Risiko Moderat)",
        "Persona C: High-Stress Impulsive Browsers (Risiko Kritis)",
    ]
    profile_pdf["persona_rank"] = [1, 2, 3][: len(profile_pdf)]
    profile_pdf["persona_name"] = persona_labels[: len(profile_pdf)]
    profile_pdf["share_pct"] = (profile_pdf["user_count"] / total_rows * 100.0).round(2)

    print("\n" + "=" * 80)
    print("PROFIL SEGMENTASI PENGGUNA K-MEANS (k=3, Urut Berdasarkan Risiko Pembatalan)")
    print("=" * 80)
    print(
        profile_pdf[
            [
                "cluster_id",
                "persona_name",
                "user_count",
                "share_pct",
                "avg_cart_abandonment_rate",
                "high_abandonment_pct",
                "avg_impulse_score",
                "avg_financial_stress",
                "avg_overall_stress",
            ]
        ].to_string(index=False)
    )
    print("=" * 80)

    # Juga simpan profil k=2 (klaster alami dengan Silhouette tertinggi)
    km2_df = trained_models[2].transform(scaled_df)
    profile_k2_pdf = (
        km2_df.groupBy("cluster_id")
        .agg(
            F.count("*").alias("user_count"),
            F.round(F.avg(TARGET_SOURCE_COL), 2).alias("avg_cart_abandonment_rate"),
            F.round(F.avg(F.col(TARGET_COL)) * 100.0, 2).alias("high_abandonment_pct"),
            F.round(F.avg("impulse_buying_score"), 2).alias("avg_impulse_score"),
            F.round(F.avg("stress_from_financial_decisions"), 2).alias("avg_financial_stress"),
            F.round(F.avg("overall_stress_level"), 2).alias("avg_overall_stress"),
            F.round(F.avg("impulse_purchases_per_month"), 2).alias("avg_impulse_purchases"),
        )
        .orderBy("avg_cart_abandonment_rate")
        .toPandas()
    )
    profile_k2_pdf["share_pct"] = (profile_k2_pdf["user_count"] / total_rows * 100.0).round(2)

    # Visualisasi K-Means: Panel Kiri = Silhouette & WSSSE, Panel Kanan = Profil 3 Persona (k=3)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ks = [r["k"] for r in k_eval_rows]
    sils = [r["silhouette_score"] for r in k_eval_rows]
    wssses = [r["wssse"] / 1e6 for r in k_eval_rows]

    ax_left = axes[0]
    ax_left_r = ax_left.twinx()
    l1 = ax_left.plot(ks, sils, marker="o", linewidth=2.6, color="#0275d8", label="Silhouette Score (Sumbu Kiri)")
    l2 = ax_left_r.plot(
        ks, wssses, marker="s", linewidth=2.4, linestyle="--", color="#d9534f", label="WSSSE / Juta (Sumbu Kanan)"
    )
    ax_left.set_title("Evaluasi Jumlah Klaster K-Means (k = 2 s.d. 5)", fontsize=11.5, fontweight="bold")
    ax_left.set_xlabel("Jumlah Klaster (k)", fontsize=10)
    ax_left.set_ylabel("Silhouette Score (Squared Euclidean)", fontsize=10, color="#0275d8")
    ax_left_r.set_ylabel("Within-Set SSE (Juta Unit)", fontsize=10, color="#d9534f")
    ax_left.set_xticks(ks)
    ax_left_r.grid(False)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="upper right", frameon=True, fontsize=9)

    # Panel Kanan: Perbandingan Metrik Utama antar 3 Persona (k=3)
    ax_right = axes[1]
    short_names = [
        f"Persona A\n(Low Risk)\n{profile_pdf.iloc[0]['share_pct']:.1f}%",
        f"Persona B\n(Moderate)\n{profile_pdf.iloc[1]['share_pct']:.1f}%",
        f"Persona C\n(Critical Risk)\n{profile_pdf.iloc[2]['share_pct']:.1f}%",
    ]
    x_pos = range(len(short_names))
    width = 0.25

    bars1 = ax_right.bar(
        [x - width for x in x_pos],
        profile_pdf["avg_cart_abandonment_rate"],
        width=width,
        color="#d9534f",
        label="Avg Cart Abandonment (%)",
    )
    bars2 = ax_right.bar(
        x_pos,
        profile_pdf["avg_impulse_score"] * 10.0,
        width=width,
        color="#f0ad4e",
        label="Avg Impulse Score (x10)",
    )
    bars3 = ax_right.bar(
        [x + width for x in x_pos],
        profile_pdf["avg_financial_stress"] * 10.0,
        width=width,
        color="#5bc0de",
        label="Avg Financial Stress (x10)",
    )
    ax_right.set_title("Profil Perilaku & Pembatalan per Persona (k=3)", fontsize=11.5, fontweight="bold")
    ax_right.set_xticks(list(x_pos))
    ax_right.set_xticklabels(short_names, fontsize=9.5, fontweight="bold")
    ax_right.set_ylabel("Nilai Rata-rata (Skala 0 - 100)", fontsize=10)
    ax_right.set_ylim(0, 88)
    ax_right.legend(loc="upper left", frameon=True, fontsize=9)

    for b in list(bars1) + list(bars2) + list(bars3):
        h = b.get_height()
        ax_right.annotate(
            f"{h:.1f}",
            (b.get_x() + b.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    plot_path = REPORTS_DIR / "kmeans_persona_clusters.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot K-Means Persona tersimpan di: {plot_path}")

    payload = {
        "total_rows": int(total_rows),
        "cluster_features": CLUSTER_FEATURES,
        "k_evaluation": k_eval_rows,
        "best_silhouette_k": int(best_sil_row["k"]),
        "best_silhouette_score": float(best_sil_row["silhouette_score"]),
        "selected_operational_k": selected_k,
        "personas_k3": profile_pdf.to_dict(orient="records"),
        "personas_k2": profile_k2_pdf.to_dict(orient="records"),
    }
    save_json_metrics("kmeans_segmentation.json", payload)
    spark.stop()
    return payload


if __name__ == "__main__":
    run_clustering()
