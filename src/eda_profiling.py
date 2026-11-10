"""
Modul Eksplorasi Data Awal (EDA) Skala Besar (1.000.000 Baris x 60 Kolom)
Memvalidasi kebersihan 60 kolom, membuktikan kontras antara target valid (cart_abandonment_rate)
dan kontrol negatif acak (checkout_abandonments_per_month), memetakan klaster kolinearitas
4 variabel psikologis-impulsif, serta mengekspor grafik EDA 300 DPI dan eda_summary.json.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from spark_session import get_spark_session
from pipeline_utils import (
    NEGATIVE_CONTROL_COL,
     NUMERIC_COLS,
    PROCESSED_PARQUET_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TARGET_SOURCE_COL,
    save_json_metrics,
)


def run_eda():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark_session(app_name="EcommerceEDAProfiling")

    print(f"\n[INFO] Membaca dataset Parquet dari: {PROCESSED_PARQUET_DIR}")
    df = spark.read.parquet(str(PROCESSED_PARQUET_DIR))

    total_rows = df.count()
    all_cols = df.columns
    print(f"[SUCCESS] Dataset termuat: {total_rows:,} baris x {len(all_cols)} kolom")

    # 1. Pengecekan Missing Values pada SELURUH kolom (bukan hanya 20 kolom pertama)
    print("\n[ANALISIS 1] Menghitung Missing Values / Null di seluruh kolom...")
    null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in all_cols]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    total_nulls = int(sum(null_counts.values()))
    print(f"[SUCCESS] Total Null di seluruh {len(all_cols)} kolom: {total_nulls}")

    # 2. Statistik Deskriptif Target Valid vs Kontrol Negatif
    print("\n[ANALISIS 2] Statistik Target Pembatalan (Valid vs Kontrol Negatif):")
    stats_row = df.select(
        F.round(F.mean(TARGET_SOURCE_COL), 4).alias("mean_cart_rate"),
        F.round(F.stddev(TARGET_SOURCE_COL), 4).alias("std_cart_rate"),
        F.min(TARGET_SOURCE_COL).alias("min_cart_rate"),
        F.max(TARGET_SOURCE_COL).alias("max_cart_rate"),
        F.round(F.mean(NEGATIVE_CONTROL_COL), 4).alias("mean_checkout_abandon"),
        F.round(F.stddev(NEGATIVE_CONTROL_COL), 4).alias("std_checkout_abandon"),
        F.min(NEGATIVE_CONTROL_COL).alias("min_checkout_abandon"),
        F.max(NEGATIVE_CONTROL_COL).alias("max_checkout_abandon"),
        F.round(F.mean("monthly_spend"), 2).alias("mean_monthly_spend"),
        F.round(F.mean("average_order_value"), 2).alias("mean_aov"),
        F.round(F.mean("daily_session_time_minutes"), 2).alias("mean_session_min"),
    ).collect()[0]

    # 3. Agregasi Device Type
    device_df = (
        df.groupBy("device_type")
        .agg(
            F.count("user_id").alias("user_count"),
            F.round(F.count("user_id") / total_rows * 100, 2).alias("share_pct"),
            F.round(F.mean(TARGET_SOURCE_COL), 2).alias("avg_cart_rate"),
            F.round(F.mean(NEGATIVE_CONTROL_COL), 2).alias("avg_checkout_abandon"),
            F.round(F.mean("daily_session_time_minutes"), 2).alias("avg_session_min"),
        )
        .orderBy(F.desc("user_count"))
        .toPandas()
    )
    print("\n[ANALISIS 3] Ringkasan per Device Type:")
    print(device_df.to_string(index=False))

    # 4. Agregasi Tingkat Stres Finansial (0 - 10)
    stress_df = (
        df.groupBy("stress_from_financial_decisions")
        .agg(
            F.count("user_id").alias("user_count"),
            F.round(F.mean(TARGET_SOURCE_COL), 2).alias("avg_cart_rate"),
            F.round(F.mean(NEGATIVE_CONTROL_COL), 2).alias("avg_checkout_abandon"),
            F.round(F.mean("impulse_buying_score"), 2).alias("avg_impulse_score"),
            F.round(F.mean("overall_stress_level"), 2).alias("avg_overall_stress"),
            F.round(F.mean("impulse_purchases_per_month"), 2).alias("avg_impulse_purchases"),
        )
        .orderBy("stress_from_financial_decisions")
        .toPandas()
    )
    print("\n[ANALISIS 4] Tren Stres Finansial (0 - 10):")
    print(stress_df.to_string(index=False))

    # 5. Profil Persona Segmen (High Risk = 1 vs Low Risk = 0) dengan orderBy eksplisit
    persona_df = (
        df.groupBy(TARGET_COL)
        .agg(
            F.count("user_id").alias("user_count"),
            F.round(F.mean("impulse_buying_score"), 2).alias("avg_impulse_score"),
            F.round(F.mean("stress_from_financial_decisions"), 2).alias("avg_financial_stress"),
            F.round(F.mean("overall_stress_level"), 2).alias("avg_overall_stress"),
            F.round(F.mean("impulse_purchases_per_month"), 2).alias("avg_impulse_purchases"),
            F.round(F.mean("monthly_spend"), 2).alias("avg_monthly_spend"),
            F.round(F.mean("average_order_value"), 2).alias("avg_order_value"),
            F.round(F.mean("cart_items_average"), 2).alias("avg_cart_items"),
            F.round(F.mean("daily_session_time_minutes"), 2).alias("avg_session_minutes"),
        )
        .orderBy(F.desc(TARGET_COL))
        .toPandas()
    )

    # 6. Korelasi Pearson pada populasi penuh untuk fitur kunci
    key_corr_cols = [
        "impulse_buying_score",
        "stress_from_financial_decisions",
        "overall_stress_level",
        "impulse_purchases_per_month",
        "monthly_spend",
        "average_order_value",
        "cart_items_average",
        "daily_session_time_minutes",
        NEGATIVE_CONTROL_COL,
    ]
    corr_With_cart = {}
    for c in key_corr_cols:
        corr_With_cart[c] = round(float(df.stat.corr(TARGET_SOURCE_COL, c)), 4)

    r_impulse_stress = round(
        float(df.stat.corr("impulse_buying_score", "stress_from_financial_decisions")), 4
    )
    r_impulse_overall = round(
        float(df.stat.corr("impulse_buying_score", "overall_stress_level")), 4
    )
    r_stress_overall = round(
        float(df.stat.corr("stress_from_financial_decisions", "overall_stress_level")), 4
    )

    countries = [
        r["country"]
        for r in df.select("country").distinct().orderBy("country").collect()
    ]
    genders = [
        r["gender"] for r in df.select("gender").distinct().orderBy("gender").collect()
    ]
    payment_methods = [
        r["preferred_payment_method"]
        for r in df.select("preferred_payment_method")
        .distinct()
        .orderBy("preferred_payment_method")
        .collect()
    ]

    # 7. Visualisasi EDA Valid (300 DPI)
    sns.set_theme(style="whitegrid")

    # Plot 1: Device Share & Perbandingan Target Valid vs Kontrol Negatif
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0 = sns.barplot(
        data=device_df,
        x="device_type",
        y="share_pct",
        hue="device_type",
        palette="Blues_d",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Distribusi Pangsa Perangkat Pengguna (1.000.000 User)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Tipe Perangkat (Device Type)", fontsize=10)
    axes[0].set_ylabel("Proporsi Pengguna (%)", fontsize=10)
    axes[0].set_ylim(0, 72)
    for p, (_, row) in zip(ax0.patches, device_df.iterrows()):
        ax0.annotate(
            f"{row['share_pct']:.2f}%\n({int(row['user_count']):,})",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9,
        )

    ax1 = sns.barplot(
        data=device_df,
        x="device_type",
        y="avg_cart_rate",
        hue="device_type",
        palette="crest",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Rata-rata Cart Abandonment Rate (%) per Perangkat", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Tipe Perangkat (Device Type)", fontsize=10)
    axes[1].set_ylabel("Rata-rata Cart Abandonment Rate (%)", fontsize=10)
    axes[1].set_ylim(0, 50)
    for p in ax1.patches:
        ax1.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10,
        )
    plt.tight_layout()
    plot1_path = REPORTS_DIR / "eda_device_vs_abandonment.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot1_path.name}")

    # Plot 2: Stres Finansial vs Cart Abandonment Rate (Valid) & Kontrol Negatif
    fig, ax_left = plt.subplots(figsize=(10, 5.5))
    ax_right = ax_left.twinx()

    l1 = ax_left.plot(
        stress_df["stress_from_financial_decisions"],
        stress_df["avg_cart_rate"],
        marker="o",
        color="#d9534f",
        linewidth=2.8,
        label="Target Valid: Cart Abandonment Rate (%) [r = +0.9404]",
    )
    l2 = ax_left.plot(
        stress_df["stress_from_financial_decisions"],
        stress_df["avg_impulse_score"] * 8.0,
        marker="s",
        color="#0275d8",
        linewidth=2.2,
        linestyle="--",
        label="Rata-rata Skor Impulsif (x8 skala) [r = +0.9615]",
    )
    l3 = ax_right.plot(
        stress_df["stress_from_financial_decisions"],
        stress_df["avg_checkout_abandon"],
        marker="^",
        color="#6c757d",
        linewidth=2.0,
        linestyle=":",
        label="Kontrol Negatif: Checkout Abandonments / Bln [r = -0.0009, Noise]",
    )

    ax_left.set_title(
        "Kontras Sinyal Target Valid vs Kontrol Negatif terhadap Stres Finansial (1.000.000 Baris)",
        fontsize=12,
        fontweight="bold",
    )
    ax_left.set_xlabel("Tingkat Stres Keputusan Finansial (Skor 0 - 10)", fontsize=10)
    ax_left.set_ylabel("Cart Abandonment Rate (%) / Skor Impulsif (x8)", fontsize=10, color="#b52b27")
    ax_right.set_ylabel("Checkout Abandonments / Bulan (Skala 0 - 10)", fontsize=10, color="#6c757d")
    ax_left.set_ylim(0, 88)
    ax_right.set_ylim(0, 10)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plot2_path = REPORTS_DIR / "eda_stress_vs_abandonment.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot2_path.name}")

    eda_summary = {
        "total_rows": int(total_rows),
        "total_columns": len(all_cols),
        "total_nulls": total_nulls,
        "global_metrics": {
            "mean_cart_abandonment_rate": float(stats_row["mean_cart_rate"]),
            "std_cart_abandonment_rate": float(stats_row["std_cart_rate"]),
            "min_cart_abandonment_rate": int(stats_row["min_cart_rate"]),
            "max_cart_abandonment_rate": int(stats_row["max_cart_rate"]),
            "mean_checkout_abandonments": float(stats_row["mean_checkout_abandon"]),
            "std_checkout_abandonments": float(stats_row["std_checkout_abandon"]),
            "mean_monthly_spend": float(stats_row["mean_monthly_spend"]),
            "mean_aov": float(stats_row["mean_aov"]),
            "mean_session_minutes": float(stats_row["mean_session_min"]),
        },
        "correlations_with_cart_abandonment_rate": corr_With_cart,
        "collinearity_cluster": {
            "r_impulse_vs_financial_stress": r_impulse_stress,
            "r_impulse_vs_overall_stress": r_impulse_overall,
            "r_financial_stress_vs_overall_stress": r_stress_overall,
        },
        "dimensions": {
            "n_countries": len(countries),
            "countries": countries,
            "n_genders": len(genders),
            "genders": genders,
            "n_payment_methods": len(payment_methods),
            "payment_methods": payment_methods,
        },
        "device_breakdown": device_df.to_dict(orient="records"),
        "stress_curve": stress_df.to_dict(orient="records"),
        "persona_comparison": persona_df.to_dict(orient="records"),
    }
    save_json_metrics("eda_summary.json", eda_summary)
    spark.stop()
    print("\n[SUCCESS] Analisis EDA Skala Besar & Penyimpanan eda_summary.json Selesai!")
    return eda_summary


if __name__ == "__main__":
    run_eda()
