"""
Membangun reports/dashboard_visualisasi.html (dashboard 3 tab) langsung dari artefak metrik.

Tab:
  1. Data & Pipeline          : konteks dataset sintetis dan pipeline Spark (deskriptif saja).
  2. Evaluasi Model           : fokus utama, seluruh hasil pembandingan model.
  3. Interpretabilitas & Arah : pohon surrogate, simulator aturan, atribusi fitur, saran pengembangan.

Semua angka dibaca dari reports/metrics/*.json|*.csv sehingga dashboard selalu sinkron dengan
pipeline. Grafik digambar dengan SVG murni (tanpa CDN) agar file bisa dibuka offline.

Jalankan: python src/build_dashboard_html.py
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "reports" / "metrics"
OUT_PATH = BASE_DIR / "reports" / "dashboard_visualisasi.html"

MODEL_ORDER = ["DT_Champion_d10", "LR_Tuned", "GBT_Tuned", "RF_Tuned", "DT_Default_d7", "DT_Shallow_d3"]
MODEL_LABEL = {
    "DT_Champion_d10": "DT-d10",
    "LR_Tuned": "LR",
    "GBT_Tuned": "GBT",
    "RF_Tuned": "RF",
    "DT_Default_d7": "DT-d7",
    "DT_Shallow_d3": "DT-d3",
}
COLLINEAR = [
    "impulse_buying_score",
    "stress_from_financial_decisions",
    "overall_stress_level",
    "impulse_purchases_per_month",
]


def _load(name):
    return json.loads((METRICS_DIR / name).read_text(encoding="utf-8"))


def _csv(name):
    with open(METRICS_DIR / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _thin(xs, ys, max_pts=120):
    """Kurangi titik kurva agar file tetap ringan, tetap mempertahankan titik awal dan akhir."""
    n = len(xs)
    if n <= max_pts:
        return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(xs, ys)]
    step = (n - 1) / (max_pts - 1)
    idx = sorted({round(i * step) for i in range(max_pts)})
    return [[round(float(xs[i]), 5), round(float(ys[i]), 5)] for i in idx]


def _size_label(key, row):
    n = row["num_nodes_or_weights"]
    fmt = f"{n:,}".replace(",", ".")
    if key == "LR_Tuned":
        return f"{fmt} bobot"
    params = row["params"]
    trees = 1
    for tok in ("numTrees=", "maxIter="):
        if tok in params:
            trees = int(params.split(tok)[1].split(",")[0])
    return f"{trees} pohon / {fmt} simpul"


def collect_data():
    eda = _load("eda_summary.json")
    prep = _load("preprocessing_summary.json")
    feat = _load("feature_metadata.json")
    comp = _load("model_comparison_summary.json")
    dt = _load("dt_champion_summary.json")
    rules = _load("rules_and_fidelity.json")
    xai = _load("explainability_shap_permutation.json")
    curves = _load("roc_pr_curves.json")
    grid = _csv("dt_tuning_grid.csv")

    comp_rows = {r["model_key"]: r for r in comp["models_comparison"]}
    models = []
    for key in MODEL_ORDER:
        r = comp_rows[key]
        models.append(
            {
                "key": key,
                "label": MODEL_LABEL[key],
                "name": r["model_name"],
                "size": _size_label(key, r),
                "nodes": r["num_nodes_or_weights"],
                "time": r["train_time_s"],
                "roc": r["roc_auc"],
                "roc_lo": r["roc_auc_ci95_low"],
                "roc_hi": r["roc_auc_ci95_high"],
                "pr": r["pr_auc"],
                "acc": r["accuracy_pct"],
                "f1": r["f1_weighted"],
                "cv": r["cv5_roc_auc_mean"],
                "cv_sd": r["cv5_roc_auc_sd"],
                "folds": r["fold_roc_aucs"],
            }
        )

    curve_data = {
        key: {
            "roc": _thin(curves[key]["roc"]["fpr"], curves[key]["roc"]["tpr"]),
            "pr": _thin(curves[key]["pr"]["recall"], curves[key]["pr"]["precision"]),
        }
        for key in MODEL_ORDER
    }

    sens = comp["sensitivity_and_negative_control"]
    validity = {}
    for row in sens:
        exp = row["experiment"]
        validity.setdefault(exp, {"target": row["target_definition"], "auc": {}})
        validity[exp]["auc"][row["model_key"]] = row["roc_auc"]

    attr_rows = {r["feature"]: r for r in xai["feature_attribution_table"]}
    shap_total = sum(r["mean_abs_shap_rf"] for r in xai["feature_attribution_table"])
    attribution = [
        {
            "feature": f,
            "dt_gini": attr_rows[f]["spark_dt_gini"] * 100,
            "rf_gini": attr_rows[f]["rf_ensemble_gini"] * 100,
            "lr_coef": attr_rows[f]["lr_standardized_weight"] * 100,
            "shap_rf": attr_rows[f]["mean_abs_shap_rf"] / shap_total * 100,
            "perm_dt": attr_rows[f]["perm_auc_drop_dt_mean"],
        }
        for f in COLLINEAR
    ]
    base = xai["sample_baseline_roc_auc"]
    joint = xai["joint_cluster_permutation"]
    joint_perm = [
        {"model": "DT", "before": base["decision_tree"], "after": base["decision_tree"] - joint["joint_auc_drop_dt_mean"]},
        {"model": "RF", "before": base["random_forest"], "after": base["random_forest"] - joint["joint_auc_drop_rf_mean"]},
        {"model": "LR", "before": base["logistic_regression"], "after": base["logistic_regression"] - joint["joint_auc_drop_lr_mean"]},
    ]

    grid_rows = [
        {
            "id": r["config_id"],
            "depth": int(r["maxDepth"]),
            "min_inst": int(r["minInstancesPerNode"]),
            "impurity": r["impurity"],
            "nodes": int(r["num_nodes"]),
            "val_auc": float(r["val_roc_auc"]),
        }
        for r in grid
    ]

    ablation = [
        {
            "id": a["model_id"],
            "depth": a["maxDepth"],
            "min_inst": a["minInstancesPerNode"],
            "impurity": a["impurity"],
            "val_nodes": a["val_num_nodes"],
            "nodes": a["num_nodes"],
            "val_auc": a["val_roc_auc"],
            "test_auc": a["test_roc_auc"],
            "test_acc": a["test_accuracy_pct"],
        }
        for a in dt["ablation_reference_models"]
    ]

    scaling = comp["scaling_benchmarks"]
    io = scaling["storage_io_benchmark"]

    return {
        "dataset": {
            "rows": eda["total_rows"],
            "raw_cols": prep["raw_columns_count"],
            "n_features": feat["tree_feature_vector_dim"],
            "n_numeric": feat["numeric_feature_count"],
            "n_categorical": feat["categorical_feature_count"],
            "lr_dims": feat["lr_ohe_scaled_vector_dim"],
            "train_rows": comp["dataset_split"]["train_rows"],
            "test_rows": comp["dataset_split"]["test_rows"],
            "pos_pct": prep["class_distribution"]["1"]["percentage"],
            "mean_cart": eda["global_metrics"]["mean_cart_abandonment_rate"],
            "csv_mb": io["csv_size_mb"],
            "parquet_mb": io["parquet_size_mb"],
            "size_reduction": io["size_reduction_pct"],
            "csv_s": io["csv_read_agg_median_s"],
            "parquet_s": io["parquet_read_agg_median_s"],
            "io_speedup": io["io_speedup_factor"],
        },
        "eda": {
            "stress": [[r["stress_from_financial_decisions"], r["avg_cart_rate"]] for r in eda["stress_curve"]],
            "devices": [[d["device_type"], d["share_pct"], d["avg_cart_rate"]] for d in eda["device_breakdown"]],
            "corr": eda["correlations_with_cart_abandonment_rate"],
            "persona": eda["persona_comparison"],
        },
        "models": models,
        "curves": curve_data,
        "mcnemar": comp["mcnemar_paired_tests"],
        "bootstrap": comp["bootstrap_paired_auc_deltas"],
        "validity": validity,
        "confusion": dt["champion_test_metrics"],
        "grid": grid_rows,
        "chosen": dt["best_validation_row"]["config_id"],
        "best_any": dt["best_unpruned_validation_row"]["config_id"],
        "ablation": ablation,
        "scaling_rows": scaling["row_scaling"],
        "scaling_cores": scaling["core_scaling"],
        "surrogate": {
            "fidelity": rules["fidelity_shallow_vs_champion"]["agreement_pct"],
            "acc": rules["shallow_d3_test_metrics"]["accuracy_pct"],
            "auc": rules["shallow_d3_test_metrics"]["roc_auc"],
            "tiers": rules["operational_3_tier_rules_test_evaluation"],
        },
        "attribution": attribution,
        "joint_perm": joint_perm,
        "xai_sizes": xai["sample_sizes"],
    }


TEMPLATE = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard Evaluasi Model: Prediksi Cart Abandonment pada Apache Spark</title>
<style>
:root {
  --bg: #f4f4f2; --surface: #fcfcfb; --border: #e2e1dc; --grid: #ecebe6;
  --text: #1d1d1b; --text-2: #52514e; --muted: #7a7974;
  --accent: #2a78d6; --accent-soft: #e6f0fb;
  --warn-bg: #fff6e0; --warn-border: #f0cf7a;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 20px 28px 0; }
header h1 { margin: 0; font-size: 20px; }
header p { margin: 4px 0 14px; color: var(--text-2); font-size: 13px; }
nav { display: flex; gap: 4px; }
nav button { border: 0; background: none; padding: 10px 16px; font: inherit; font-weight: 600; color: var(--text-2); cursor: pointer; border-bottom: 3px solid transparent; }
nav button[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
nav button small { display: block; font-weight: 400; font-size: 11px; color: var(--muted); }
main { padding: 22px 28px 40px; max-width: 1320px; margin: 0 auto; }
section[role="tabpanel"] { display: none; }
section[role="tabpanel"].active { display: block; }
.note { background: var(--warn-bg); border: 1px solid var(--warn-border); border-radius: 8px; padding: 12px 16px; margin-bottom: 18px; color: #4a3a10; }
.lead { background: var(--accent-soft); border: 1px solid #c3dbf6; border-radius: 8px; padding: 12px 16px; margin-bottom: 18px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 18px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .k { font-size: 12px; color: var(--text-2); }
.tile .v { font-size: 26px; font-weight: 700; margin: 2px 0; font-variant-numeric: tabular-nums; }
.tile .s { font-size: 12px; color: var(--muted); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-bottom: 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
.grid2 .card { margin-bottom: 0; }
.card h3 { margin: 0 0 2px; font-size: 15px; }
.card .sub { margin: 0 0 10px; font-size: 12px; color: var(--text-2); }
.card .cap { margin: 8px 0 0; font-size: 12.5px; color: var(--text-2); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: var(--text-2); margin-bottom: 6px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; background: var(--c); vertical-align: -1px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { font-family: inherit; fill: var(--text-2); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; }
th { color: var(--text-2); font-weight: 600; background: #f7f7f5; }
th:first-child, td:first-child { text-align: left; }
tr.hl td { background: var(--accent-soft); font-weight: 600; }
.flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; }
.flow div { flex: 1 1 140px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-size: 12.5px; position: relative; }
.flow b { display: block; font-size: 13px; color: var(--accent); }
.heat td { text-align: center; color: #fff; font-weight: 600; border: 2px solid var(--surface); }
.heat td.lt { color: var(--text); }
.heat td.sel { outline: 4px solid #eb6834; outline-offset: -4px; }
.heat td.best { outline: 4px dashed #eb6834; outline-offset: -4px; }
.cm { display: grid; grid-template-columns: 110px 1fr 1fr; gap: 4px; max-width: 420px; font-size: 13px; }
.cm div { padding: 10px; border-radius: 6px; text-align: center; }
.cm .h { color: var(--text-2); background: none; font-size: 12px; }
.tree { font-size: 12.5px; }
.sim { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
.sim label { display: block; font-size: 13px; margin-bottom: 12px; }
.sim input[type=range] { width: 100%; }
.sim .out { border-radius: 10px; padding: 14px 16px; border: 1px solid var(--border); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.future { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.future div { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 8px; padding: 12px 14px; font-size: 13px; }
.future b { display: block; margin-bottom: 4px; }
#tip { position: fixed; pointer-events: none; background: #1d1d1b; color: #fff; padding: 6px 9px; border-radius: 6px; font-size: 12px; display: none; z-index: 10; max-width: 260px; }
footer { color: var(--muted); font-size: 12px; text-align: center; padding: 0 0 24px; }
@media (max-width: 720px) { .grid2, .sim { grid-template-columns: 1fr; } main { padding: 16px; } }
</style>
</head>
<body>
<header>
  <h1>Evaluasi Model Klasifikasi pada Apache Spark untuk Prediksi Cart Abandonment</h1>
  <p>Kelompok 8 Teknologi Big Data, TRPL Sekolah Vokasi IPB. Seluruh angka dibaca dari <code>reports/metrics/</code>.</p>
  <nav role="tablist">
    <button role="tab" id="t-data" aria-controls="p-data" aria-selected="false">1. Data &amp; Pipeline<small>konteks dataset sintetis</small></button>
    <button role="tab" id="t-model" aria-controls="p-model" aria-selected="true">2. Evaluasi Model<small>fokus utama</small></button>
    <button role="tab" id="t-xai" aria-controls="p-xai" aria-selected="false">3. Interpretabilitas &amp; Pengembangan<small>aturan, atribusi, saran</small></button>
  </nav>
</header>
<main>

<section role="tabpanel" id="p-data" aria-labelledby="t-data">
  <div class="note"><b>Catatan data sintetis.</b> Dataset Kaggle ini dibangkitkan secara sintetis. Grafik di tab ini hanya menggambarkan cara data dibangkitkan dan menjadi konteks bagi model, bukan temuan tentang perilaku konsumen sebenarnya.</div>
  <div class="tiles" id="data-tiles"></div>
  <div class="card">
    <h3>Alur pipeline Split-Before-Fit</h3>
    <p class="sub">Data uji dipisahkan sebelum transformer apa pun di-fit dan hanya dipakai satu kali pada tahap evaluasi.</p>
    <div class="flow" id="flow"></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Tingkat cart abandonment per skor stres finansial</h3><p class="sub">Rata-rata per level skor (N = 1.000.000)</p><div id="c-stress"></div></div>
    <div class="card"><h3>Korelasi Pearson terhadap cart_abandonment_rate</h3><p class="sub">Empat skor psikologis vs atribut numerik lain</p><div id="c-corr"></div><p class="cap">Sinyal hanya ada pada empat skor yang saling berkorelasi (r = 0,84 sampai 0,92). Atribut numerik lain memiliki |r| di bawah 0,003.</p></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Profil kelas tinggi vs rendah</h3><p class="sub">Rata-rata empat skor psikologis per label</p><div class="legend" id="lg-persona"></div><div id="c-persona"></div></div>
    <div class="card"><h3>Perangkat pengguna</h3><p class="sub">Porsi pengguna dan rata-rata cart abandonment per perangkat</p><div id="t-device"></div><p class="cap">Rata-rata pembatalan hampir sama di setiap perangkat, sehingga perangkat tidak membawa sinyal.</p></div>
  </div>
</section>

<section role="tabpanel" id="p-model" aria-labelledby="t-model" class="active">
  <div class="lead"><b>Ringkasan.</b> LR dan GBT memperoleh ROC-AUC tertinggi, tetapi selisihnya terhadap DT-d10 hanya sekitar 0,0002. DT-d10 dilatih jauh lebih cepat dan strukturnya dapat dibaca. DT-d10 dan RF setara secara praktis karena urutannya berbalik pada validasi silang. Target kontrol negatif berada di sekitar 0,50, yang konsisten dengan pipeline tanpa kebocoran data.</div>
  <div class="tiles" id="model-tiles"></div>
  <div class="card">
    <h3>Kinerja enam model pada data uji</h3>
    <p class="sub">N uji = <span id="n-test"></span>. IK 95% dari bootstrap; CV = validasi silang 5-fold pada sampel 20% data latih.</p>
    <div id="t-models"></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Kurva ROC</h3><p class="sub">Diperbesar pada FPR 0 sampai 0,3</p><div class="legend lg-models"></div><div id="c-roc"></div></div>
    <div class="card"><h3>Kurva precision-recall</h3><p class="sub">Diperbesar pada precision 0,8 sampai 1</p><div class="legend lg-models"></div><div id="c-pr"></div></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Waktu latih vs ROC-AUC uji</h3><p class="sub">Sumbu X logaritmik; ukuran titik sebanding log jumlah simpul atau bobot</p><div id="c-tradeoff"></div><p class="cap">Titik di kiri atas berarti cepat dan akurat. DT-d10 berada sangat dekat dengan LR dan GBT, sedangkan RF paling lambat.</p></div>
    <div class="card"><h3>ROC-AUC uji vs validasi silang</h3><p class="sub">Garis tipis menunjukkan rentang lima fold</p><div class="legend" id="lg-cv"></div><div id="c-cv"></div><p class="cap">Pada CV, RF berada di atas DT-d10, kebalikan dari data uji. DT-d7 turun pada data penuh karena Spark menggabungkan daun bersaudara berkelas sama (SPARK-3159).</p></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Confusion matrix DT-d10</h3><p class="sub">Data uji, ambang probabilitas 0,5</p><div id="cm"></div></div>
    <div class="card"><h3>Uji berpasangan terhadap DT-d10</h3><p class="sub">McNemar dengan koreksi kontinuitas dan selisih ROC-AUC bootstrap</p><div id="t-pairs"></div><p class="cap">Dengan 200 ribu data uji, selisih 0,001 pun signifikan. Besar selisihnya yang perlu dibaca, bukan hanya nilai p.</p></div>
  </div>
  <div class="card">
    <h3>Tuning 72 konfigurasi Decision Tree (ROC-AUC validasi)</h3>
    <p class="sub">Dilatih pada 150.269 baris train-sub dan dievaluasi pada 100.086 baris validasi; data uji tidak disentuh. Bingkai oranye tegas = konfigurasi terpilih (minInstancesPerNode &ge; 20); bingkai oranye putus-putus = skor tertinggi tanpa batas daun. Arahkan kursor untuk jumlah simpul.</p>
    <div class="grid2" id="heat"></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Ablasi Decision Tree</h3><p class="sub">Setiap varian berbeda satu pengaturan; dilatih ulang pada seluruh data latih</p><div id="t-ablation"></div></div>
    <div class="card"><h3>Uji validitas: kontrol negatif dan sensitivitas ambang</h3><p class="sub">ROC-AUC uji; model dilatih ulang pada subsampel data latih</p><div class="legend" id="lg-valid"></div><div id="c-valid"></div></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Skalabilitas terhadap ukuran data latih</h3><p class="sub">Waktu latih (detik, skala log), local[*]</p><div class="legend lg-scale"></div><div id="c-rows"></div></div>
    <div class="card"><h3>Skalabilitas thread DT-d10</h3><p class="sub">Median waktu latih per jumlah thread executor</p><div id="c-cores"></div><p class="cap">Mesin hanya memiliki 2 core fisik. Dua pengulangan pada 1 thread berbeda jauh (23,51 dan 12,42 detik), sehingga angka percepatan bersifat kasar.</p></div>
  </div>
</section>

<section role="tabpanel" id="p-xai" aria-labelledby="t-xai">
  <div class="grid2">
    <div class="card">
      <h3>Pohon surrogate empat daun (DT-d3)</h3>
      <p class="sub">Fidelitas terhadap DT-d10 <b id="fid"></b> pada data uji; akurasi <b id="sacc"></b></p>
      <div id="tree"></div>
    </div>
    <div class="card">
      <h3>Simulator aturan surrogate</h3>
      <p class="sub">Geser skor untuk melihat daun yang dituju dan proporsi pengguna uji berlabel tinggi di daun tersebut.</p>
      <div class="sim">
        <div>
          <label>impulse_buying_score: <b id="v-imp"></b><input type="range" id="s-imp" min="0" max="10" step="1" value="5"></label>
          <label>stress_from_financial_decisions: <b id="v-str"></b><input type="range" id="s-str" min="0" max="10" step="1" value="5"></label>
          <p class="cap">Aturan ini ringkasan dari DT-d10 pada data sintetis. Untuk dipakai di produksi, model harus dilatih ulang dan divalidasi pada data riil.</p>
        </div>
        <div class="out" id="sim-out"></div>
      </div>
    </div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Atribusi fitur pada empat prediktor kolinear</h3><p class="sub">Porsi (%) menurut empat metode</p><div class="legend" id="lg-attr"></div><div id="c-attr"></div><p class="cap">Gini pada pohon tunggal memusatkan bobot pada dua fitur, sedangkan RF, koefisien LR, dan SHAP menyebarkannya. Bobot mendekati nol tidak berarti fitur tidak relevan.</p></div>
    <div class="card"><h3>Permutasi bersama empat prediktor</h3><p class="sub">ROC-AUC model audit sebelum dan sesudah keempat fitur diacak bersamaan</p><div class="legend" id="lg-joint"></div><div id="c-joint"></div><p class="cap">Ketika keempat fitur diacak bersamaan, semua model turun ke tingkat tebakan acak. Hampir seluruh sinyal ada di kelompok fitur ini.</p></div>
  </div>
  <div class="card">
    <h3>Arah pengembangan untuk penelitian berikutnya</h3>
    <p class="sub">Disusun dari keterbatasan yang tercatat pada jurnal dan laporan</p>
    <div class="future">
      <div><b>Tuning dengan validasi silang di Spark</b>Ganti subsampel tetap dengan CrossValidator pada seluruh data latih, lalu bandingkan konfigurasi terpilih dengan D10_M20_GIN.</div>
      <div><b>Uji mekanisme Random Forest</b>Latih RF dengan featureSubsetStrategy = "all" untuk menguji dugaan bahwa sekitar 54% kandidat split tidak memuat fitur informatif (m = 8 dari 57).</div>
      <div><b>Resolusi probabilitas pohon</b>Bandingkan ROC-AUC sebelum dan sesudah penggabungan daun (SPARK-3159), atau tambahkan kalibrasi probabilitas, terutama untuk pohon dangkal.</div>
      <div><b>Atribusi yang memperhitungkan korelasi</b>Gunakan permutasi berkelompok atau SHAP bersyarat sebagai laporan standar ketika prediktor saling berkorelasi kuat.</div>
      <div><b>Pengukuran waktu berulang di klaster</b>Ulangi benchmark waktu latih minimal lima kali pada klaster multinode agar variasi antarpengulangan dan biaya shuffle terukur.</div>
      <div><b>Replikasi pada data riil</b>Jalankan pipeline yang sama pada log clickstream riil yang memuat noise, ketidakseimbangan kelas, dan perubahan pola dari waktu ke waktu.</div>
    </div>
  </div>
</section>
</main>
<footer>Dibangkitkan oleh <code>src/build_dashboard_html.py</code> dari artefak <code>reports/metrics/</code>. Tidak memerlukan koneksi internet.</footer>
<div id="tip"></div>

<script>
const D = __DATA__;
const C = {DT_Champion_d10:"#2a78d6", LR_Tuned:"#eb6834", GBT_Tuned:"#1baf7a", RF_Tuned:"#eda100", DT_Default_d7:"#e87ba4", DT_Shallow_d3:"#008300"};
const NS = "http://www.w3.org/2000/svg";
const nf = (x, d) => Number(x).toLocaleString("id-ID", {minimumFractionDigits: d, maximumFractionDigits: d});
const $ = id => document.getElementById(id);
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px"; }
function hideTip() { tip.style.display = "none"; }
function el(tag, attrs, parent, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  if (parent) parent.appendChild(n);
  return n;
}
function bindTip(node, html) { node.addEventListener("mousemove", e => showTip(e, html)); node.addEventListener("mouseleave", hideTip); }
function legend(id, items) {
  document.querySelectorAll(id).forEach(box => {
    box.innerHTML = items.map(([n, c]) => `<span style="--c:${c}">${n}</span>`).join("");
  });
}
function niceTicks(min, max, n) {
  const span = max - min, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function logTicks(min, max) {
  const out = []; for (let p = Math.floor(Math.log10(min)); p <= Math.ceil(Math.log10(max)); p++) for (const m of [1, 2, 5]) { const v = m * Math.pow(10, p); if (v >= min && v <= max) out.push(v); }
  return out;
}
// Frame generik: mengembalikan fungsi skala dan grup plot
function frame(id, o) {
  const W = o.w || 560, H = o.h || 300, m = Object.assign({l: 54, r: 16, t: 10, b: 42}, o.m || {});
  const host = $(id); host.innerHTML = "";
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || ""}, host);
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const xs = o.xlog ? v => m.l + (Math.log10(v) - Math.log10(o.x0)) / (Math.log10(o.x1) - Math.log10(o.x0)) * pw : v => m.l + (v - o.x0) / (o.x1 - o.x0) * pw;
  const ys = o.ylog ? v => m.t + ph - (Math.log10(v) - Math.log10(o.y0)) / (Math.log10(o.y1) - Math.log10(o.y0)) * ph : v => m.t + ph - (v - o.y0) / (o.y1 - o.y0) * ph;
  const g = el("g", {}, svg);
  const cid = "clip-" + id;
  const cp = el("clipPath", {id: cid}, el("defs", {}, svg));
  el("rect", {x: m.l, y: m.t - 6, width: pw, height: ph + 6}, cp);
  if (!o.noY) {
    const yt = o.yticks || (o.ylog ? logTicks(o.y0, o.y1) : niceTicks(o.y0, o.y1, 5));
    yt.forEach(v => { el("line", {x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v), stroke: "#ecebe6"}, g); el("text", {x: m.l - 6, y: ys(v) + 4, "text-anchor": "end", "font-size": 11}, g, (o.yfmt || (v => nf(v, 2)))(v)); });
  }
  if (!o.noX) {
    const xt = o.xticks || (o.xlog ? logTicks(o.x0, o.x1) : niceTicks(o.x0, o.x1, 6));
    xt.forEach(v => el("text", {x: xs(v), y: H - m.b + 16, "text-anchor": "middle", "font-size": 11}, g, (o.xfmt || (v => nf(v, 1)))(v)));
  }
  el("line", {x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, stroke: "#b9b8b2"}, g);
  if (o.xlab) el("text", {x: m.l + pw / 2, y: H - 6, "text-anchor": "middle", "font-size": 12}, g, o.xlab);
  if (o.ylab) el("text", {x: 12, y: m.t + ph / 2, "text-anchor": "middle", "font-size": 12, transform: `rotate(-90 12 ${m.t + ph / 2})`}, g, o.ylab);
  const plot = el("g", {"clip-path": `url(#${cid})`}, svg);
  return {svg, g, plot, xs, ys, m, pw, ph, W, H};
}
function lines(id, series, o) {
  const f = frame(id, o);
  series.forEach(s => {
    const pts = s.points;
    const d = pts.map((p, i) => (i ? "L" : "M") + f.xs(p[0]).toFixed(1) + " " + f.ys(p[1]).toFixed(1)).join(" ");
    el("path", {d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round"}, f.plot);
    if (s.markers) pts.forEach(p => { const c = el("circle", {cx: f.xs(p[0]), cy: f.ys(p[1]), r: 4.5, fill: s.color, stroke: "#fcfcfb", "stroke-width": 2}, f.g); bindTip(c, `<b>${s.name}</b><br>${o.tipfmt ? o.tipfmt(p) : nf(p[0], 2) + " : " + nf(p[1], 2)}`); });
  });
  return f;
}
function bars(id, cats, series, o) {
  const f = frame(id, Object.assign({x0: 0, x1: 1, noX: true}, o));
  const band = f.pw / cats.length, inner = band * 0.72, bw = inner / series.length;
  cats.forEach((c, i) => {
    const x0 = f.m.l + i * band + (band - inner) / 2;
    series.forEach((s, j) => {
      const v = s.values[i], y = f.ys(Math.max(v, o.y0)), base = f.ys(o.y0);
      const r = el("rect", {x: x0 + j * bw + 1, y: Math.min(y, base), width: Math.max(bw - 2, 1), height: Math.abs(base - y), fill: s.color, rx: 3}, f.g);
      bindTip(r, `<b>${c}</b><br>${s.name}: ${(o.tipfmt || (v => nf(v, 2)))(v)}`);
      if (o.labels) el("text", {x: x0 + j * bw + bw / 2, y: y - 4, "text-anchor": "middle", "font-size": 10.5, fill: "#1d1d1b"}, f.g, (o.labfmt || (v => nf(v, 2)))(v));
    });
    el("text", {x: f.m.l + i * band + band / 2, y: f.H - f.m.b + 16, "text-anchor": "middle", "font-size": 11.5}, f.g, c);
  });
  return f;
}
function hbars(id, cats, vals, o) {
  const H = o.h || (cats.length * 30 + 30), W = o.w || 560, ml = o.ml || 230;
  const host = $(id); host.innerHTML = "";
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`}, host);
  const pw = W - ml - 60, xs = v => ml + v / o.max * pw;
  cats.forEach((c, i) => {
    const y = 10 + i * 30;
    el("text", {x: ml - 8, y: y + 15, "text-anchor": "end", "font-size": 12}, svg, c);
    const r = el("rect", {x: ml, y: y + 3, width: Math.max(xs(vals[i]) - ml, 1.5), height: 18, rx: 3, fill: o.colors ? o.colors[i] : "#2a78d6"}, svg);
    bindTip(r, `<b>${c}</b><br>${o.tipfmt ? o.tipfmt(vals[i]) : nf(vals[i], 4)}`);
    el("text", {x: xs(vals[i]) + 6, y: y + 16, "font-size": 11.5, fill: "#1d1d1b"}, svg, o.fmt ? o.fmt(vals[i]) : nf(vals[i], 4));
  });
}
function table(id, head, rows, hl) {
  $(id).innerHTML = `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) => `<tr class="${hl && hl(i) ? "hl" : ""}">${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function tiles(id, items) { $(id).innerHTML = items.map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join(""); }

// ---------- Tab 1: Data & Pipeline ----------
const ds = D.dataset;
tiles("data-tiles", [
  ["Ukuran data", nf(ds.rows, 0), `${ds.raw_cols} kolom mentah, ${ds.n_features} prediktor`],
  ["Parquet Snappy", nf(ds.parquet_mb, 2) + " MB", `dari ${nf(ds.csv_mb, 2)} MB CSV (hemat ${nf(ds.size_reduction, 2)}%)`],
  ["Baca + agregasi", nf(ds.io_speedup, 2) + "×", `median 3 kali: ${nf(ds.csv_s, 3)} s → ${nf(ds.parquet_s, 3)} s`],
  ["Split 80:20 (seed 42)", nf(ds.train_rows, 0), `latih; ${nf(ds.test_rows, 0)} uji`],
  ["Label positif", nf(ds.pos_pct, 2) + "%", `cart_abandonment_rate ≥ 40% (rata-rata ${nf(ds.mean_cart, 2)}%)`],
]);
$("flow").innerHTML = [
  ["1. Ingesti", "CSV 1 juta baris ke Spark"], ["2. Prapemrosesan", "label, fitur tanggal, Parquet Snappy"], ["3. Split", "80:20 sebelum fit apa pun"],
  ["4. Pipeline fitur", `fit hanya pada data latih (${ds.n_features} / ${ds.lr_dims} dimensi)`], ["5. Tuning", "72 konfigurasi DT pada data validasi"],
  ["6. Evaluasi", "data uji dipakai sekali"], ["7. Uji validitas", "kontrol negatif, sensitivitas ambang"], ["8. Interpretabilitas", "surrogate, atribusi, permutasi"],
].map(([a, b]) => `<div><b>${a}</b>${b}</div>`).join("");
lines("c-stress", [{name: "Cart abandonment (%)", color: "#2a78d6", markers: true, points: D.eda.stress}],
  {x0: 0, x1: 10, y0: 0, y1: 80, xticks: [0, 2, 4, 6, 8, 10], xfmt: v => v, yfmt: v => v + "%", xlab: "stress_from_financial_decisions", ylab: "Rata-rata cart abandonment",
   tipfmt: p => `skor ${p[0]}: ${nf(p[1], 2)}%`});
const corr = D.eda.corr, cKeys = Object.keys(corr).filter(k => k !== "checkout_abandonments_per_month");
hbars("c-corr", cKeys, cKeys.map(k => Math.abs(corr[k])), {max: 1, fmt: v => nf(v, 4), colors: cKeys.map((k, i) => i < 4 ? "#2a78d6" : "#b9b8b2")});
const P = {}; D.eda.persona.forEach(p => P[p.is_high_abandonment] = p);
legend("#lg-persona", [["Label tinggi (1)", "#eb6834"], ["Label rendah (0)", "#2a78d6"]]);
bars("c-persona", ["impulse_buying", "financial_stress", "overall_stress", "impulse_purchases"], [
  {name: "Label tinggi", color: "#eb6834", values: [P[1].avg_impulse_score, P[1].avg_financial_stress, P[1].avg_overall_stress, P[1].avg_impulse_purchases]},
  {name: "Label rendah", color: "#2a78d6", values: [P[0].avg_impulse_score, P[0].avg_financial_stress, P[0].avg_overall_stress, P[0].avg_impulse_purchases]},
], {y0: 0, y1: 10, labels: true, yfmt: v => nf(v, 0)});
table("t-device", ["Perangkat", "Porsi pengguna", "Rata-rata cart abandonment"], D.eda.devices.map(d => [d[0], nf(d[1], 2) + "%", nf(d[2], 2) + "%"]));

// ---------- Tab 2: Evaluasi Model ----------
const M = D.models, byKey = {}; M.forEach(m => byKey[m.key] = m);
const vc = D.validity;
const neg = vc["Negative_Control_Checkout_Abandonments_ge_5"].auc;
tiles("model-tiles", [
  ["ROC-AUC tertinggi", nf(byKey.LR_Tuned.roc, 4), `LR dan GBT (${nf(byKey.GBT_Tuned.roc, 4)})`],
  ["DT-d10 (model acuan)", nf(byKey.DT_Champion_d10.roc, 4), `${nf(byKey.DT_Champion_d10.time, 2)} s, ${byKey.DT_Champion_d10.nodes} simpul`],
  ["Kecepatan DT-d10", nf(byKey.GBT_Tuned.time / byKey.DT_Champion_d10.time, 1) + "×", `lebih cepat dari GBT; ${nf(byKey.RF_Tuned.time / byKey.DT_Champion_d10.time, 1)}× dari RF`],
  ["Kontrol negatif", nf(Math.min(...Object.values(neg)), 4) + " – " + nf(Math.max(...Object.values(neg)), 4), "ROC-AUC setara tebakan acak"],
]);
$("n-test").textContent = nf(D.confusion.total_test, 0);
table("t-models", ["Model", "Ukuran", "ROC-AUC [IK 95%]", "PR-AUC", "Akurasi (%)", "F1 (b)", "ROC-AUC CV (SB)", "Waktu latih (s)"],
  M.map(m => [`<span style="color:${C[m.key]}">●</span> ${m.label}`, m.size, `${nf(m.roc, 4)} [${nf(m.roc_lo, 4)}; ${nf(m.roc_hi, 4)}]`, nf(m.pr, 4), nf(m.acc, 2), nf(m.f1, 4), `${nf(m.cv, 4)} (${nf(m.cv_sd, 4)})`, nf(m.time, 2)]),
  i => M[i].key === "DT_Champion_d10");
legend(".lg-models", M.map(m => [m.label, C[m.key]]));
lines("c-roc", M.map(m => ({name: m.label, color: C[m.key], points: D.curves[m.key].roc})), {x0: 0, x1: 0.3, y0: 0.6, y1: 1, xticks: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], xfmt: v => nf(v, 2), xlab: "False positive rate", ylab: "True positive rate"});
lines("c-pr", M.map(m => ({name: m.label, color: C[m.key], points: D.curves[m.key].pr})), {x0: 0, x1: 1, y0: 0.8, y1: 1, xfmt: v => nf(v, 1), xlab: "Recall", ylab: "Precision"});
{
  const f = frame("c-tradeoff", {x0: 4, x1: 300, xlog: true, y0: 0.96, y1: 0.995, xfmt: v => v, yfmt: v => nf(v, 3), xlab: "Waktu latih (detik, log)", ylab: "ROC-AUC uji", m: {l: 58}});
  M.forEach(m => {
    const r = 5 + 2.2 * Math.log10(m.nodes);
    const c = el("circle", {cx: f.xs(m.time), cy: f.ys(m.roc), r, fill: C[m.key], "fill-opacity": 0.85, stroke: "#fcfcfb", "stroke-width": 2}, f.g);
    bindTip(c, `<b>${m.label}</b><br>ROC-AUC ${nf(m.roc, 4)}<br>${nf(m.time, 2)} s<br>${m.size}`);
    const below = m.key === "GBT_Tuned";
    el("text", {x: f.xs(m.time) + r + 4, y: f.ys(m.roc) + (below ? 14 : 4), "font-size": 11.5, fill: "#1d1d1b"}, f.g, m.label);
  });
}
{
  $("lg-cv").innerHTML = "<span style=\"--c:#1d1d1b\">Lingkaran penuh = ROC-AUC uji</span><span style=\"--c:#b9b8b2\">Kotak = rata-rata CV 5-fold</span>";
  const f = frame("c-cv", {x0: 0, x1: 1, noX: true, y0: 0.96, y1: 0.995, yfmt: v => nf(v, 3), ylab: "ROC-AUC"});
  const band = f.pw / M.length;
  M.forEach((m, i) => {
    const cx = f.m.l + band * i + band / 2;
    const lo = Math.min(...m.folds), hi = Math.max(...m.folds);
    el("line", {x1: cx + 8, x2: cx + 8, y1: f.ys(lo), y2: f.ys(hi), stroke: "#7a7974", "stroke-width": 2}, f.g);
    const cvp = el("rect", {x: cx + 3, y: f.ys(m.cv) - 5, width: 10, height: 10, fill: "#fcfcfb", stroke: C[m.key], "stroke-width": 2.5}, f.g);
    bindTip(cvp, `<b>${m.label}</b><br>CV ${nf(m.cv, 4)} (SB ${nf(m.cv_sd, 4)})<br>fold: ${m.folds.map(v => nf(v, 4)).join(", ")}`);
    const tp = el("circle", {cx: cx - 8, cy: f.ys(m.roc), r: 6, fill: C[m.key]}, f.g);
    bindTip(tp, `<b>${m.label}</b><br>Uji ${nf(m.roc, 4)}`);
    el("text", {x: cx, y: f.H - f.m.b + 16, "text-anchor": "middle", "font-size": 11.5}, f.g, m.label);
  });
}
{
  const c = D.confusion, tot = c.total_test;
  const cell = (v, strong) => `<div style="background:${strong ? "#2a78d6" : "#cde2fb"};color:${strong ? "#fff" : "#1d1d1b"}"><b>${nf(v, 0)}</b><br><small>${nf(v / tot * 100, 2)}%</small></div>`;
  $("cm").innerHTML = `<div class="cm"><div class="h"></div><div class="h">Prediksi 0</div><div class="h">Prediksi 1</div>
    <div class="h">Aktual 0</div>${cell(c.tn, true)}${cell(c.fp, false)}<div class="h">Aktual 1</div>${cell(c.fn, false)}${cell(c.tp, true)}</div>
    <p class="cap">Presisi kelas positif ${nf(c.precision_positive * 100, 2)}%, recall ${nf(c.recall_positive * 100, 2)}%, akurasi ${nf(c.accuracy_pct, 2)}%.</p>`;
}
{
  const L = k => (byKey[k] || {}).label || k;
  const boot = {}; D.bootstrap.forEach(b => boot[b.model_a + "|" + b.model_b] = b);
  const rows = D.mcnemar.filter(t => t.model_b === "DT_Champion_d10" || t.model_a === "DT_Champion_d10").map(t => {
    const b = boot[t.model_a + "|" + t.model_b] || {};
    const p = t.p_value_chi2_cc < 0.001 ? t.p_value_chi2_cc.toExponential(1).replace(".", ",") : nf(t.p_value_chi2_cc, 3);
    return [`${L(t.model_a)} vs ${L(t.model_b)}`, nf(t.a_correct_b_wrong_b, 0), nf(t.a_wrong_b_correct_c, 0), nf(t.chi2_continuity_corrected, 2), p,
      b.delta_roc_auc_mean !== undefined ? `${b.delta_roc_auc_mean > 0 ? "+" : ""}${nf(b.delta_roc_auc_mean, 5)} [${nf(b.delta_roc_auc_ci95_low, 5)}; ${nf(b.delta_roc_auc_ci95_high, 5)}]` : "–"];
  });
  table("t-pairs", ["A vs B", "b: A benar", "c: B benar", "χ²", "Nilai p", "ΔROC-AUC (A − B) [IK 95%]"], rows);
}
{
  const host = $("heat"); host.innerHTML = "";
  const depths = [...new Set(D.grid.map(g => g.depth))].sort((a, b) => a - b), mins = [...new Set(D.grid.map(g => g.min_inst))].sort((a, b) => a - b);
  const vals = D.grid.map(g => g.val_auc), lo = 0.96, hi = Math.max(...vals);
  const ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  const col = v => ramp[Math.min(ramp.length - 1, Math.max(0, Math.floor((v - lo) / (hi - lo + 1e-9) * ramp.length)))];
  ["gini", "entropy"].forEach(imp => {
    const div = document.createElement("div");
    let h = `<h3 style="font-size:13px;margin:0 0 6px">${imp === "gini" ? "Gini" : "Entropy"}</h3><table class="heat"><thead><tr><th>maxDepth</th>${mins.map(m => `<th>minInst ${m}</th>`).join("")}</tr></thead><tbody>`;
    depths.forEach(d => {
      h += `<tr><th>${d}</th>`;
      mins.forEach(mi => {
        const g = D.grid.find(x => x.depth === d && x.min_inst === mi && x.impurity === imp);
        const cls = [g.id === D.chosen ? "sel" : "", g.id === D.best_any ? "best" : "", ["#cde2fb", "#9ec5f4", "#6da7ec"].includes(col(g.val_auc)) ? "lt" : ""].join(" ");
        h += `<td class="${cls}" style="background:${col(g.val_auc)}" data-t="${g.id}: ROC-AUC ${nf(g.val_auc, 4)}, ${nf(g.nodes, 0)} simpul">${nf(g.val_auc, 4)}</td>`;
      });
      h += "</tr>";
    });
    div.innerHTML = h + "</tbody></table>";
    host.appendChild(div);
  });
  host.querySelectorAll("td[data-t]").forEach(td => bindTip(td, td.dataset.t));
}
table("t-ablation", ["Varian", "Depth", "minInst", "Impurity", "Simpul val → penuh", "ROC-AUC val", "ROC-AUC uji", "Akurasi uji (%)"],
  D.ablation.map(a => [a.id.split("_")[0], a.depth, a.min_inst, a.impurity, `${nf(a.val_nodes, 0)} → ${nf(a.nodes, 0)}`, nf(a.val_auc, 4), nf(a.test_auc, 4), nf(a.test_acc, 2)]),
  i => D.ablation[i].id.startsWith("M2"));
{
  const exps = [["Negative_Control_Checkout_Abandonments_ge_5", "Kontrol negatif"], ["Threshold_35pct", "Ambang 35%"], ["Threshold_40pct", "Ambang 40%"], ["Threshold_45pct", "Ambang 45%"]];
  const keys = ["LR_Tuned", "DT_Champion_d10", "RF_Tuned"];
  legend("#lg-valid", keys.map(k => [byKey[k].label, C[k]]));
  bars("c-valid", exps.map(e => e[1]), keys.map(k => ({name: byKey[k].label, color: C[k], values: exps.map(e => vc[e[0]].auc[k])})),
    {y0: 0, y1: 1, yfmt: v => nf(v, 1), tipfmt: v => nf(v, 4), ylab: "ROC-AUC uji"});
}
{
  const keys = [["DT_Champion_d10", "dt_median_s"], ["LR_Tuned", "lr_median_s"], ["GBT_Tuned", "gbt_median_s"], ["RF_Tuned", "rf_median_s"]];
  legend(".lg-scale", keys.map(([k]) => [byKey[k].label, C[k]]));
  lines("c-rows", keys.map(([k, col]) => ({name: byKey[k].label, color: C[k], markers: true, points: D.scaling_rows.map(r => [r.train_rows / 1000, r[col]])})),
    {x0: 150, x1: 820, y0: 2, y1: 200, ylog: true, xticks: [160, 320, 480, 640, 800], xfmt: v => v + "k", yfmt: v => v, xlab: "Baris data latih (ribu)", ylab: "Detik", tipfmt: p => `${nf(p[0] * 1000, 0)} baris: ${nf(p[1], 2)} s`});
  const cores = D.scaling_cores;
  bars("c-cores", cores.map(c => c.master), [{name: "Median waktu latih", color: "#2a78d6", values: cores.map(c => c.dt_fit_median_s)}],
    {y0: 0, y1: 20, labels: true, labfmt: v => nf(v, 2) + " s", ylab: "Detik", yfmt: v => v, tipfmt: v => nf(v, 2) + " s"});
}

// ---------- Tab 3: Interpretabilitas & Pengembangan ----------
const S = D.surrogate;
$("fid").textContent = nf(S.fidelity, 2) + "%"; $("sacc").textContent = nf(S.acc, 2) + "%";
const tier = i => S.tiers[i];
const riskColor = r => r >= 80 ? "#d03b3b" : r >= 20 ? "#ec835a" : "#0ca30c";
{
  const host = $("tree"); host.innerHTML = "";
  const svg = el("svg", {viewBox: "0 0 640 330", class: "tree"}, host);
  const box = (x, y, w, lines, fill, stroke) => { el("rect", {x: x - w / 2, y, width: w, height: 18 + 15 * lines.length, rx: 8, fill, stroke}, svg); lines.forEach((t, i) => el("text", {x, y: y + 18 + i * 15, "text-anchor": "middle", "font-size": 11.5, fill: "#1d1d1b", "font-weight": i ? 400 : 600}, svg, t)); };
  const edge = (x1, y1, x2, y2, lab) => { el("path", {d: `M${x1} ${y1} C${x1} ${(y1 + y2) / 2} ${x2} ${(y1 + y2) / 2} ${x2} ${y2}`, fill: "none", stroke: "#b9b8b2", "stroke-width": 1.5}, svg); el("text", {x: (x1 + x2) / 2 + (x2 > x1 ? 8 : -8), y: (y1 + y2) / 2 + 4, "text-anchor": x2 > x1 ? "start" : "end", "font-size": 11}, svg, lab); };
  const leafBox = (x, y, lines, rate) => box(x, y, 190, lines, riskColor(rate) + "22", riskColor(rate));
  edge(330, 43, 110, 110, "≤ 4,5"); edge(330, 43, 450, 110, "> 4,5");
  edge(450, 143, 320, 205, "≤ 5,5"); edge(450, 143, 545, 205, "> 5,5");
  edge(320, 238, 190, 285, "≤ 5,5"); edge(320, 238, 440, 285, "> 5,5");
  box(330, 10, 190, ["impulse_buying_score"], "#e6f0fb", "#2a78d6");
  leafBox(110, 110, ["Daun 1: prediksi 0", `${nf(tier(2).test_users, 0)} pengguna, ${nf(tier(2).high_abandon_rate_pct, 2)}% tinggi`], tier(2).high_abandon_rate_pct);
  box(450, 110, 190, ["impulse_buying_score"], "#e6f0fb", "#2a78d6");
  box(320, 205, 230, ["stress_from_financial_decisions"], "#e6f0fb", "#2a78d6");
  leafBox(545, 205, ["Daun 4: prediksi 1", "impulse > 5,5"], tier(0).high_abandon_rate_pct);
  leafBox(190, 285, ["Daun 2: prediksi 0", `${nf(tier(1).test_users, 0)} pengguna, ${nf(tier(1).high_abandon_rate_pct, 2)}% tinggi`], tier(1).high_abandon_rate_pct);
  leafBox(440, 285, ["Daun 3: prediksi 1", "zona ambang, stres tinggi"], tier(0).high_abandon_rate_pct);
  host.insertAdjacentHTML("beforeend", `<p class="cap">Daun 3 dan 4 sama-sama memprediksi 1 dan dihitung bersama: ${nf(tier(0).test_users, 0)} pengguna uji (${nf(tier(0).coverage_pct, 2)}%), ${nf(tier(0).high_abandon_rate_pct, 2)}% di antaranya berlabel tinggi. Warna daun menunjukkan proporsi label tinggi.</p>`);
}
function sim() {
  const imp = +$("s-imp").value, str = +$("s-str").value;
  $("v-imp").textContent = imp; $("v-str").textContent = str;
  let t, path, pred;
  if (imp <= 4.5) { t = tier(2); pred = 0; path = "impulse ≤ 4,5"; }
  else if (imp <= 5.5 && str <= 5.5) { t = tier(1); pred = 0; path = "4,5 < impulse ≤ 5,5 dan stres ≤ 5,5"; }
  else { t = tier(0); pred = 1; path = imp <= 5.5 ? "4,5 < impulse ≤ 5,5 dan stres > 5,5" : "impulse > 5,5"; }
  const col = riskColor(t.high_abandon_rate_pct);
  $("sim-out").style.borderColor = col;
  $("sim-out").innerHTML = `<span class="badge" style="background:${col};color:#fff">Prediksi ${pred} · ${pred ? "risiko tinggi" : t.high_abandon_rate_pct > 20 ? "zona ambang" : "risiko rendah"}</span>
    <p style="margin:10px 0 4px">Jalur: <b>${path}</b></p>
    <p style="margin:0">Pada data uji, segmen ini berisi <b>${nf(t.test_users, 0)}</b> pengguna (${nf(t.coverage_pct, 2)}%), dan <b>${nf(t.high_abandon_rate_pct, 2)}%</b> di antaranya berlabel tinggi. Rata-rata cart abandonment ${nf(t.avg_cart_abandonment_rate, 2)}%.</p>`;
}
$("s-imp").addEventListener("input", sim); $("s-str").addEventListener("input", sim); sim();
{
  const methods = [["dt_gini", "Gini DT-d10 (Spark)", "#2a78d6"], ["rf_gini", "Gini RF", "#eb6834"], ["lr_coef", "|koef. LR|", "#1baf7a"], ["shap_rf", "SHAP RF", "#eda100"]];
  legend("#lg-attr", methods.map(m => [m[1], m[2]]));
  bars("c-attr", ["impulse_buying", "financial_stress", "overall_stress", "impulse_purchases"], methods.map(([k, n, c]) => ({name: n, color: c, values: D.attribution.map(a => a[k])})),
    {y0: 0, y1: 100, yfmt: v => v + "%", tipfmt: v => nf(v, 2) + "%", ylab: "Porsi atribusi"});
}
{
  legend("#lg-joint", [["Sebelum", "#2a78d6"], ["Sesudah diacak bersama", "#b9b8b2"]]);
  bars("c-joint", D.joint_perm.map(j => j.model), [{name: "Sebelum", color: "#2a78d6", values: D.joint_perm.map(j => j.before)}, {name: "Sesudah", color: "#b9b8b2", values: D.joint_perm.map(j => j.after)}],
    {y0: 0, y1: 1, labels: true, labfmt: v => nf(v, 3), yfmt: v => nf(v, 1), tipfmt: v => nf(v, 4), ylab: "ROC-AUC"});
}

// ---------- Tab switching ----------
const tabs = document.querySelectorAll('[role="tab"]');
function openTab(id) {
  tabs.forEach(t => { const on = t.id === id; t.setAttribute("aria-selected", on); $(t.getAttribute("aria-controls")).classList.toggle("active", on); });
  history.replaceState(null, "", "#" + id.slice(2));
}
tabs.forEach(t => t.addEventListener("click", () => openTab(t.id)));
if (location.hash && $("t-" + location.hash.slice(1))) openTab("t-" + location.hash.slice(1));
</script>
</body>
</html>
"""


def build_dashboard(out_path: Path = OUT_PATH) -> Path:
    data = collect_data()
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = build_dashboard()
    print(f"[SAVED] {path.relative_to(BASE_DIR)}")
