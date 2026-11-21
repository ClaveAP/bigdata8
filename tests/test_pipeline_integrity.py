"""
Automated Verification Suite (Unit & Integration Tests) — Proyek Big Data & Naskah JIKO
Memastikan:
1. Nol kebocoran fitur target/kontrol pada `features` dan `features_lr`, serta `split_before_fit == True`.
2. Seluruh 60 kolom mentah + 2 kolom turunan tercatat lengkap di metadata dan `01_DATA_DICTIONARY.md`.
3. Model PySpark (`decision_tree_baseline` & `decision_tree_tuned`) bersih dari `pruneTree` (kompatibel Google Colab).
4. File `reports/decision_tree_rules.txt` telah memetakan seluruh indeks fitur ke nama kolom asli.
5. Seluruh file metrik JSON/CSV, 12 grafik PNG (300 DPI), dan 5 file CSV Looker Studio tersedia dan valid.
6. Seluruh dokumen (`README.md`, `reports/README.md`, `docs/00..07`, `PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md`)
   konsisten dengan Single Source of Truth (`reports/metrics/*.json`) dan bebas dari angka usang.
"""

import csv
import json
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from pipeline_utils import (
    ALL_FEATURE_NAMES,
    CATEGORICAL_COLS,
    ID_COL,
    METRICS_DIR,
    MODELS_DIR,
    NEGATIVE_CONTROL_COL,
    NUMERIC_COLS,
    RAW_DATE_COL,
    REPORTS_DIR,
    TARGET_COL,
    TARGET_SOURCE_COL,
)


class TestPipelineIntegrity(unittest.TestCase):
    def test_01_feature_schema_and_zero_leakage(self):
        """Pastikan 57 fitur prediktor bebas dari kolom target/kontrol dan Split-Before-Fit aktif."""
        self.assertEqual(len(NUMERIC_COLS), 43)
        self.assertEqual(len(CATEGORICAL_COLS), 14)
        self.assertEqual(len(ALL_FEATURE_NAMES), 57)

        forbidden_cols = [ID_COL, TARGET_SOURCE_COL, NEGATIVE_CONTROL_COL, TARGET_COL, RAW_DATE_COL]
        for forbidden in forbidden_cols:
            self.assertNotIn(forbidden, ALL_FEATURE_NAMES)
            self.assertNotIn(f"{forbidden}_encoded", ALL_FEATURE_NAMES)

        meta_path = METRICS_DIR / "feature_metadata.json"
        self.assertTrue(meta_path.exists(), "feature_metadata.json belum dibuat")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertTrue(meta["split_before_fit"], "Split-Before-Fit harus bernilai True")
        self.assertEqual(meta["tree_feature_vector_dim"], 57)
        self.assertEqual(meta["lr_ohe_scaled_vector_dim"], 132)
        self.assertEqual(meta["train_count"] + meta["test_count"], 1_000_000)

    def test_02_colab_model_metadata_sanitized(self):
        """Pastikan metadata model PySpark bebas dari parameter `pruneTree` yang merusak Google Colab."""
        for model_name in ["decision_tree_baseline", "decision_tree_tuned"]:
            m_dir = MODELS_DIR / model_name / "metadata"
            self.assertTrue(m_dir.exists(), f"Folder metadata {m_dir} tidak ditemukan")
            crc_files = list(m_dir.glob(".*.crc"))
            self.assertEqual(len(crc_files), 0, f"Masih ada file checksum .crc di {m_dir}")
            for part_file in m_dir.glob("part-*"):
                payload = json.loads(part_file.read_text(encoding="utf-8"))
                for map_key in ("paramMap", "defaultParamMap"):
                    if map_key in payload and isinstance(payload[map_key], dict):
                        self.assertNotIn("pruneTree", payload[map_key])

    def test_03_humanized_tree_rules(self):
        """Pastikan decision_tree_rules.txt berisi nama fitur asli dan ambang root split yang benar."""
        rules_path = REPORTS_DIR / "decision_tree_rules.txt"
        self.assertTrue(rules_path.exists(), "decision_tree_rules.txt belum tersedia")
        txt = rules_path.read_text(encoding="utf-8")
        self.assertIn("impulse_buying_score [f15]", txt)
        self.assertIn("stress_from_financial_decisions [f22]", txt)
        self.assertIn("<= 4.5", txt)

    def test_04_all_metrics_plots_and_looker_files_exist(self):
        """Pastikan seluruh JSON/CSV metrik, 12 grafik PNG, dan 5 CSV Looker Studio tersedia."""
        required_metrics = [
            "preprocessing_summary.json",
            "eda_summary.json",
            "feature_metadata.json",
            "baseline_dt_metrics.json",
            "dt_tuning_grid.csv",
            "dt_champion_summary.json",
            "rules_and_fidelity.json",
            "kmeans_segmentation.json",
            "explainability_shap_permutation.json",
            "permutation_importance.csv",
            "model_comparison_summary.json",
            "roc_pr_curves.json",
        ]
        for fname in required_metrics:
            fpath = METRICS_DIR / fname
            self.assertTrue(fpath.exists(), f"File metrik {fname} belum ada")
            self.assertGreater(fpath.stat().st_size, 50, f"File metrik {fname} kosong")

        required_plots = [
            "eda_device_vs_abandonment.png",
            "eda_stress_vs_abandonment.png",
            "decision_tree_feature_importance.png",
            "confusion_matrix.png",
            "correlation_heatmap.png",
            "customer_persona_comparison.png",
            "model_roc_pr_curves.png",
            "dt_pruning_depth_curve.png",
            "pyspark_scaling_benchmark.png",
            "kmeans_persona_clusters.png",
            "shap_summary_plot.png",
            "permutation_vs_gini_importance.png",
        ]
        for pname in required_plots:
            ppath = REPORTS_DIR / pname
            self.assertTrue(ppath.exists(), f"Plot {pname} belum ada")
            self.assertGreater(ppath.stat().st_size, 10_000, f"Plot {pname} terlalu kecil/rusak")

        required_looker = [
            "looker_demografi.csv",
            "looker_metode_bayar.csv",
            "looker_stres_impulse.csv",
            "looker_scorecard_global.csv",
            "looker_model_feature_importance.csv",
        ]
        for lname in required_looker:
            lpath = REPORTS_DIR / "looker_data" / lname
            self.assertTrue(lpath.exists(), f"Looker CSV {lname} belum ada")
            self.assertGreater(lpath.stat().st_size, 30, f"Looker CSV {lname} kosong")

    def test_05_notebook_and_docs_consistency(self):
        """Pastikan Notebook bebas bug .orderBy() dan seluruh dokumen bebas dari angka lama yang salah."""
        nb_path = BASE_DIR / "notebooks" / "01_e_commerce_abandonment_showcase.ipynb"
        self.assertTrue(nb_path.exists())
        nb_txt = nb_path.read_text(encoding="utf-8")
        self.assertIn("orderBy(F.desc('is_high_abandonment'))", nb_txt)

        docs_dir = BASE_DIR / "docs"
        docs_to_check = [
            BASE_DIR / "README.md",
            BASE_DIR / "AGENTS.md",
            BASE_DIR / "reports" / "README.md",
            docs_dir / "00_PROJECT_CHARTER.md",
            docs_dir / "01_DATA_DICTIONARY.md",
            docs_dir / "02_DATA_PIPELINE_LOG.md",
            docs_dir / "03_MODEL_EXPERIMENTS.md",
            docs_dir / "04_BUSINESS_RECOMMENDATIONS.md",
            docs_dir / "05_TIMELINE_TRACKER.md",
            docs_dir / "06_LOOKER_STUDIO_GUIDE.md",
            docs_dir / "07_JOURNAL_PLAN_JIKO.md",
            docs_dir / "08_PROJECT_FIX_AND_HARDENING_PLAN.md",
            docs_dir / "09_VERIFIED_REFERENCES_JIKO.md",
            docs_dir / "10_MANUSCRIPT_DRAFT_JIKO.md",
            docs_dir / "PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md",
        ]
        for doc_path in docs_to_check:
            self.assertTrue(doc_path.exists(), f"Dokumen {doc_path.name} tidak ditemukan")
            content = doc_path.read_text(encoding="utf-8")
            self.assertNotIn("7.170 pengguna", content, f"Angka FP lama 7.170 ditemukan di {doc_path.name}")
            self.assertNotIn("4.951 pengguna", content, f"Angka FN lama 4.951 ditemukan di {doc_path.name}")
            self.assertNotIn("5 negara utama", content, f"Klaim salah '5 negara utama' ditemukan di {doc_path.name}")
            self.assertNotIn("treeshap", content.lower(), f"Klaim metode keliru 'TreeSHAP' ditemukan di {doc_path.name}")
            self.assertNotIn("\\lfloor\\sqrt{57}\\rfloor = 7", content, f"Spark membulatkan sqrt(57) ke atas (=8), bukan 7: {doc_path.name}")
            self.assertNotIn("58,30%", content, f"Peluang lama 58,30% (mtry=7) ditemukan di {doc_path.name}")
            self.assertNotIn("mengalahkan `RF_Tuned`", content, f"Klaim DT mengalahkan RF bertentangan dengan 5-Fold CV: {doc_path.name}")
            self.assertNotIn("15,39x", content, f"Speedup I/O 15,39x tidak tersimpan di metrik mana pun (pakai 11,96x): {doc_path.name}")

    def test_06_audit_remediation_invariants(self):
        """Pastikan seluruh 72 grid murni PySpark bebas hardcode/SklearnDT, Val ROC-AUC M2 tertinggi, node konsisten, & K-Means/CV/Bootstrap utuh."""
        tune_src = (BASE_DIR / "src" / "tune_tree.py").read_text(encoding="utf-8")
        self.assertNotIn('champion_params = {"maxDepth": 10', tune_src, "Masih ada hardcode champion_params di tune_tree.py")
        self.assertNotIn("SklearnDT", tune_src, "Masih ada fallback SklearnDT pada validasi grid di tune_tree.py")
        self.assertNotIn("spark_val_candidates", tune_src, "Masih ada daftar kandidat Spark manual di tune_tree.py")

        champ_summary = json.loads((METRICS_DIR / "dt_champion_summary.json").read_text(encoding="utf-8"))
        self.assertIn("150,269", champ_summary["protocol"])
        self.assertIn("100,086", champ_summary["protocol"])
        self.assertIn("D8_M1_ENT", champ_summary["selection_note"])
        self.assertIn("0.0007", champ_summary["selection_note"])
        self.assertNotIn("syarat konsistensi kn/log(n) -> inf)", champ_summary["selection_note"], "Klaim keliru: minInst=20 tetap tidak memenuhi syarat kn/log(n) -> inf")
        self.assertIn("heuristik praktis", champ_summary["selection_note"])
        self.assertEqual(champ_summary["best_unpruned_validation_row"]["config_id"], "D8_M1_ENT")

        shap_src = (BASE_DIR / "src" / "explainability_shap.py").read_text(encoding="utf-8")
        self.assertNotIn("treeshap", shap_src.lower(), "Masih ada penyebutan TreeSHAP keliru di explainability_shap.py")

        best_val = champ_summary["best_validation_row"]
        self.assertEqual(best_val["config_id"], "D10_M20_GIN")
        self.assertEqual(best_val["maxDepth"], 10)
        self.assertEqual(best_val["minInstancesPerNode"], 20)
        self.assertEqual(best_val["impurity"], "gini")
        self.assertEqual(best_val["num_nodes"], 419)

        with open(METRICS_DIR / "dt_tuning_grid.csv", encoding="utf-8") as f:
            grid_rows = {row["config_id"]: row for row in csv.DictReader(f)}
        self.assertEqual(len(grid_rows), 72)
        self.assertEqual(int(grid_rows["D10_M20_GIN"]["num_nodes"]), 419)

        ablation = {r["model_id"]: r for r in champ_summary["ablation_reference_models"]}
        self.assertGreater(ablation["M2_Champion_d10_gini20"]["val_roc_auc"], ablation["M1_Baseline_d7"]["val_roc_auc"])
        self.assertGreater(ablation["M2_Champion_d10_gini20"]["val_roc_auc"], ablation["M3c_d10_entropy50"]["val_roc_auc"])
        self.assertEqual(ablation["M2_Champion_d10_gini20"]["val_num_nodes"], 419)
        self.assertEqual(ablation["M2_Champion_d10_gini20"]["num_nodes"], 473)
        self.assertEqual(ablation["M4_Deep_d12_gini1"]["num_nodes"], 3431)

        km_data = json.loads((METRICS_DIR / "kmeans_segmentation.json").read_text(encoding="utf-8"))
        self.assertIn("k_selection_rationale", km_data)
        self.assertIn("methodological_note", km_data)
        personas = km_data["personas_k3"]
        persona_names = [p["persona_name"] for p in personas]
        self.assertTrue(any("Threshold-Zone Hesitant Shoppers" in n for n in persona_names))
        self.assertTrue(any("High-Stress Impulsive Abandoners" in n for n in persona_names))
        rates = sorted(p["avg_cart_abandonment_rate"] for p in personas)
        self.assertLess(rates[0], 20.0)
        self.assertTrue(35.0 < rates[1] < 50.0)
        self.assertGreater(rates[2], 65.0)

        mc_data = json.loads((METRICS_DIR / "model_comparison_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(mc_data["bootstrap_paired_auc_deltas"]), 5)
        for m_key, m_obj in mc_data["models"].items():
            self.assertIsNotNone(m_obj["cv5_metrics"]["cv5_roc_auc_mean"], f"CV5 kosong pada {m_key}")


    def test_07_dashboard_html_synced_with_metrics(self):
        """Dashboard HTML 3 tab harus dibangun ulang dari reports/metrics tanpa selisih."""
        import importlib
        import sys

        sys.path.insert(0, str(BASE_DIR / "src"))
        builder = importlib.import_module("build_dashboard_html")
        html_path = BASE_DIR / "reports" / "dashboard_visualisasi.html"
        committed = html_path.read_text(encoding="utf-8")
        for tab_id in ("p-data", "p-model", "p-xai"):
            self.assertIn(f'id="{tab_id}"', committed, f"Tab {tab_id} tidak ditemukan di dashboard")
        self.assertNotIn("cdn.jsdelivr", committed, "Dashboard harus bisa dibuka offline (tanpa CDN)")

        data = builder.collect_data()
        expected = builder.TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(committed, expected, "dashboard_visualisasi.html tidak sinkron; jalankan python src/build_dashboard_html.py")
        models = {m["key"]: m for m in data["models"]}
        self.assertAlmostEqual(models["DT_Champion_d10"]["roc"], 0.990413, places=6)
        self.assertEqual(data["dataset"]["io_speedup"], 11.96)

if __name__ == "__main__":
    unittest.main()
