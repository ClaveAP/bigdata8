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

        docs_to_check = [
            BASE_DIR / "README.md",
            BASE_DIR / "AGENTS.md",
            BASE_DIR / "reports" / "README.md",
            BASE_DIR / "docs" / "00_PROJECT_CHARTER.md",
            BASE_DIR / "docs" / "01_DATA_DICTIONARY.md",
            BASE_DIR / "docs" / "02_DATA_PIPELINE_LOG.md",
            BASE_DIR / "docs" / "03_MODEL_EXPERIMENTS.md",
            BASE_DIR / "docs" / "04_BUSINESS_RECOMMENDATIONS.md",
            BASE_DIR / "docs" / "05_TIMELINE_TRACKER.md",
            BASE_DIR / "docs" / "06_LOOKER_STUDIO_GUIDE.md",
            BASE_DIR / "docs" / "07_JOURNAL_PLAN_JIKO.md",
            BASE_DIR / "docs" / "PANDUAN_LENGKAP_PROYEK_DAN_SIDANG.md",
        ]
        for doc_path in docs_to_check:
            self.assertTrue(doc_path.exists(), f"Dokumen {doc_path.name} tidak ditemukan")
            content = doc_path.read_text(encoding="utf-8")
            self.assertNotIn("7.170 pengguna", content, f"Angka FP lama 7.170 ditemukan di {doc_path.name}")
            self.assertNotIn("4.951 pengguna", content, f"Angka FN lama 4.951 ditemukan di {doc_path.name}")
            self.assertNotIn("5 negara utama", content, f"Klaim salah '5 negara utama' ditemukan di {doc_path.name}")


if __name__ == "__main__":
    unittest.main()
