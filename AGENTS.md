# AGENTS.md — Collaboration & AI Agent Guidelines

Dokumen ini adalah panduan operasional utama untuk seluruh AI coding agent (Cursor, Antigravity, Claude Code, GitHub Copilot) dan kontributor manusia yang bekerja di repositori ini.

---

## 1. Konteks & Tujuan Proyek
- **Mata Kuliah**: Teknologi Big Data (Semester 5)
- **Kelompok**: Kelompok 8 — Anggrek Mekar Cilibende (Galih, Trias, Risha, Raihan)
- **Topik**: Menganalisis faktor-faktor pendorong pembatalan transaksi di tahap akhir (*cart/checkout abandonment*) pada platform e-commerce (skala `1.000.000` baris data) untuk menemukan pola perilaku pengguna, mengevaluasi trade-off model klasifikasi terdistribusi, dan merancang strategi intervensi bisnis.
- **Metode Pembelajaran Mesin Utama**: **Decision Tree Classifier** (lengkap dengan *Validation-Based Hyperparameter Tuning*, *Tree Pruning Ablation*, *Multi-Model Benchmark*, *Feature Importance*, *SHAP & Permutation Importance*, *K-Means Clustering*, dan ekstraksi *IF-THEN Business Rules*).
- **Target Deliverables**:
  1. Pipeline Big Data PySpark terdistribusi (`Split-Before-Fit`, Snappy Parquet).
  2. Model Decision Tree terlatih & terevaluasi (`ROC-AUC 0,9904`, `Akurasi 93,99%`, Surrogate 4-daun `98,28%` fidelitas).
  3. Dashboard Visualisasi Interaktif (Looker Studio Data Marts & HTML Dashboard).
  4. Laporan Akhir & Slide Presentasi Showcase.
  5. **Draft Jurnal Ilmiah (Manuskrip Publikasi JIKO Sinta 4)**.
- **Dataset**: `dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based` (`1.000.000` baris, `60` kolom mentah $\rightarrow$ `62` kolom terproses di Parquet $\rightarrow$ `57` fitur prediktor independen: `43` numerik + `14` kategorikal).
- **Timeline Tracker**: Mengacu pada [Google Sheets](https://docs.google.com/spreadsheets/d/199qGgqLir1qj1HC9fz9oXuhW5leLQrQtXzch0n_XTgg/edit?gid=0#gid=0) dan tercermin di `docs/05_TIMELINE_TRACKER.md`.

---

## 2. Aturan Emas (Golden Rules)

### Rule 1: Living Documentation First
Setiap kali melakukan perubahan kode, pembersihan data, atau pelatihan model, **AI/kontributor WAJIB menyinkronkan dokumentasi terkait di `docs/`**:
- Tambah/hapus/ubah fitur $\rightarrow$ update `docs/01_DATA_DICTIONARY.md`.
- Transformasi data / pipeline logic $\rightarrow$ update `docs/02_DATA_PIPELINE_LOG.md`.
- Eksperimen model (metrik ROC-AUC, F1, training time) $\rightarrow$ update `docs/03_MODEL_EXPERIMENTS.md`.
- Insight baru / rekomendasi strategi $\rightarrow$ update `docs/04_BUSINESS_RECOMMENDATIONS.md`.

### Rule 2: Standar Big Data (PySpark Local Mode)
- Data berjumlah `1.000.000` baris. Gunakan **PySpark** (`pyspark.sql.DataFrame`) untuk operasi agregasi, filtering, dan feature engineering skala besar.
- Hindari memuat seluruh CSV mentah ke Pandas DataFrame jika melakukan operasi berat yang berisiko *Out of Memory* (OOM). Simpan hasil olahan dalam format **Parquet** di `data/processed/`.

### Rule 3: Integritas Target Variabel & Metrik
- Target utama: Tingkat pembatalan keranjang (`cart_abandonment_rate >= 40,0%` $\rightarrow$ `is_high_abandonment = 1`), sedangkan `checkout_abandonments_per_month` dipertahankan sebagai **kontrol negatif bebas bocor** ($r = -0,0009$, $\text{AUC} \approx 0,50$).
- Untuk klasifikasi: Binerkan target (High vs Low Abandonment) sesuai threshold yang tercatat di `docs/02_DATA_PIPELINE_LOG.md`. Jangan mengubah definisi target secara sepihak.
- Metrik evaluasi utama: **ROC-AUC**, **PR-AUC**, dan **F1-Score** (dilengkapi uji statistik McNemar dan Bootstrap 95% CI).

### Rule 4: Model Explainability & Segmentasi
- Proyek ini **bukan hanya melatih model hitam (black box)**.
- Wajib menyertakan analisis interpretabilitas menggunakan **SHAP (SHapley Additive exPlanations)** dan **Permutation Importance** (`src/explainability_shap.py`) untuk membuktikan faktor pendorong pembatalan serta mengaudit efek kolinearitas.
- Wajib menyertakan segmentasi pengguna (**K-Means Clustering** pada `src/clustering.py`) untuk menghasilkan profil persona belanja.

### Rule 5: Protokol Git & Keamanan Data
- **DILARANG AUTO-PUSH**: Jangan melakukan `git push` ke branch `main` tanpa konfirmasi eksplisit dari pemilik repositori.
- **DILARANG COMMIT DATA BESAR & KREDENSIAL**: File dataset mentah di `data/raw/*.csv` dan file kredensial (`kaggle.json`, `.env`) tidak boleh masuk ke riwayat Git.
- Gunakan konvensi commit yang jelas: `feat:`, `fix:`, `docs:`, `refactor:`.

### Rule 6: Integritas Sitasi & Penyusunan Manuskrip Jurnal (JIKO Sinta 4)
Saat menyusun, mencari referensi, atau meninjau draf jurnal ilmiah pada repositori ini ([`docs/07_JOURNAL_PLAN_JIKO.md`](docs/07_JOURNAL_PLAN_JIKO.md)):
- **DILARANG MENYITIR DARI ABSTRAK SAJA**: Setiap referensi yang dikutip **WAJIB** dibaca dan diverifikasi dari isi penuh artikel (*full-text*: Pendahuluan, Metode, Hasil, atau Pembahasan). Dilarang keras mengutip klaim, angka, atau metode hanya berdasarkan ringkasan abstrak.
- **WAJIB JURNAL SINTA / TERAKREDITASI BEREPUTASI**: Seluruh rujukan literatur **WAJIB** bersumber dari jurnal nasional terakreditasi **Sinta (Sinta 1–6, diutamakan pula menyitir artikel relevan dari JIKO)** atau jurnal internasional terakreditasi/bereputasi (**Scopus / Web of Science**). Jangan mengutip blog, laporan tidak terverifikasi, atau *preprint* tanpa tinjauan sejawat.
- **KETERBARUAN & VERIFIKASI DOI**: Gunakan minimal **15 referensi** format **IEEE** yang terbit **kurang dari 4 tahun terakhir (2022–2026)** lengkap dengan **DOI valid terverifikasi**, dan pastikan 100% daftar pustaka benar-benar disitir di dalam teks naskah.
- **FOKUS NARASI (ARAHAN DOSEN PEMBIMBING — IBU GEMA)**: Karena dataset Kaggle bersifat statis/sintetis, **JANGAN** membahas hasil sebagai temuan perilaku konsumen dunia nyata. Fokuskan isi jurnal pada **proses rekayasa pipeline Big Data PySpark (*Split-Before-Fit*, Snappy Parquet) serta perbandingan dan interpretabilitas model (*trade-off* LR vs GBT vs DT vs RF, *pre-pruning*, fidelitas aturan 4 daun, dan *Greedy Collinear Masking*)** agar dapat menjadi rujukan eksperimental bagi penelitian Big Data serupa.
- **KEPATUHAN FORMAT JIKO**: Bahasa Indonesia baku, `5.000–8.500` kata (`<= 16` halaman A4, spasi `1,5`), judul `<= 12` kata *Title Case*, abstrak `150–250` kata, kata kunci `<= 5`, Pendahuluan tanpa sub-bab (`2–3` halaman), Hasil murni objektif terpisah dari Pembahasan kritis, Kesimpulan **1 paragraf utuh tanpa daftar berpoin/bernomor**, serta skor Turnitin AI `< 20%`.

---

## 3. Struktur Modul & Pembagian Tugas
- `src/download_data.py`: Skrip otomatisasi unduh dataset via Kaggle API / Kagglehub.
- `src/spark_session.py`: Inisialisasi SparkSession dengan alokasi memori yang optimal.
- `src/pipeline_utils.py`: Konstanta skema terpusat, sanitasi metadata model Spark (`pruneTree`), dan utilitas I/O JSON metrik.
- `src/preprocessing.py`: Pipeline pembersihan data, derivasi `days_since_last_purchase`, binarisasi target, dan ekspor Snappy Parquet.
- `src/eda_profiling.py`: Audit skema & korelasi 60 kolom mentah serta pembuatan grafik EDA.
- `src/feature_engineering.py`: Pipeline transformasi fitur terdistribusi (*Split-Before-Fit* `seed=42`, `57` fitur pohon & `132` dimensi OHE+Scaled LR).
- `src/train_classifier.py`: Skrip pelatihan model baseline Decision Tree (`maxDepth=5`).
- `src/tune_tree.py`: Skrip 72-konfigurasi grid tuning pada Validation Split & ablasi terkontrol (`M0..M4`).
- `src/extract_rules_insights.py`: Penerjemahan aturan IF-THEN ke nama kolom asli & analisis *surrogate fidelity*.
- `src/clustering.py`: Skrip PySpark K-Means clustering (`K=2..5`) untuk segmentasi persona pengguna.
- `src/explainability_shap.py`: Analisis TreeSHAP, Permutation Importance lintas model, & *joint collinear cluster permutation*.
- `src/compare_models.py`: Benchmark komparatif 6 model (`LR`, `GBT`, `DT_d10`, `RF`, `DT_d7`, `DT_d3`), uji McNemar, Bootstrap CI, sensitivitas threshold, kontrol negatif, & skalabilitas komputasi PySpark.
- `src/export_looker_data.py`: Generator 5 berkas CSV Data Mart Looker Studio.
- `src/export_report_plots.py`: Generator grafik laporan 300 DPI & dashboard HTML mandiri.
- `src/generate_showcase_notebook.py`: Generator notebook interaktif Google Colab / VS Code (`notebooks/01_e_commerce_abandonment_showcase.ipynb`).
- `tests/test_pipeline_integrity.py`: Suite pengujian otomatis 5 lapis (anti-leakage, skema, fidelitas aturan, kompatibilitas Colab, konsistensi dokumentasi).
