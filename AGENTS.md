# AGENTS.md — Collaboration & AI Agent Guidelines

Dokumen ini adalah panduan operasional utama untuk seluruh AI coding agent (Cursor, Antigravity, Claude Code, GitHub Copilot) dan kontributor manusia yang bekerja di repositori ini.

---

## 1. Konteks & Tujuan Proyek
- **Mata Kuliah**: Teknologi Big Data (Semester 5)
- **Kelompok**: Anggrek Mekar Cilibende (Galih, Trias, Risha, Raihan)
- **Topik**: Menganalisis faktor-faktor pendorong pembatalan transaksi di tahap akhir (*cart/checkout abandonment*) pada platform e-commerce (skala 1.000.000 baris data) untuk menemukan pola perilaku pengguna dan merancang strategi intervensi bisnis.
- **Metode Pembelajaran Mesin Utama**: **Decision Tree Classifier** (lengkap dengan *Hyperparameter Tuning*, *Tree Pruning*, *Feature Importance*, dan ekstraksi *IF-THEN Business Rules*).
- **Target Deliverables**:
  1. Pipeline Big Data PySpark terdistribusi.
  2. Model Decision Tree terlatih & terevaluasi.
  3. Dashboard Visualisasi Interaktif (BI/Streamlit).
  4. Laporan Akhir & Slide Presentasi Showcase.
  5. **Draft Jurnal Ilmiah (Manuskrip Publikasi)**.
- **Dataset**: `dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based` (1 juta baris, 57 kolom).
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
- Data berjumlah 1.000.000 baris. Gunakan **PySpark** (`pyspark.sql.DataFrame`) atau **DuckDB** untuk operasi agregasi, filtering, dan feature engineering skala besar.
- Hindari memuat seluruh CSV mentah ke Pandas DataFrame jika melakukan operasi berat yang berisiko *Out of Memory* (OOM). Simpan hasil olahan dalam format **Parquet** di `data/processed/`.

### Rule 3: Integritas Target Variabel & Metrik
- Target utama: Tingkat pembatalan keranjang/checkout (`checkout_abandonments_per_month` atau `cart_abandonment_rate`).
- Untuk klasifikasi: Binerkan target (High vs Low Abandonment) sesuai threshold yang tercatat di `docs/02_DATA_PIPELINE_LOG.md`. Jangan mengubah definisi target secara sepihak.
- Metrik evaluasi utama: **ROC-AUC** dan **F1-Score** (akurasi biasa tidak cukup karena potensi class imbalance).

### Rule 4: Model Explainability & Segmentasi
- Proyek ini **bukan hanya melatih model hitam (black box)**.
- Wajib menyertakan analisis interpretabilitas menggunakan **SHAP (SHapley Additive exPlanations)** untuk membuktikan faktor pendorong pembatalan.
- Wajib menyertakan segmentasi pengguna (**K-Means Clustering**) untuk menghasilkan profil persona belanja.

### Rule 5: Protokol Git & Keamanan Data
- **DILARANG AUTO-PUSH**: Jangan melakukan `git push` ke branch `main` tanpa konfirmasi eksplisit dari pemilik repositori.
- **DILARANG COMMIT DATA BESAR & KREDENSIAL**: File dataset mentah di `data/raw/*.csv` dan file kredensial (`kaggle.json`, `.env`) tidak boleh masuk ke riwayat Git.
- Gunakan konvensi commit yang jelas: `feat:`, `fix:`, `docs:`, `refactor:`.

---

## 3. Struktur Modul & Pembagian Tugas
- `src/download_data.py`: Skrip otomatisasi unduh dataset via Kaggle API / Kagglehub.
- `src/spark_session.py`: Inisialisasi SparkSession dengan alokasi memori yang optimal.
- `src/preprocessing.py`: Pipeline transformasi fitur terdistribusi.
- `src/train_classifier.py`: Skrip pelatihan model klasifikasi prediktif.
- `src/clustering.py`: Skrip K-Means clustering untuk segmentasi pengguna.
- `notebooks/`: Khusus untuk eksplorasi visual (EDA) dan pembuatan grafik laporan.
- `reports/`: Folder keluaran plot grafik, confusion matrix, dan SHAP feature importance.
