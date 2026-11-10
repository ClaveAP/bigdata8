# Rencana, Brief Tim, & Blueprint Final Naskah Jurnal JIKO (Sinta 4) (`07_JOURNAL_PLAN_JIKO.md`)

> **Status Eksperimen & Kode (`P1`)**: **100% SELESAI DIEKSEKUSI & TERVERIFIKASI**  
> Dokumen ini berfungsi sebagai **brief bersama Kelompok 8 (Galih, Trias, Risha, Raihan)** sekaligus **panduan eksekusi wajib** saat menulis naskah penuh di sesi penulisan terpisah (*Argon*).  
> Seluruh angka eksperimen (`E0–E9`) dan grafik 300 DPI sudah terverifikasi di `reports/metrics/*.json`.

---

## 1. Arah Utama Naskah & Arahan Dosen Pembimbing (Ibu Gema)

Sesuai hasil konsultasi dengan dosen (**Ibu Gema**), karena dataset Kaggle yang digunakan (`1.000.000` baris $\times$ `60` kolom) bersifat **statis dan sintetis**, arah penulisan jurnal **HARUS berfokus pada proses pemodelan dan perbandingan metode**, bukan mengklaim temuan perilaku belanja konsumen di dunia nyata.

### Panduan Fokus Penulisan untuk Seluruh Anggota Tim (*Do vs Don't*)

| Aspek | ❌ YANG HARUS DIHINDARI (*Don't*) | ✅ YANG HARUS DIBAHAS (*Do* — Arahan Bu Gema) |
| :--- | :--- | :--- |
| **Sudut Pandang Utama** | Membahas seolah-olah angka Kaggle mencerminkan psikologi pembeli Amazon/Shopify asli. | Membahas **desain pipeline Big Data terdistribusi (PySpark)** dan **perbandingan objektif antar-algoritma** sebagai **rujukan metodologis** bagi peneliti lain yang menguji dataset tabular statis skala jutaan baris serupa. |
| **Interpretasi Skor `~0,99` AUC** | Mengklaim *"model kami sakti menebak perilaku manusia 99%"*. | Menjelaskan **batas Bayes optimal data sintetis (`~94,10%` Akurasi / `0,9907` AUC)** dan membuktikan kebebasan dari *data leakage* melalui protokol *Split-Before-Fit* dan uji Kontrol Negatif (`AUC ~0,50`). |
| **Klaim Pemilihan Model** | Mengklaim *"Decision Tree model dengan akurasi tertinggi"*. | Menyampaikan **analisis *trade-off* 3 sumbu**: `LR_Tuned` dan `GBT_Tuned` unggul tipis `+0,00024` AUC (`0,9906`), namun `DT_Champion_d10` (`0,9904`) menang pada **kecepatan latih (`14,42s`)**, **transparansi aturan *white-box*** (Surrogate 4-daun fidelitas `98,28%`), dan **mengalahkan Random Forest (`0,9894`, `354,79s`)**. |
| **Nilai Kontribusi Jurnal** | Rekomendasi promo e-commerce sebagai kesimpulan utama ilmiah. | Rekomendasi bisnis ditempatkan hanya sebagai **ilustrasi penerapan (*use-case*)**, sedangkan kontribusi ilmiah utama adalah **arsitektur PySpark, regulasi *pre-pruning*, fenomena *DT > RF* pada fitur derau, dan deteksi *Greedy Collinear Masking* dengan TreeSHAP**. |

---

## 2. Identitas & Aturan Ketat Jurnal Target (JIKO — Sinta 4)

- **Jurnal Target**: **JIKO (Jurnal Informatika dan Komputer)** — Terakreditasi Sinta 4 (`e-ISSN: 2477-3964`).
- **Tautan Kerja & Template**:
  - Draft Kolaborasi Google Docs: [Google Docs Draft Jurnal](https://docs.google.com/document/d/1PmyoDR4VLFh8KZuLFNhsFb7ruX9wgAUm/edit?tab=t.0)
  - Template Resmi Word / LaTeX JIKO & *Statement of Originality*: wajib diikuti sesuai pedoman submisi OJS JIKO (kuarto A4, spasi `1,5`).
- **Bahasa Naskah**: **Bahasa Indonesia** akademis baku (mengacu gaya terbitan JIKO; istilah teknis asing dicetak *miring*).

### Checklist Batasan Teknis Wajib JIKO & Aturan Sitasi
1. **Panjang & Tata Letak**: **`5.000–8.500` kata** (target ideal **`6.000–6.500` kata** termasuk tabel, gambar, dan daftar pustaka), maksimal **`16` halaman A4** (spasi `1,5`, tidak termasuk daftar riwayat penulis).
2. **Judul**: Maksimal **`12` kata**, format *Title Case*.
3. **Identitas Penulis**: Nama lengkap **tanpa gelar akademik**, dilengkapi Program Studi/Departemen, Fakultas, Universitas, Kota, Negara, dan alamat email aktif.
4. **Abstrak & Kata Kunci**: **`150–250` kata** (memuat 4 unsur: tujuan/masalah, desain pendekatan, temuan utama, ringkasan interpretasi/simpulan) + **maksimal `5` kata kunci**.
5. **Pendahuluan (`2–3` halaman)**: Ditulis menyatu **tanpa sub-bab (*subsection*)**, memuat latar belakang masalah komputasi Big Data & *trade-off* model, kajian pustaka relevan, tujuan studi, serta paragraf penutup yang menegaskan alur hal-hal yang dibahas.
6. **Metode Penelitian**: Menjabarkan urutan langkah logis eksperimen, arsitektur pemrosesan terdistribusi, serta penomoran gambar/tabel berurutan.
7. **Hasil Penelitian (*Strictly Objective*)**: Memaparkan temuan faktual secara objektif **tanpa interpretasi atau perbandingan dengan penelitian terdahulu**. Semua tabel & gambar wajib dirujuk di teks; hasil uji statistik wajib memuat nama uji (`McNemar`, `Bootstrap`), nilai statistik ($\chi^2$), derajat bebas (`df=1`), *p-value* eksak, dan konsistensi angka desimal.
8. **Pembahasan (*Critical Analysis*)**: Analisis kritis — menafsirkan temuan teknis, membandingkan dengan teori & penelitian sebelumnya, menjelaskan kebaruan (*novelty*), implikasi metodologis, serta keterbatasan riset.
9. **Kesimpulan**: Mengulang topik dalam kalimat pembuka yang jernih beserta urgensinya; ditulis dalam **1 paragraf utuh (DILARANG memakai daftar nomor/*numbering* atau *bullet points*)**.
10. **Orisinalitas & AI Check**: Melampirkan *Statement of Originality* dan bukti **Turnitin Generative AI `< 20%`** (naskah hasil draf wajib disunting/dihumanisasi tim agar alami, variatif, dan bebas pola kalimat repetitif).
11. **ATURAN WAJIB SITASI (TIDAK BOLEH DILANGGAR — lihat `AGENTS.md` Rule 6)**:
    - **Minimal `15` referensi** berformat **IEEE** lengkap dengan **DOI valid yang dapat ditelusuri**. Seluruh daftar pustaka wajib dikutip di badan naskah.
    - **Keterbaruan**: Terbit **kurang dari `4` tahun terakhir (`2022–2026`)**.
    - **Sumber Wajib Jurnal Terakreditasi Sinta / Bereputasi**: Seluruh sitasi wajib diambil dari jurnal nasional terakreditasi **Sinta (`Sinta 1–6`)** — termasuk disarankan menyitir **artikel JIKO terdahulu** yang relevan — dan/atau jurnal internasional terindeks/bereputasi (**Scopus / WoS**). Dilarang menyitir blog, skripsi yang tidak terbit di jurnal, atau *preprint* tanpa *peer review*.
    - **DILARANG MENYITIR DARI ABSTRAK SAJA**: Sebelum menyitir sebuah makalah, isi lengkapnya (*full-text*: Metode, Hasil, Pembahasan) **WAJIB** dibaca/diverifikasi langsung agar kutipan tidak melenceng atau halusinasi.

### Opsi Judul Naskah ($\le 12$ Kata, *Title Case*)
1. **Opsi A (Rekomendasi Utama — 11 Kata)**:  
   *Evaluasi Komparatif Model Klasifikasi dan Interpretabilitas Pohon Keputusan pada Apache Spark*
2. **Opsi B (11 Kata)**:  
   *Analisis Trade-Off Akurasi, Efisiensi, dan Interpretabilitas Decision Tree pada Apache Spark*
3. **Opsi C (10 Kata)**:  
   *Komparasi Algoritma Klasifikasi Terdistribusi dan Pemangkasan Pohon Keputusan pada PySpark*

---

## 3. Garis Besar Isi Jurnal per Bagian (*Brief* untuk Tim & Penulis Argon)

| Bagian Naskah | Target Panjang | Fokus Bahasan Utama (Sesuai Arahan Bu Gema) | Data / Elemen Wajib dari Repo |
| :--- | :---: | :--- | :--- |
| **Judul & Penulis** | — | Judul netral berfokus metode/model ($\le 12$ kata); lengkapi profil 4 penulis tanpa gelar. | Pilih `Opsi A`/`B`/`C`; lengkapi prodi, fakultas, univ, kota, email. |
| **Abstrak** | `150–250` kata | Masalah skala `10⁶` baris & dilema akurasi vs interpretabilitas; arsitektur PySpark *Split-Before-Fit* + 6 model; hasil komparasi + *surrogate fidelity*; kesimpulan rujukan model. | Angka kompresi Parquet (`83,75%`), `LR`/`GBT` (`0,9906`) vs `DT` (`0,9904`, `14,42s`) vs `RF` (`0,9894`, `354,79s`), fidelitas `98,28%`. |
| **1. Pendahuluan** | `2–3` halaman (**tanpa sub-bab**) | Alur corong: (1) Tantangan klasifikasi tabular skala besar & biaya I/O; (2) Dilema praktis *black-box ensemble* vs model linier vs pohon *white-box*; (3) Celah riset umum (*data leakage* pra-split, tuning pada data uji, bias *feature importance* kolinear); (4) Posisi penelitian ini sebagai **pipeline & benchmark rujukan** pada data statis teruji. | Sitasi literatur Sinta/JIKO & Scopus `< 4` tahun (full-text verified) tentang PySpark, klasifikasi e-commerce, *tree pruning*, & XAI. |
| **2. Metode Penelitian** | `~1.400` kata | Penjelasan sistematis tahapan teknis: (a) Spesifikasi lingkungan & karakteristik dataset uji statis/sintetis (`10⁶` baris $\times$ `60` kolom); (b) Arsitektur penyimpanan kolumnar Snappy Parquet & isolasi **Split-Before-Fit** (`seed=42`, dual vektor `57` & `132` dim); (c) Protokol tuning 72 konfigurasi pada Validation Split + ablasi `M0–M4`; (d) Skenario komparasi adil 6 model, 5-Fold CV, McNemar, Bootstrap 95% CI, Kontrol Negatif, & Sensitivitas Threshold; (e) Ekstraksi aturan *surrogate*, Audit XAI (*TreeSHAP* & *Permutation*), serta uji skalabilitas baris/core. | **Gambar 1** (Diagram Pipeline Metodologi) & **Tabel 1** (Ringkasan Skema Dataset & Pemisahan Split). |
| **3. Hasil** | `~1.300` kata (**objektif murni**) | Penyajian fakta statistik & numerik secara kronologis: (a) Kinerja I/O CSV vs Parquet; (b) Kurva tuning validasi 72 titik & hasil ablasi `M0–M4` (*node explosion* `3.431` vs `473`); (c) Tabel kinerja 6 model + 5-Fold CV; (d) Hasil Uji McNemar & Bootstrap 95% CI; (e) Kontrol Negatif (`~0,50`) & Sensitivitas Threshold (`35%–45%`); (f) Fidelitas aturan 4 daun (`98,28%`) & perbandingan atribusi Gini vs RF vs SHAP vs *Joint Permutation*; (g) Kurva skalabilitas baris (`20%–100%`) & core (`local[1..4]`). | **Tabel 2, 3, 4, 5** dan **Gambar 2, 3, 4, 5** (seluruh p-value, `df=1`, `χ²`, & CI dilaporkan konsisten tanpa opini/komparasi literatur). |
| **4. Pembahasan** | `~1.800` kata (**analisis kritis**) | **Inti kontribusi rujukan jurnal**: (1) *Mengapa performa menyentuh `~94%` Akurasi / `~0,99` AUC?* — bedah batas Bayes optimal pada data sintetis vs bukti nol *leakage*; (2) *Trade-Off 3 Sumbu* — mengapa selisih `0,00024` AUC tidak membenarkan kompleksitas & waktu latih `GBT`/`LR`/`RF` bila target sistem membutuhkan aturan transparan; (3) *Anomali DT Mengalahkan RF (`0,9904` vs `0,9894`)* — analisis `featureSubsetStrategy='sqrt'` saat sinyal terkonsentrasi di sedikit fitur di tengah 53 fitur derau; (4) *Pelajaran Kolinearitas XAI* — bahaya mengandalkan Gini pohon tunggal tanpa SHAP/Permutation; (5) *Keterbatasan Studi* — dataset statis/sintetis & eksperimen Spark mode lokal 2 *physical cores*. | Referensi balik ke **Tabel 2–5**, **Gambar 2–5**, serta perbandingan dengan temuan penelitian Sinta/Scopus lain yang dibaca *full-text*. |
| **5. Kesimpulan** | `150–220` kata (**1 paragraf utuh**) | Satu paragraf lengkap tanpa *bullet* atau nomor: menegaskan ulang urgensi evaluasi model terdistribusi pada skala sejuta baris, merangkum efektivitas pipeline PySpark + Snappy Parquet + *DT_Champion_d10* (`AUC 0,9904`, `14,42s`) dan *Surrogate 4-daun* (`98,28%`), serta posisinya sebagai acuan replikasi eksperimental pada data serupa. | Angka inti terverifikasi dari `model_comparison_summary.json` & `rules_and_fidelity.json`. |
| **Daftar Pustaka** | `>= 15` entri | Format **IEEE** lengkap dengan **DOI aktif**. Rencana komposisi: `2–3` artikel **JIKO**, `4–5` artikel Sinta (`PySpark` / klasifikasi big data), `4–5` artikel Sinta/Scopus (*Decision Tree pruning* & *benchmark* model), `4–5` artikel Sinta/Scopus (*Explainable AI* / SHAP / kolinearitas). | Wajib diverifikasi penuh (bukan abstrak) sebelum disitir! |

---

## 4. Ringkasan Angka Eksperimen Terverifikasi (`E0–E9`) untuk Tabel & Teks

Seluruh angka di bawah ini diambil langsung dari `reports/metrics/*.json` (Single Source of Truth):

### 4.1 Karakteristik Dataset, Pra-pemrosesan Parquet, & Pencegahan Kebocoran (`E0`)
- **Populasi**: `1.000.000` baris $\times$ `60` kolom mentah (`0` *missing values*).
- **Efisiensi Penyimpanan Kolumnar**: Konversi CSV (`233,60 MB`) ke Apache Parquet ber-kompresi Snappy (`37,96 MB`) memangkas ukuran penyimpanan sebesar **`83,75%`** dan mempercepat waktu baca + agregasi penuh dari **`10,847 detik` menjadi `0,705 detik` (`15,39x` *speedup*)**.
- **Ruang Fitur (`57` Prediktor Independen)**: `43` fitur numerik (termasuk `days_since_last_purchase` dari `last_purchase_date`) + `14` fitur kategorikal (`10` negara, `4` gender, `3` perangkat, `6` metode bayar, `8` kategori produk, dll.).
  - Vektor `features` (Model Pohon): `57` dimensi (`StringIndexer`).
  - Vektor `features_lr` (Regresi Logistik): `132` dimensi (`OneHotEncoder` + `StandardScaler(withMean=False, withStd=True)`).
- **Protokol *Split-Before-Fit* (`seed=42`)**:
  - Train Set: `799.723` baris (`79,97%`), dibagi secara internal menjadi **Train-Sub (`599.774` baris / `75%`)** dan **Validation (`199.949` baris / `25%`)** untuk pemilihan hyperparameter.
  - Test Set: `200.277` baris (`20,03%`), diisolasi penuh dan hanya dievaluasi satu kali di akhir.

### 4.2 Hasil Komparasi 6 Model Klasifikasi (`E1`, `E5`, `E9` — Tabel 3 Jurnal)

| Model | Konfigurasi Optimal | Kompleksitas Struktur | 5-Fold CV ROC-AUC | Test ROC-AUC (95% CI) | Test PR-AUC | Test Akurasi | Test F1-Weighted | Waktu Latih (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LR_Tuned`** | `regParam=0.0`, OHE+Scaled | 132 bobot kontinu | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9902, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9410`** | `18,26` |
| **`GBT_Tuned`** | `maxIter=20, maxDepth=5` | 20 pohon (958 node) | `0,9905 ± 0,0004` | **`0,9906`** (`[0,9901, 0,9912]`) | **`0,9912`** | **`94,10%`** | **`0,9409`** | `76,01` |
| **`DT_Champion_d10`** | `maxDepth=10, minInst=20, gini` | **1 pohon (473 node)** | `0,9883 ± 0,0005` | **`0,9904`** (`[0,9900, 0,9910]`) | **`0,9905`** | **`93,99%`** | **`0,9399`** | **`14,42`** |
| **`RF_Tuned`** | `numTrees=40, maxDepth=10, minInst=20` | 40 pohon (24.944 node) | `0,9894 ± 0,0005` | `0,9894` (`[0,9888, 0,9900]`) | `0,9900` | `93,92%` | `0,9392` | `354,79` |
| **`DT_Default_d7`** | `maxDepth=7, minInst=20, gini` | 1 pohon (41 node) | `0,9793 ± 0,0043` | `0,9773` (`[0,9763, 0,9782]`) | `0,9643` | `94,09%` | `0,9409` | `8,34` |
| **`DT_Shallow_d3`** | `maxDepth=3, minInst=20, gini` | **1 pohon (7 node / 4 daun)** | `0,9688 ± 0,0064` | `0,9651` (`[0,9639, 0,9664]`) | `0,9524` | `94,06%` | `0,9406` | `8,80` |

### 4.3 Uji Statistik Berpasangan McNemar ($df = 1$) & Ukuran Efek Bootstrap ($\Delta\text{ROC-AUC}$)
1. **`LR_Tuned` vs `DT_Champion_d10`**: $b = 1.202$, $c = 986$, $\chi^2_{\text{McNemar}} = 21,13$, $p = 4,23 \times 10^{-6}$; $\Delta\text{ROC-AUC} = +0,00024$ (`95% CI [+0,00013, +0,00037]`).
2. **`GBT_Tuned` vs `DT_Champion_d10`**: $b = 1.200$, $c = 985$, $\chi^2_{\text{McNemar}} = 20,96$, $p = 4,61 \times 10^{-6}$; $\Delta\text{ROC-AUC} = +0,00016$ (`95% CI [+0,00006, +0,00030]`).
3. **`RF_Tuned` vs `DT_Champion_d10`**: $b = 1.813$, $c = 1.945$, $\chi^2_{\text{McNemar}} = 4,57$, $p = 0,0326$; $\Delta\text{ROC-AUC} = -0,00109$ (`95% CI [-0,00128, -0,00089]`) — **`DT_Champion_d10` secara signifikan mengungguli `RF_Tuned` baik pada ROC-AUC maupun Akurasi, dengan waktu latih `24,6x` lebih singkat**.

### 4.4 Ablasi *Pre-Pruning* Decision Tree (`E2` — 72 Konfigurasi Validasi)
- **Efek Kedalaman & *Overfitting* (`M0` $\rightarrow$ `M2` $\rightarrow$ `M4`)**: Peningkatan kedalaman dari `maxDepth=3` (`7` node, Test AUC `0,9651`) ke `maxDepth=10` (`473` node, Test AUC `0,9904`) meningkatkan kualitas pemeringkatan probabilitas. Namun tanpa *pre-pruning* pada `maxDepth=12, minInstancesPerNode=1` (`M4`), jumlah node meledak **`7,25x` lipat (`3.431` node)** diikuti penurunan Validation AUC ke `0,9787` dan Test Akurasi ke `93,67%`.
- **Ablasi Murni `minInstancesPerNode` (`M2` vs `M3a` pada `maxDepth=10, gini`)**: Menaikkan ambang daun dari `20` ke `50` memangkas node sebesar **37,6%** (`473` $\rightarrow$ `295` node) dengan Test AUC `0,9835`.
- **Ablasi Murni `impurity` (`M2` vs `M3b` pada `maxDepth=10, minInst=20`)**: `gini` (`473` node, Test AUC `0,9904`) unggul tipis atas `entropy` (`227` node, Test AUC `0,9883`).

### 4.5 Fidelitas Pohon Ringkas (`E6`) & Resolusi Kolinearitas Fitur (`E8`)
- **Surrogate Rule Fidelity**: Pohon dangkal `DT_Shallow_d3` (`maxDepth=3`, hanya **4 daun aturan IF-THEN** yang memeriksa `impulse_buying_score <= 4,5`, `<= 5,5`, dan `stress_from_financial_decisions <= 5,5`) menghasilkan **kesepakatan prediksi `98,28%` (`196.825 / 200.277`)** terhadap `DT_Champion_d10` (`473` node) dan mencapai Test Akurasi **`94,06%`**.
- **Greedy Collinear Feature Masking**: Keempat prediktor psikologis (`impulse_buying_score` $r=+0,9615$, `stress_from_financial_decisions` $r=+0,9404$, `overall_stress_level` $r=+0,8635$, `impulse_purchases_per_month` $r=+0,8501$) membentuk klaster multikolinear. Pada pohon tunggal `DT_Champion_d10`, `99,72%` Gini importance terserap oleh 2 fitur pertama (`95,00%` + `4,72%`), sedangkan pada `RF_Tuned` dan **TreeSHAP**, bobot terdistribusi ke keempat fitur (`35,14%`, `31,15%`, `17,08%`, `14,04%`). Pengacakan serentak ke-4 fitur ini (*Joint Cluster Permutation*) menjatuhkan ROC-AUC seluruh model sebesar **`-0,4835` s.d. `-0,4899`** menuju `~0,50` (acak).

### 4.6 Uji Kontrol Negatif, Sensitivitas Threshold (`E7`), & Skalabilitas PySpark (`E3–E4`)
- **Kontrol Negatif (`checkout_abandonments_per_month >= 5`)**: Menghasilkan ROC-AUC **`0,4975`** (`LR` & `DT`) dan **`0,5004`** (`RF`), membuktikan ketiadaan kebocoran data pada pipeline.
- **Sensitivitas Ambang Target (`35%`, `40%`, `45%`)**: ROC-AUC konsisten pada rentang **`0,9882 – 0,9905`** (uji subsampel) dan `0,9904 – 0,9906` pada data penuh.
- **Skalabilitas Komputasi**: Waktu pelatihan `DT_Champion_d10` bertumbuh sub-linear/linier dari `5,95 detik` (`20%` Train / `160.127` baris) ke `14,42 detik` (`100%` Train / `799.723` baris) — tetap **`24,6x` lebih cepat** daripada `RF_Tuned` (`354,79s`) dan **`5,27x` lebih cepat** daripada `GBT_Tuned` (`76,01s`) — dengan percepatan paralelisme **`2,30x`** dari `local[1]` (`20,68s`) ke `local[2]` (`9,01s`).

---

## 5. Daftar Gambar & Tabel yang Akan Dimasukkan ke Manuskrip JIKO

| Nomor | Jenis | Sumber Artefak di Repo | Judul Tabel / Gambar pada Naskah |
| :---: | :---: | :--- | :--- |
| **Tabel 1** | Tabel | `reports/metrics/feature_metadata.json` | Ringkasan Skema Dataset, Kelompok Fitur, dan Statistik Deskriptif Target (`N = 1.000.000`) |
| **Tabel 2** | Tabel | `reports/metrics/dt_champion_summary.json` | Hasil Ablasi Parameter *Pre-Pruning* dan Kriteria *Impurity* pada Decision Tree (`M0–M4`) |
| **Tabel 3** | Tabel | `reports/metrics/model_comparison.csv` | Perbandingan Kinerja Prediktif, Kompleksitas Struktur, dan Waktu Pelatihan 6 Model pada Data Uji (`N = 200.277`) |
| **Tabel 4** | Tabel | `reports/metrics/mcnemar_tests.csv` | Hasil Uji Signifikansi Berpasangan McNemar ($\chi^2, df=1$) dan Selang Kepercayaan Bootstrap 95% ($\Delta\text{ROC-AUC}$) |
| **Tabel 5** | Tabel | `reports/metrics/rules_and_fidelity.json` | Ekstraksi Aturan Keputusan *White-Box* 4 Daun (`DT_Shallow_d3`) beserta Tingkat Fidelitas Surrogate (`98,28%`) |
| **Gambar 1** | Diagram | Diagram Arsitektur Metodologi | Arsitektur Pipeline Pemrosesan Terdistribusi PySpark dengan Isolasi *Split-Before-Fit* |
| **Gambar 2** | Plot 300 DPI | `reports/model_roc_pr_curves.png` | Kurva ROC dan *Precision-Recall* Komparatif Enam Model Klasifikasi pada Data Uji |
| **Gambar 3** | Plot 300 DPI | `reports/dt_pruning_depth_curve.png` | Kurva Validasi 72 Konfigurasi: Pengaruh `maxDepth`, `minInstancesPerNode`, dan `Impurity` terhadap ROC-AUC dan Jumlah Node |
| **Gambar 4** | Plot 300 DPI | `reports/permutation_vs_gini_importance.png` | Perbandingan Atribusi Fitur Lintas Metode (*Single-Tree Gini*, *Ensemble RF Gini*, *LR Weight*, dan *Permutation Importance*) |
| **Gambar 5** | Plot 300 DPI | `reports/pyspark_scaling_benchmark.png` | Evaluasi Skalabilitas Waktu Pelatihan PySpark terhadap Volume Baris Data dan Jumlah *Core* Eksekutor |

---

## 6. Checklist Pra-Penulisan & Eksekusi Bertahap (Sesi Argon Terpisah)

Sebelum mulai menulis naskah utuh di sesi terpisah agar kredit tetap efisien dan hasil akurat:
- [ ] **ACC Tim atas Brief ini**: Galih, Trias, Risha, & Raihan menyepakati fokus proses/perbandingan model sesuai arahan **Ibu Gema**.
- [ ] **Pilihan Judul & Data Penulis**: Tetapkan pilihan judul (**Opsi A / B / C**) serta kumpulkan nama tanpa gelar, prodi/departemen, fakultas, universitas, kota, negara, dan email seluruh penulis.
- [ ] **Siapkan Template JIKO**: Pastikan format paragraf & style dari file `.docx` / `.tex` resmi JIKO sudah siap.
- [ ] **Eksekusi 2 Fase Saat Penulisan**:
  - **Fase 1 (Riset & Verifikasi Referensi Full-Text)**: Cari & verifikasi `>= 15` jurnal terakreditasi **Sinta / Scopus (`2022–2026`)** + artikel **JIKO** dengan membaca isi artikel penuh (*dilarang dari abstrak saja*) dan mencatat DOI valid terlebih dahulu.
  - **Fase 2 (Penulisan Modular Naskah)**: Tulis draf Bahasa Indonesia per kloter (*Judul–Metode*, lalu *Hasil–Pembahasan–Kesimpulan–Referensi*), lalu lakukan penyuntingan manusia bersama tim guna memastikan **Turnitin AI `< 20%`** sebelum *Statement of Originality* dan submisi ke OJS.
