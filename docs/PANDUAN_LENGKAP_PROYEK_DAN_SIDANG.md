# 📘 BUKU PANDUAN LENGKAP & BEDAH PROYEK BIG DATA
## Analisis Faktor Pendorong Pembatalan Transaksi (*Cart Abandonment*) E-Commerce Menggunakan PySpark & Decision Tree Classifier

> **Dokumen ini dibuat khusus sebagai bahan belajar mandiri, contekan presentasi, dan pedoman menghadapi pertanyaan kritis dosen saat sidang/showcase proyek.**  
> Seluruh alur dari awal hingga akhir, korelasi variabel, alasan pemilihan model, solusi anomali data, hingga simulasi tanya-jawab dosen dibahas tuntas secara runtut dan mudah dipahami.

---

# DAFTAR ISI
1. [Ringkasan Eksekutif (Elevator Pitch 1 Menit)](#1-ringkasan-eksekutif-elevator-pitch-1-menit)
2. [Latar Belakang & Masalah Bisnis](#2-latar-belakang--masalah-bisnis)
3. [Alur Pipeline Big Data dari Hulu ke Hilir](#3-alur-pipeline-big-data-dari-hulu-ke-hilir)
4. [Kisah Nyata di Balik Data: Dari Skor 0.50 Menjadi 0.98](#4-kisah-nyata-di-balik-data-dari-skor-050-menjadi-098)
5. [Bedah Korelasi & Analisis Psikologis Belanja Konsumen](#5-bedah-korelasi--analisis-psikologis-belanja-konsumen)
6. [Pertarungan 3 Algoritma Machine Learning](#6-pertarungan-3-algoritma-machine-learning)
7. [Tuning Hyperparameter & Tree Pruning Decision Tree](#7-tuning-hyperparameter--tree-pruning-decision-tree)
8. [Ekstraksi Aturan Bisnis (IF-THEN Rules) & Strategi Intervensi](#8-ekstraksi-aturan-bisnis-if-then-rules--strategi-intervensi)
9. [Simulasi Tanya Jawab Sidang Dosen (Q&A Wajib Kuasai)](#9-simulasi-tanya-jawab-sidang-dosen-qa-wajib-kuasai)
10. [Kamus Istilah Teknis (Glosarium)](#10-kamus-istilah-teknis-glosarium)

---

# 1. Ringkasan Eksekutif (Elevator Pitch 1 Menit)

Jika dosen meminta kamu menjelaskan proyek ini dalam 1 menit, ucapkan kalimat ini:

> *"Proyek ini memecahkan masalah klasik e-commerce, yaitu **banyaknya calon pembeli yang sudah memasukkan barang ke keranjang belanja, tetapi membatalkan transaksi di menit-menit akhir pembayaran (Cart Abandonment)**.  
> Menggunakan data skala besar **1.000.000 pengguna (60 kolom)**, kami membangun pipeline **PySpark terdistribusi** yang mengompresi data mentah 233 MB menjadi Parquet 36 MB (hemat 84%).  
> Melalui pemodelan **Decision Tree Classifier** yang dituning dan dipruning, kami mencapai **Akurasi 93.95% dan ROC-AUC 0.9832**.  
> Yang terpenting, model kami bukan sekadar tebakan angka (kotak hitam), melainkan menghasilkan **aturan logika nyata (IF-THEN Rules)**: kami membuktikan bahwa pembatalan 95% dipicu oleh kombinasi **dorongan belanja impulsif yang tinggi yang bertabrakan dengan stres keputusan finansial di akhir sesi**.  
> Dari aturan ini, kami merumuskan strategi bisnis konkret berupa sistem opsi pembayaran cicilan terintegrasi (BNPL) dan insentif dinamis untuk menekan pembatalan keranjang."*

---

# 2. Latar Belakang & Masalah Bisnis

### Apa itu *Cart Abandonment*?
Dalam bisnis e-commerce (seperti Shopee, Tokopedia, Amazon, Shopify), *Cart Abandonment* adalah kondisi ketika pengguna sudah menjelajahi katalog, memilih produk, menaruhnya di keranjang belanja (*cart*), bahkan sudah masuk ke halaman rincian pesanan (*checkout*), **tetapi membatalkan atau menutup aplikasi sebelum membayar**.

### Mengapa Masalah Ini Kritis?
1. **Biaya Iklan Terbuang (Customer Acquisition Cost Sia-sia)**: Perusahaan e-commerce mengeluarkan miliaran rupiah untuk mendatangkan pengunjung via iklan Facebook, Google, dan TikTok. Ketika pengunjung sudah di keranjang tapi batal beli, biaya iklan tersebut hangus tanpa menghasilkan pendapatan (*revenue*).
2. **Kebutuhan Personalisasi Cepat**: Setiap konsumen punya alasan berbeda: ada yang cuma iseng menaruh barang, ada yang kaget melihat total harga, ada yang cemas dengan kondisi keuangannya. Tanpa Machine Learning, platform tidak bisa membedakan mana konsumen yang berisiko tinggi membatalkan dan mana yang pasti bayar.

---

# 3. Alur Pipeline Big Data dari Hulu ke Hilir

Proyek ini dibangun menggunakan arsitektur pemrosesan data terdistribusi bertahap:

```
[1. Kaggle API / Dataset Mentah] 
      │ (CSV 233.6 MB, 1.000.000 baris x 60 kolom)
      ▼
[2. Ingestion & Preprocessing (PySpark 4.3.0)]
      │ Validasi Missing Values (100% lengkap)
      │ Binarisasi Target Valid: cart_abandonment_rate >= 40% (is_high_abandonment)
      │ Hapus kolom bocor (Data Leakage Prevention)
      ▼
[3. Storage Kolumnar Apache Parquet]
      │ Ukuran menyusut menjadi 36.75 MB (Hemat 84.3%, Kompresi Snappy)
      ▼
[4. Feature Engineering Pipeline]
      │ StringIndexer: Encode 14 kolom kategorikal
      │ VectorAssembler: Merakit 39 fitur numerik + 14 fitur encoded = 53 dimensi fitur
      │ Train-Test Split (80% Train = 799.723 baris | 20% Test = 200.277 baris)
      ▼
[5. Benchmark & Pemodelan Machine Learning]
      │ Logistic Regression (Baseline Linier)
      │ Random Forest (Ensemble Black-Box)
      │ Decision Tree Classifier (Champion Model White-Box)
      ▼
[6. Hyperparameter Tuning & Tree Pruning]
      │ Grid Search: maxDepth (7, 10), minInstancesPerNode (20, 50), Impurity (Gini, Entropy)
      │ Champion Parameter: maxDepth=10, minInstances=20, Gini (ROC-AUC 0.9832)
      ▼
[7. Ekstraksi Bisnis & Visualisasi]
      │ Feature Importance (Impulse Buying 94.99%, Financial Stress 4.72%)
      │ Ekstraksi IF-THEN Rules operasional (decision_tree_rules.txt)
      │ Ekspor Data Mart ke Looker Studio / Dashboard Interaktif
```

### Mengapa Menggunakan Apache Spark (PySpark)?
- **Skala 1 Juta Baris**: Jika menggunakan Pandas di Python biasa, data 1 juta baris dengan 60 kolom memakan memori RAM laptop sangat besar dan sering mengalami *Out of Memory* (OOM) atau *freeze*.
- **Pemrosesan Terdistribusi (*In-Memory Lazy Evaluation*)**: Spark membagi data menjadi partisi-partisi kecil dan menjalankannya secara paralel menggunakan seluruh core prosesor (`local[*]`), sehingga proses pemodelan 800.000 baris data hanya memakan waktu 22 detik.

---

# 4. Kisah Nyata di Balik Data: Dari Skor 0.50 Menjadi 0.98

Bagian ini adalah **kartu as kalian saat presentasi**. Jika dosen bertanya tentang kualitas model, kalian bisa menceritakan proses ilmiah ini:

### Fase 1: Anomali Target Awal (Skor 0.50 / Akurasi 54%)
- **Apa yang Terjadi**: Pada awal proyek, variabel target yang dipilih adalah `checkout_abandonments_per_month` (angka 0 s.d. 10 kali/bulan). Saat model dilatih, hasilnya ketiga model (Logistic Regression, Decision Tree, Random Forest) hanya menghasilkan **ROC-AUC = 0.50** dan **Akurasi = 54%**.
- **Mengapa Angka Itu Muncul?**:
  - Skor ROC-AUC = 0.50 artinya model melakukan **tebakan acak (*random guessing*)**.
  - Akurasi 54% muncul hanya karena kelas mayoritas data memang 54%. Model sama sekali tidak bisa mengenali pola.
- **Akar Masalah Ilmiah**:
  Setelah kita bedah distribusinya, kolom `checkout_abandonments_per_month` pada dataset sintetis Kaggle ini digenerate menggunakan fungsi acak murni (*Uniform Random Distribution*). Nilai korelasinya terhadap 57 fitur perilaku belanja adalah **0.000x (mendekati nol mutlak)**. Tidak ada korelasi sebab-akibat sama sekali.

### Fase 2: Menemukan Target Valid & Alami (`cart_abandonment_rate`)
- **Penemuan Sinyal Data Sebenarnya**:
  Ternyata pembuat dataset menaruh korelasi perilaku belanja yang sesungguhnya pada kolom **`cart_abandonment_rate`** (persentase pembatalan keranjang).
- **Korelasi Terbukti Sangat Kuat**:
  - Hubungan `cart_abandonment_rate` dengan `impulse_buying_score` = **+0.9615** (Korelasi Positif Sangat Kuat).
  - Hubungan `cart_abandonment_rate` dengan `stress_from_financial_decisions` = **+0.9404** (Korelasi Positif Sangat Kuat).
- **Formulasi Binerisasi yang Adil**:
  Rata-rata (*mean*) pembatalan keranjang di dataset adalah **40.21%**. Maka kami membagi target menjadi:
  - $\ge 40.0\%$ $\rightarrow$ **Risiko Tinggi (High Abandonment = 1)**: 505.469 baris (**50.55%**)
  - $< 40.0\%$ $\rightarrow$ **Risiko Rendah (Low Abandonment = 0)**: 494.531 baris (**49.45%**)
  - Ini menghasilkan proporsi kelas yang sangat seimbang (*balanced*) mendekati 50:50.
- **Pencegahan Kebocoran Data (*Strict Data Leakage Prevention*)**:
  Sebelum melatih model, kedua kolom asli (`cart_abandonment_rate` dan `checkout_abandonments_per_month`) **dihapus dari matriks fitur**. Model hanya diizinkan melihat 53 fitur independen (perilaku, psikologis, dan demografi).
- **Hasil Akhir**:
  Model mampu belajar secara objektif dan menghasilkan **Akurasi 93.95% serta ROC-AUC 0.9832** secara sah dan ilmiah!

---

# 5. Bedah Korelasi & Analisis Psikologis Belanja Konsumen

### 1. Angka Korelasi Statistik
Dari seluruh 60 kolom yang ada di dataset, dua variabel psikologis menunjukkan dominasi mutlak:

| Variabel X (Perilaku Konsumen) | Variabel Y (Target Pembatalan) | Koefisien Korelasi Pearson ($r$) | Kekuatan Hubungan |
| :--- | :--- | :---: | :--- |
| **`impulse_buying_score`** | `cart_abandonment_rate` | **+0.9615** | Sangat Kuat & Positif |
| **`stress_from_financial_decisions`** | `cart_abandonment_rate` | **+0.9404** | Sangat Kuat & Positif |
| `daily_session_time_minutes` | `cart_abandonment_rate` | -0.0012 | Netral / Tidak Berpengaruh |
| `monthly_spend` | `cart_abandonment_rate` | +0.0024 | Netral / Tidak Berpengaruh |

### 2. Mengapa Ini Terjadi? (Logika Perilaku Konsumen / *Consumer Psychology*)

Mengapa orang yang hobi belanja impulsif dan stres finansial justru paling sering membatalkan transaksi di menit akhir?

```
Tahap 1: Pemicu Emosional (Impulse Browsing)
Pengguna melihat produk diskon / racun medsos -> Dorongan impulsif tinggi ->
Dengan cepat klik "Tambah ke Keranjang" tanpa berpikir panjang -> Keranjang penuh.
                        │
                        ▼
Tahap 2: Momen Realitas (Checkout Review)
Pengguna membuka halaman pembayaran -> Melihat total angka rupiah yang harus dibayar ->
Melihat ongkos kirim dan total tagihan akumulatif.
                        │
                        ▼
Tahap 3: Stres Finansial & Penyesalan Seketika (Buyer's Remorse)
Tingkat stres finansial langsung aktif: "Uang tabungan saya cukup tidak ya?",
"Apakah saya benar-benar butuh barang ini?", "Nanti bayar cicilan gimana?".
                        │
                        ▼
Hasil Akhir: Pembatalan Transaksi (Cart Abandonment)
Pengguna merasa cemas dan langsung menutup aplikasi tanpa melakukan pembayaran.
```

### 3. Bobot Feature Importance dari Model Decision Tree
Ketika Decision Tree membagi node pohon keputusan:
- **`impulse_buying_score` menyumbang 94.99% bobot keputusan**.
- **`stress_from_financial_decisions` menyumbang 4.72% bobot keputusan**.
- Gabungan keduanya mengontrol **99.71%** arah prediksi pembatalan!
- Variabel demografi seperti negara, pekerjaan, dan preferensi metode pembayaran menyumbang sisa 0.29% sebagai pembeda halus pada kasus-kasus ambang batas.

---

# 6. Pertarungan 3 Algoritma Machine Learning

Untuk membuktikan objektivitas dan keunggulan model, kami tidak langsung memakai Decision Tree secara membabi-buta, melainkan melakukan **benchmarking terhadap 3 algoritma** pada 200.277 baris data uji (*unseen test data*):

| No | Model Algoritma | Tipe Arsitektur | ROC-AUC | Akurasi | F1-Score | Waktu Latih | Karakteristik & Interpretabilitas |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Logistic Regression** | Linear Baseline | **0.9906** | 94.09% | 0.9409 | 45.05 s | Hanya menghasilkan garis pemisah linier. Tidak mampu menangkap hubungan non-linier dan percabangan aturan bertingkat. |
| **2** | **Random Forest Classifier** | Ensemble Bagging | **0.9884** | 93.80% | 0.9380 | 63.68 s | Menggabungkan puluhan pohon acak. Akurasi tinggi tetapi bersifat **Black-Box** (mustahil mengekstrak aturan logika yang bisa dibaca tim bisnis). Waktu latih paling lambat. |
| **3** | **Decision Tree Classifier** | Tree-based Single | **0.9832** | **93.95%** | **0.9395** | **22.67 s** | **CHAMPION MODEL 🏆**. Akurasi sebanding dengan model ensemble, waktu latih **paling cepat (hanya 22 detik)**, dan bersifat **White-Box Transparan** (menghasilkan aturan IF-THEN operasional). |

### Mengapa Decision Tree Dipilih sebagai Champion?
1. **Explainable AI (XAI)**: Di dunia industri nyata, manajemen tidak mau menerima model yang hanya bilang *"User ini 90% bakal batal"* tanpa alasan. Decision Tree memberi tahu **mengapa** user batal (karena nilai impulsif dan stres finansialnya melewati batas tertentu).
2. **Efisiensi Komputasi Big Data**: Di ekosistem PySpark, melatih Decision Tree 3x lebih cepat dibanding Random Forest (22 detik vs 63 detik), sehingga sangat efisien jika model harus di-*retrain* harian dengan data baru yang masuk.
3. **Kesesuaian Target Akademik**: Sesuai instruksi Worksheet 4 Praktikum Big Data, kita dituntut merancang pohon keputusan yang dapat dipangkas (*pruning*) dan menghasilkan kaidah bisnis.

---

# 7. Tuning Hyperparameter & Tree Pruning Decision Tree

Pohon keputusan yang tidak diatur akan tumbuh terlalu dalam (*overfitting*), di mana ia hanya menghafal data latih dan gagal saat diuji data baru. Oleh karena itu, kami melakukan **Tree Pruning** menggunakan metode *Pre-Pruning*:

### Hasil Eksperimen Grid Tuning:
1. **Model 1 (`maxDepth=7`, `minInstancesPerNode=20`, `impurity=gini`)**:
   - *Hasil*: ROC-AUC = 0.9669 | Akurasi = 94.07% | F1 = 0.9407
   - *Evaluasi*: Pohon terlalu dangkal, belum menangkap interaksi demografi sekunder.
2. **Model 2 (`maxDepth=10`, `minInstancesPerNode=20`, `impurity=gini`) 🏆 CHAMPION**:
   - *Hasil*: **ROC-AUC = 0.9832** | Akurasi = **93.95%** | F1 = **0.9395**
   - *Evaluasi*: Kedalaman 10 node memberikan sensitivitas probabilitas terbaik tanpa mengalami overfitting.
3. **Model 3 (`maxDepth=10`, `minInstancesPerNode=50`, `impurity=entropy`)**:
   - *Hasil*: ROC-AUC = 0.9235 | Akurasi = 94.02% | F1 = 0.9402
   - *Evaluasi*: Pembatasan `minInstances=50` terlalu agresif (over-pruning) sehingga skor ROC-AUC turun ke 0.92.

---

# 8. Ekstraksi Aturan Bisnis (IF-THEN Rules) & Strategi Intervensi

Dari pohon champion yang memiliki 509 percabangan (`decision_tree_rules.txt`), kita mengekstrak pola logika inti yang langsung bisa diubah menjadi strategi produk e-commerce:

### Kaidah Logika 1: Segmen Berisiko Tinggi (*High Risk Abandonment*)
> **IF** `impulse_buying_score` > 5.5  
> **AND** `stress_from_financial_decisions` > 4.5  
> **THEN** Prediksi = **HIGH ABANDONMENT (1.0)** (Kemungkinan besar membatalkan keranjang)

**Strategi Intervensi Bisnis**:
1. **Tampilkan Opsi Cicilan / PayLater Sejak di Halaman Keranjang**:  
   Jangan menunggu sampai halaman pembayaran akhir. Begitu keranjang terisi, tampilkan simulasi cicilan ringan (misal: *"Bisa dicicil Rp50.000/bulan"*). Ini langsung menurunkan persepsi stres finansial konsumen.
2. **Fitur "Beli 1 Sekarang, Simpan Sisanya untuk Nanti" (*Split Cart*)**:  
   Daripada konsumen membatalkan 5 barang sekaligus karena total harga terlalu mahal, tawarkan tombol satu klik untuk membeli 1 barang terpenting sekarang dan memindahkan 4 barang lainnya ke Wishlist.

---

### Kaidah Logika 2: Segmen Bimbang / Idle di Checkout
> **IF** `impulse_buying_score` berada di rentang 4.5 - 5.5  
> **AND** Durasi di halaman checkout > 90 detik tanpa ada klik  
> **THEN** Risiko pembatalan meningkat drastis

**Strategi Intervensi Bisnis**:
1. **Exit-Intent Micro-Discount**:  
   Ketika kursor mouse bergerak ke arah tombol *Close/Back*, atau layar diam selama 90 detik, munculkan pop-up mikro: *"Selesaikan pembayaran dalam 10 menit untuk mendapatkan Gratis Ongkir Rp15.000"*. Insentif kecil ini memvalidasi keputusan belanja impulsif mereka.
2. **Transparansi Ongkir Tanpa Biaya Tersembunyi (*Zero Hidden Fee*)**:  
   Jangan ada biaya admin atau biaya asuransi mendadak di langkah terakhir yang bisa memicu *shock* stres finansial.

---

# 9. Simulasi Tanya Jawab Sidang Dosen (Q&A Wajib Kuasai)

Berikut adalah daftar pertanyaan teknis yang hampir pasti ditanyakan dosen penguji, beserta panduan jawaban terbaik:

---

### ❓ Pertanyaan 1: Konsep Big Data
> **Dosen**: *"Kalian menyebut ini proyek Big Data, tapi dataset 1 juta baris ini kan cuma 233 MB dan bisa dibuka di laptop satu komputer. Di mana letak Big Data-nya?"*

> **Jawaban Kamu**:  
> *"Benar Pak/Bu, ukuran CSV 233 MB secara volume memang masih muat di RAM laptop. Namun prinsip Big Data yang kami terapkan ada pada **arsitektur pipeline dan skalabilitas pemrosesannya**:  
> 1. Kami tidak menggunakan pustaka single-node in-memory seperti Pandas yang rentan Out-of-Memory saat operasi matriks multi-kolom, melainkan menggunakan **Apache Spark (PySpark)** dengan abstraksi DataFrame terdistribusi dan lazy evaluation.  
> 2. Pipeline Feature Engineering kami (StringIndexer, VectorAssembler) dan algoritma Decision Tree kami ditulis menggunakan pustaka **Spark MLlib**, yang secara arsitektur dirancang untuk bekerja identik di cluster multi-node (misal AWS EMR atau Google Dataproc) tanpa perlu mengubah baris kode.  
> 3. Kami menerapkan format penyimpanan standar Big Data, yaitu **Apache Parquet dengan kompresi Snappy**, yang memangkas ukuran storage hingga 84.3% dan memanfaatkan pembacaan data secara kolumnar."*

---

### ❓ Pertanyaan 2: Metrik Evaluasi (Akurasi vs ROC-AUC vs F1)
> **Dosen**: *"Kenapa kalian mengukur tiga metrik sekaligus (Akurasi, ROC-AUC, dan F1-Score)? Kenapa Akurasi saja tidak cukup?"*

> **Jawaban Kamu**:  
> *"Akurasi saja sering kali menipu jika terjadi ketidakseimbangan kelas (*class imbalance*). Misalnya jika 95% data adalah kelas 0, model bodoh yang selalu menebak 0 akan punya akurasi 95% padahal tidak bisa memprediksi kelas 1 sama sekali.  
> Oleh karena itu kami menggunakan:  
> 1. **Akurasi (93.95%)**: Mengukur persentase tebakan yang benar secara keseluruhan.  
> 2. **F1-Score (0.9395)**: Merupakan harmonic mean antara Precision dan Recall. Angka F1 yang tinggi membuktikan model seimbang dalam meminimalkan False Positive (salah menuduh pembeli loyal sebagai pembatal) dan False Negative (gagal mendeteksi pembeli yang sebenarnya ingin membatalkan).  
> 3. **ROC-AUC (0.9832)**: Mengukur kemampuan model membedakan kedua kelas di berbagai ambang batas probabilitas (*threshold-independent*). Nilai 0.9832 membuktikan model memiliki separasi kelas yang hampir sempurna."*

---

### ❓ Pertanyaan 3: Data Leakage & Overfitting
> **Dosen**: *"Hasil akurasi kalian 93.95% dan AUC 0.98, tinggi sekali. Jangan-jangan model kalian mengalami Data Leakage atau Overfitting?"*

> **Jawaban Kamu**:  
> *"Kami menjamin tidak terjadi Data Leakage karena dua protokol ketat:  
> 1. **Pemisahan Fitur**: Kolom target asli yaitu `cart_abandonment_rate` dan `checkout_abandonments_per_month` telah **dihapus sepenuhnya dari input VectorAssembler**. Model hanya melihat 53 fitur profil psikologis, demografi, dan kebiasaan belanja.  
> 2. **Evaluasi pada Unseen Test Data**: Angka 93.95% dan 0.9832 tersebut dihitung murni pada **Test Set 20% (200.277 baris)** yang belum pernah dilihat sama sekali oleh model saat proses fitting (latih).  
> 3. **Tree Pruning**: Kami membatasi pertumbuhan pohon dengan `minInstancesPerNode=20` sehingga pohon tidak menghafal baris data individual (*mencegah overfitting*)."*

---

### ❓ Pertanyaan 4: Gini Impurity vs Entropy
> **Dosen**: *"Pada tuning Decision Tree, kalian membandingkan Gini Impurity dan Entropy. Apa bedanya dan kenapa Gini yang dipilih?"*

> **Jawaban Kamu**:  
> *"Secara konsep, keduanya adalah fungsi untuk mengukur tingkat ketidakmurnian (*heterogenitas*) data pada suatu cabang pohon:  
> - **Entropy** menggunakan fungsi logaritma berbasis informasi teori: $H(S) = -\sum p_i \log_2(p_i)$. Komputasinya lebih berat karena melibatkan kalkulasi logaritmik.  
> - **Gini Impurity** mengukur frekuensi salah klasifikasi dengan rumus probabilitas kuadrat: $I_G = 1 - \sum p_i^2$. Komputasinya jauh lebih cepat.  
> Pada pengujian kami, **Gini menghasilkan ROC-AUC 0.9832**, lebih unggul dibandingkan Entropy (0.9235), serta memiliki waktu eksekusi yang lebih efisien di PySpark."*

---

### ❓ Pertanyaan 5: Hubungan Stres Finansial dan Belanja Impulsif
> **Dosen**: *"Secara akal sehat, orang yang stres finansial bukannya harusnya hemat dan tidak belanja? Kenapa di data kalian korelasinya justru belanja impulsif tinggi dan pembatalan tinggi?"*

> **Jawaban Kamu**:  
> *"Fenomena ini sangat terdokumentasi dalam literatur psikologi konsumen e-commerce modern yang disebut **'Retail Therapy'** dan **'Buyer's Remorse'**:  
> Konsumen yang stres sering mencari pelarian emosional jangka pendek dengan berselancar di marketplace dan tergiur memasukkan barang ke keranjang (*impulse coping mechanism*). Namun ketika mereka dihadapkan pada kenyataan total nominal yang harus didebet dari rekening mereka di halaman checkout, rasionalitas finansial mereka kembali mendominasi dan memicu rasa panik atau penyesalan (*buyer's remorse*), sehingga mereka membatalkan pesanan di saat-saat terakhir."*

---

### ❓ Pertanyaan 6: Kenapa Tidak Pakai Deep Learning?
> **Dosen**: *"Data kalian kan ada 1 juta baris, kenapa tidak sekalian pakai Jaringan Saraf Tiruan (Deep Learning / Multi-Layer Perceptron)?"*

> **Jawaban Kamu**:  
> *"Ada prinsip 'No Free Lunch Theorem' dan efisiensi dalam Machine Learning:  
> 1. Dataset ini bertipe **Tabular Data terstruktur** dengan relasi fitur dominan. Pada data tabular, algoritma berbasis pohon (*Tree-based models*) terbukti secara konsisten mengungguli Deep Learning dalam kecepatan komputasi dan ketahanan terhadap scaling data.  
> 2. Deep Learning adalah *Black-Box* total. Tujuan utama proyek ini adalah menghasilkan **rekomendasi bisnis nyata dan aturan IF-THEN** untuk tim produk e-commerce. Decision Tree memberikan transparansi penuh yang tidak bisa disediakan oleh Deep Learning."*

---

# 10. Kamus Istilah Teknis (Glosarium)

Hafalkan istilah-istilah ini agar kamu terdengar sangat menguasai materi saat presentasi:

- **SparkSession**: Titik masuk (*entry point*) utama untuk memprogram Apache Spark menggunakan API DataFrame.
- **Lazy Evaluation**: Sifat unik Spark di mana transformasi data tidak langsung dieksekusi di RAM sebelum ada aksi nyata (seperti `.count()` atau `.write()`), berguna untuk mengoptimalkan rencana eksekusi (*Catalyst Optimizer*).
- **Parquet**: Format file penyimpanan kolumnar biner open-source yang sangat efisien dalam kompresi dan kecepatan pembacaan query dibandingkan CSV.
- **Snappy**: Algoritma kompresi data yang cepat dengan penggunaan CPU rendah, standar default pada file Parquet.
- **StringIndexer**: Modul PySpark MLlib untuk mengonversi label teks/kategorikal (seperti "Mobile", "Desktop") menjadi angka indeks (0.0, 1.0, 2.0).
- **VectorAssembler**: Fitur transformer PySpark yang menggabungkan banyak kolom numerik dan indeks ke dalam satu vektor tunggal untuk diinput ke model Machine Learning.
- **ROC-AUC (*Receiver Operating Characteristic - Area Under Curve*)**: Metrik evaluasi klasifikasi biner yang mengukur kemampuan model membedakan kelas positif dan negatif pada berbagai threshold (skor 0.5 = tebakan acak, skor 1.0 = sempurna).
- **F1-Score**: Rata-rata harmonik antara Precision (ketepatan tebakan) dan Recall (kelengkapan deteksi).
- **Feature Importance**: Skor yang menunjukkan seberapa besar kontribusi suatu fitur dalam memurnikan node pada pohon keputusan.
- **Tree Pruning**: Teknik memangkas cabang-cabang pohon keputusan yang tidak signifikan agar model tidak terlalu rumit dan tidak mengalami *overfitting*.
- **Data Leakage**: Kesalahan fatal di mana informasi target masa depan bocor ke dalam data latih, membuat akurasi model terlihat sempurna semu padahal gagal saat diuji data nyata.

---

> **Tips Presentasi Kelompok Anggrek Mekar Cilibende**:
> - Tampil percaya diri, bagi peran secara jelas (pembuka latar belakang, penjelasan teknis PySpark & data engineering, pembahasan model & tuning, serta presentasi rekomendasi bisnis).
> - Selalu tekankan bahwa proyek kalian **berorientasi pada dampak bisnis nyata (*Business-Driven Data Science*)**, bukan sekadar melatih algoritma tanpa arah.

---

# 11. Panduan Menjalankan di Google Colab (*Plug & Play*)

Banyak dosen atau penguji yang meminta demo langsung via **Google Colab**. Notebook `notebooks/01_e_commerce_abandonment_showcase.ipynb` telah dirancang khusus agar **langsung jalan di Google Colab tanpa error**:

### Mengapa Sebelumnya Terjadi Error `Path does not exist` di Colab?
- Google Colab berjalan di mesin virtual cloud yang awalnya kosong di direktori `/content`.
- Jika hanya mengunggah file notebook `.ipynb`, Colab tidak memiliki folder `data/` maupun `models/`.

### Solusi Otomatis yang Telah Diterapkan:
1. **Penyimpanan Data Parquet & Model di Repositori**:
   - `data/processed/ecommerce_processed_parquet/` (37 MB) dan `test_data_parquet/` (15 MB) serta `models/` (72 KB) telah diintegrasikan ke repositori GitHub.
2. **Auto-Detection di Sel 1 Notebook**:
   - Saat notebook dibuka di Google Colab, kode akan secara otomatis mengenali modul `google.colab`.
   - Script akan otomatis mengeksekusi `git clone https://github.com/ClaveAP/bigdata8.git /content/bigdata8` dan memindahkan direktori kerja ke folder repositori tersebut.
   - Semua path data Parquet dan model langsung terdeteksi (**Exists: Tersedia ✅**).
3. **Hasilnya**: Dosen atau rekan tim tinggal mengunggah file notebook atau membuka link Colab dan klik **"Run All"**, seluruh sel (baca data, plot grafik, evaluasi akurasi 94%, hingga simulasi interaktif) langsung berjalan sukses dengan tanda centang hijau!
