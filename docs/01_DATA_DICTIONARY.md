# Data Dictionary: E-Commerce Shopper Behavior & Lifestyle (60 Kolom Mentah + 2 Kolom Turunan)

Dokumen ini merupakan referensi tunggal (*Single Source of Truth*) untuk skema data pada proyek **Big Data E-Commerce Cart Abandonment** dan naskah jurnal **JIKO**.
- **Sumber Dataset**: `dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based` (Kaggle, Lisensi Terbuka untuk Riset/Akademik dengan atribusi).
- **Volume Populasi**: `1.000.000` baris profil konsumen unik.
- **Jumlah Kolom**:
  - **CSV Mentah**: `60` kolom (`0` nilai *null/missing* di seluruh 60 kolom).
  - **Parquet Terproses (`ecommerce_processed_parquet`)**: `62` kolom (`60` kolom asli + `days_since_last_purchase` + target biner `is_high_abandonment`).
  - **Fitur Prediktor Independen (`ALL_FEATURE_NAMES`)**: `57` fitur (`43` numerik + `14` kategorikal), setelah mengecualikan `user_id` (ID), `last_purchase_date` (digantikan oleh `days_since_last_purchase`), `cart_abandonment_rate` (sumber target), dan `checkout_abandonments_per_month` (kontrol negatif/noise).

---

## Ringkasan Struktur Matematis & Korelasi Fitur

Berdasarkan audit populasi penuh ($N = 1.000.000$) di [`reports/metrics/schema_and_correlation_audit.csv`](file:///Users/admin/code/code-kuliah/bigdata-pjbl/reports/metrics/schema_and_correlation_audit.csv):
1. **Klaster 4 Variabel Psikologis Kolinear (Penggerak Utama Pembatalan)**:
   - `impulse_buying_score`: $r = +0,9615$ terhadap `cart_abandonment_rate` ($r = +0,8414$ terhadap `is_high_abandonment`).
   - `stress_from_financial_decisions`: $r = +0,9404$ terhadap `cart_abandonment_rate` ($r = +0,8282$ terhadap `is_high_abandonment`).
   - `overall_stress_level`: $r = +0,8635$ terhadap `cart_abandonment_rate` ($r = +0,7697$ terhadap `is_high_abandonment`).
   - `impulse_purchases_per_month`: $r = +0,8501$ terhadap `cart_abandonment_rate` ($r = +0,7426$ terhadap `is_high_abandonment`).
   - Keempat variabel ini memiliki kolinearitas silang sangat tinggi ($r = 0,8419 \text{ s.d. } 0,9162$), sehingga pohon keputusan tunggal (*greedy*) mengalami *Collinear Feature Masking* (menyerap $>99\%$ *Gini importance* pada 2 variabel pertama), sedangkan *Random Forest*, *Logistic Regression*, *Permutation Importance*, dan *SHAP* mendistribusikan bobot ke keempat variabel tersebut.
2. **53 Fitur Prediktor Lainnya (Demografi, Finansial, Sesi, Retur, & Temporal)**:
   - Berdistribusi seragam/independen terhadap `cart_abandonment_rate` dengan $|r| < 0,0030$.
3. **Variabel Kontrol Negatif (`checkout_abandonments_per_month`)**:
   - Berdistribusi seragam murni (`0..10`, rata-rata `4,9975`, SD `3,1628`) dengan korelasi $r = -0,000944$ terhadap `cart_abandonment_rate`. Digunakan pada eksperimen **E7** untuk membuktikan tidak adanya *overfitting* semu (seluruh model menghasilkan $\text{ROC-AUC} \approx 0,50$).

---

## 1. Identitas, Temporal, & Variabel Target (5 Kolom Mentah + 2 Kolom Turunan)

| No | Nama Kolom | Tipe Data Asli | Peran dalam Pipeline | Kardinalitas / Rentang | Mean ± SD | Korelasi ($r$) ke `cart_abandonment_rate` | Deskripsi |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| 1 | `user_id` | `INTEGER` | Identifier (Dikecualikan dari fitur) | `1` s.d. `1.000.000` (Unik) | `500.000,5 ± 288.675,3` | `-0,0015` | Identitas unik setiap pengguna |
| 2 | `last_purchase_date` | `DATE` (`STRING` ISO) | Sumber Fitur Temporal (Dikecualikan setelah ekstraksi) | `2025-01-01` s.d. `2027-01-01` (`731` tanggal) | — | — | Tanggal transaksi terakhir konsumen |
| 3 | `cart_abandonment_rate` | `INTEGER` | **Sumber Target Utama** (Dikecualikan dari fitur) | `0` s.d. `90` (`%`) | `40,21 ± 25,43` | `+1,0000` | Persentase keranjang belanja yang dibatalkan sebelum *checkout* tuntas |
| 4 | `checkout_abandonments_per_month` | `INTEGER` | **Kontrol Negatif E7** (Dikecualikan dari fitur) | `0` s.d. `10` (`11` nilai) | `5,00 ± 3,16` | `-0,0009` | Frekuensi pembatalan bulanan (berdistribusi acak seragam; digunakan sebagai *negative control*) |
| 5 | `days_since_last_purchase` *(Turunan)* | `INTEGER` | **Fitur Numerik #43** | `0` s.d. `730` hari | `365,0 ± 211,0` | `+0,0004` | Selisih hari dari `last_purchase_date` terhadap tanggal referensi maksimum dataset (`2027-01-01`) |
| 6 | `is_high_abandonment` *(Turunan)* | `INTEGER` (`0`/`1`) | **Target Klasifikasi Biner** | `0` (`494.531` / `49,45%`), `1` (`505.469` / `50,55%`) | `0,5055 ± 0,5000` | `+0,8653` | Label biner: `1` jika `cart_abandonment_rate >= 40.0`, dan `0` jika `< 40.0` |

---

## 2. Demografi & Latar Belakang Sosial-Ekonomi (14 Kolom Mentah: 4 Numerik + 10 Kategorikal)

| No | Nama Kolom | Tipe Data | Peran Fitur | Kardinalitas / Nilai Unik | Mean ± SD | Korelasi ($r$) | Deskripsi |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| 7 | `age` | `INTEGER` | Numerik | `18` s.d. `80` tahun (`63` nilai) | `49,00 ± 18,19` | `+0,0008` | Usia konsumen dalam tahun |
| 8 | `income_level` | `INTEGER` | Numerik | `10.000` s.d. `200.000` USD (`188.991` nilai) | `104.994,57 ± 54.851,48` | `-0,0019` | Pendapatan tahunan konsumen (numerik kontinu, **bukan** kategorikal) |
| 9 | `has_children` | `INTEGER` | Numerik (Biner) | `0` (Tidak), `1` (Ya) | `0,40 ± 0,49` | `-0,0013` | Indikator kepemilikan anak |
| 10 | `household_size` | `INTEGER` | Numerik | `1` s.d. `10` orang | `5,51 ± 2,87` | `+0,0021` | Jumlah anggota keluarga dalam satu rumah tangga |
| 11 | `gender` | `VARCHAR` | Kategorikal (`gender_encoded`) | `4` kategori: `Female`, `Male`, `Non-binary`, `Other` | — | — | Identitas gender konsumen |
| 12 | `country` | `VARCHAR` | Kategorikal (`country_encoded`) | `10` negara: `Australia`, `Brazil`, `Canada`, `China`, `France`, `Germany`, `India`, `Japan`, `UK`, `USA` | — | — | Negara domisili pengguna (~100.000 pengguna per negara) |
| 13 | `urban_rural` | `VARCHAR` | Kategorikal (`urban_rural_encoded`) | `3` kategori: `Urban`, `Suburban`, `Rural` | — | — | Klasifikasi wilayah tempat tinggal |
| 14 | `employment_status` | `VARCHAR` | Kategorikal (`employment_status_encoded`) | `5` kategori: `Employed`, `Self-Employed`, `Student`, `Retired`, `Unemployed` | — | — | Status pekerjaan konsumen |
| 15 | `education_level` | `VARCHAR` | Kategorikal (`education_level_encoded`) | `5` kategori: `High School`, `Associate Degree`, `Bachelor's`, `Master's`, `PhD` | — | — | Jenjang pendidikan terakhir |
| 16 | `relationship_status` | `VARCHAR` | Kategorikal (`relationship_status_encoded`) | `5` kategori: `Single`, `In a Relationship`, `Married`, `Divorced`, `Widowed` | — | — | Status pernikahan/hubungan |
| 17 | `occupation` | `VARCHAR` | Kategorikal (`occupation_encoded`) | `8` kategori profesi (`Education`, `Engineering`, `Finance`, `Healthcare`, `IT`, `Marketing`, `Other`, `Retail`) | — | — | Bidang pekerjaan utama |
| 18 | `ethnicity` | `VARCHAR` | Kategorikal (`ethnicity_encoded`) | `5` kategori etnis | — | — | Latar belakang etnis pengguna |
| 19 | `language_preference` | `VARCHAR` | Kategorikal (`language_preference_encoded`) | `6` bahasa (`English`, `Mandarin`, `Spanish`, `French`, `German`, `Hindi`) | — | — | Bahasa antarmuka pilihan pengguna |
| 20 | `device_type` | `VARCHAR` | Kategorikal (`device_type_encoded`) | `3` perangkat: `Mobile` (`60,02%`), `Desktop` (`29,98%`), `Tablet` (`9,99%`) | — | — | Perangkat utama untuk mengakses platform *e-commerce* |

---

## 3. Psikologis, Gaya Hidup, & Kesehatan Mental (16 Kolom Mentah: 15 Numerik + 1 Kategorikal)

| No | Nama Kolom | Tipe Data | Peran Fitur | Kardinalitas / Nilai Unik | Mean ± SD | Korelasi ($r$) | Deskripsi |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| 21 | `impulse_buying_score` | `INTEGER` | **Numerik (Klaster Kolinear #1)** | `0` s.d. `10` (`11` nilai) | `5,00 ± 3,22` | **`+0,9615`** | Skor kecenderungan belanja impulsif konsumen (prediktor terkuat pertama) |
| 22 | `stress_from_financial_decisions` | `INTEGER` | **Numerik (Klaster Kolinear #2)** | `0` s.d. `10` (`11` nilai) | `5,00 ± 3,17` | **`+0,9404`** | Tingkat kecemasan/stres saat menghadapi keputusan finansial (prediktor terkuat kedua) |
| 23 | `overall_stress_level` | `INTEGER` | **Numerik (Klaster Kolinear #3)** | `0` s.d. `10` (`11` nilai) | `5,00 ± 3,22` | **`+0,8635`** | Tingkat stres umum harian konsumen (kolinear kuat dengan stres finansial $r = +0,9162$) |
| 24 | `impulse_purchases_per_month` | `INTEGER` | **Numerik (Klaster Kolinear #4)** | `0` s.d. `7` (`8` nilai) | `3,29 ± 1,82` | **`+0,8501`** | Frekuensi pembelian impulsif per bulan (kolinear kuat dengan skor impulsif) |
| 25 | `budgeting_style` | `VARCHAR` | Kategorikal (`budgeting_style_encoded`) | `3` kategori: `Strict`, `Moderate`, `Loose` | — | — | Gaya pengelolaan anggaran rumah tangga |
| 26 | `brand_loyalty_score` | `INTEGER` | Numerik | `0` s.d. `10` (`11` nilai) | `5,00 ± 3,16` | `-0,0000` | Skor kesetiaan terhadap merek tertentu |
| 27 | `environmental_consciousness` | `INTEGER` | Numerik | `0` s.d. `10` (`11` nilai) | `5,01 ± 3,16` | `-0,0008` | Tingkat kepedulian terhadap produk ramah lingkungan |
| 28 | `health_conscious_shopping` | `INTEGER` | Numerik (Biner) | `0` atau `1` | `0,50 ± 0,50` | `-0,0014` | Indikator kebiasaan memilih produk berorientasi kesehatan |
| 29 | `travel_frequency` | `INTEGER` | Numerik | `0` s.d. `12` kali/tahun | `5,99 ± 3,74` | `+0,0002` | Frekuensi perjalanan per tahun |
| 30 | `hobby_count` | `INTEGER` | Numerik | `0` s.d. `5` hobi | `2,50 ± 1,71` | `-0,0004` | Jumlah hobi aktif yang dimiliki konsumen |
| 31 | `social_media_influence_score` | `INTEGER` | Numerik | `0` s.d. `10` (`11` nilai) | `5,00 ± 3,16` | `+0,0011` | Tingkat keterpengaruh belanja oleh konten media sosial |
| 32 | `reading_habits` | `INTEGER` | Numerik | `0` s.d. `24` buku/tahun (`25` nilai) | `12,00 ± 7,21` | `-0,0003` | Frekuensi membaca tahunan (numerik diskrit, sebelumnya belum dimasukkan ke vektor fitur) |
| 33 | `exercise_frequency` | `INTEGER` | Numerik | `0` s.d. `7` hari/minggu | `3,50 ± 2,29` | `+0,0027` | Frekuensi olahraga mingguan |
| 34 | `sleep_quality` | `INTEGER` | Numerik | `4` s.d. `9` (`6` nilai) | `6,50 ± 1,71` | `-0,0004` | Skor kualitas tidur harian |
| 35 | `physical_activity_level` | `INTEGER` | Numerik | `0` s.d. `10` (`11` nilai) | `4,99 ± 3,16` | `-0,0011` | Tingkat aktivitas fisik harian |
| 36 | `mental_health_score` | `INTEGER` | Numerik | `0` s.d. `10` (`11` nilai) | `5,00 ± 3,16` | `-0,0001` | Indeks kesejahteraan mental umum |

---

## 4. Kebiasaan Transaksi, Pembayaran, & Loyalitas (14 Kolom Mentah: 11 Numerik + 3 Kategorikal)

| No | Nama Kolom | Tipe Data | Peran Fitur | Kardinalitas / Nilai Unik | Mean ± SD | Korelasi ($r$) | Deskripsi |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| 37 | `weekly_purchases` | `INTEGER` | Numerik | `0` s.d. `20` transaksi | `9,99 ± 6,06` | `+0,0001` | Frekuensi pembelian rata-rata per minggu |
| 38 | `monthly_spend` | `INTEGER` | Numerik | `0` s.d. `5.000` USD | `2.498,78 ± 1.444,21` | `-0,0019` | Total pengeluaran belanja bulanan (USD) |
| 39 | `average_order_value` | `INTEGER` | Numerik | `10` s.d. `500` USD | `255,03 ± 141,71` | `-0,0008` | Nilai rata-rata per pesanan (*Average Order Value* / AOV) |
| 40 | `coupon_usage_frequency` | `INTEGER` | Numerik | `0` s.d. `4` (`5` nilai) | `2,00 ± 1,42` | `+0,0022` | Tingkat pemakaian kupon/kode promo |
| 41 | `loyalty_program_member` | `INTEGER` | Numerik (Biner) | `0` atau `1` | `0,50 ± 0,50` | `-0,0003` | Keanggotaan program loyalitas toko |
| 42 | `referral_count` | `INTEGER` | Numerik | `0` s.d. `10` referral | `5,00 ± 3,16` | `-0,0004` | Jumlah pengguna baru yang berhasil direferensikan |
| 43 | `weekend_shopper` | `INTEGER` | Numerik (Biner) | `0` atau `1` | `0,50 ± 0,50` | `+0,0010` | Preferensi berbelanja di akhir pekan |
| 44 | `browse_to_buy_ratio` | `INTEGER` | Numerik | `10` s.d. `100` | `55,00 ± 26,27` | `+0,0003` | Rasio kunjungan penjelajahan terhadap pembelian aktual |
| 45 | `return_frequency` | `INTEGER` | Numerik | `0` s.d. `12` kali | `6,00 ± 3,74` | `+0,0002` | Frekuensi pengembalian barang (*return*) |
| 46 | `return_rate` | `INTEGER` | Numerik | `0` s.d. `100` (`%`) | `50,00 ± 29,16` | `-0,0005` | Persentase barang yang diretur |
| 47 | `premium_subscription` | `INTEGER` | Numerik (Biner) | `0` (`64,06%`), `1` (`35,94%`) | `0,36 ± 0,48` | `-0,0008` | Status langganan layanan premium (seperti Amazon Prime) |
| 48 | `preferred_payment_method` | `VARCHAR` | Kategorikal (`preferred_payment_method_encoded`) | `6` metode: `Apple Pay`, `Bank Transfer`, `Credit Card`, `Debit Card`, `Google Pay`, `PayPal` | — | — | Metode pembayaran utama pilihan konsumen |
| 49 | `product_category_preference` | `VARCHAR` | Kategorikal (`product_category_preference_encoded`) | `8` kategori (`Apparel`, `Beauty`, `Books`, `Electronics`, `Groceries`, `Home & Kitchen`, `Sports`, `Toys`) | — | — | Kategori produk favorit konsumen |
| 50 | `shopping_time_of_day` | `VARCHAR` | Kategorikal (`shopping_time_of_day_encoded`) | `4` waktu: `Morning`, `Afternoon`, `Evening`, `Night` | — | — | Waktu berbelanja paling aktif dalam sehari |

---

## 5. Interaksi Aplikasi & Engagement Digital (12 Kolom Mentah: Seluruhnya Numerik)

| No | Nama Kolom | Tipe Data | Peran Fitur | Kardinalitas / Nilai Unik | Mean ± SD | Korelasi ($r$) | Deskripsi |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| 51 | `daily_session_time_minutes` | `INTEGER` | Numerik | `0` s.d. `120` menit | `60,01 ± 34,94` | `-0,0003` | Durasi harian penggunaan aplikasi/situs *e-commerce* |
| 52 | `product_views_per_day` | `INTEGER` | Numerik | `0` s.d. `50` produk | `25,02 ± 14,72` | `+0,0010` | Jumlah halaman detail produk yang dilihat per hari |
| 53 | `ad_views_per_day` | `INTEGER` | Numerik | `0` s.d. `20` tayangan | `10,00 ± 6,06` | `-0,0015` | Jumlah impresi iklan harian |
| 54 | `ad_clicks_per_day` | `INTEGER` | Numerik | `0` s.d. `5` klik | `2,50 ± 1,71` | `-0,0015` | Jumlah klik iklan harian |
| 55 | `wishlist_items_count` | `INTEGER` | Numerik | `0` s.d. `20` item | `10,00 ± 6,05` | `-0,0019` | Jumlah barang yang disimpan di *wishlist* |
| 56 | `cart_items_average` | `INTEGER` | Numerik | `1` s.d. `10` item | `5,49 ± 2,87` | `+0,0004` | Rata-rata jumlah item di dalam keranjang belanja |
| 57 | `purchase_conversion_rate` | `INTEGER` | Numerik | `0` s.d. `100` (`%`) | `50,00 ± 29,16` | `+0,0007` | Rasio konversi kunjungan menjadi transaksi |
| 58 | `app_usage_frequency` | `INTEGER` | Numerik | `0` s.d. `7` hari/minggu (`8` nilai) | `3,50 ± 2,29` | `+0,0005` | Frekuensi penggunaan aplikasi *mobile* per minggu |
| 59 | `notification_response_rate` | `INTEGER` | Numerik | `0` s.d. `100` (`%`) | `49,99 ± 29,15` | `+0,0027` | Persentase respons terhadap *push notification* promosi |
| 60 | `account_age_months` | `INTEGER` | Numerik | `1` s.d. `24` bulan | `12,51 ± 6,92` | `+0,0004` | Lama usia akun terdaftar (dalam bulan) |
| 61 | `social_sharing_frequency` | `INTEGER` | Numerik | `0` s.d. `8` kali/bulan | `3,59 ± 1,93` | `-0,0004` | Frekuensi membagikan tautan produk ke media sosial |
| 62 | `review_writing_frequency` | `INTEGER` | Numerik | `0` s.d. `8` ulasan/bulan (`9` nilai) | `3,45 ± 1,86` | `-0,0007` | Frekuensi menulis ulasan produk |
