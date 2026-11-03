# Data Dictionary: E-Commerce Shopper Behavior & Lifestyle

Dataset ini memiliki 57 fitur yang dikelompokkan ke dalam 4 kategori besar:

## 1. Demografi & Latar Belakang (15 Fitur)
| Nama Fitur | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `user_id` | String | Identitas unik pengguna |
| `age` | Integer | Usia pengguna |
| `gender` | Categorical | Jenis kelamin pengguna |
| `country` | Categorical | Negara domisili pengguna |
| `urban_rural` | Categorical | Klasifikasi tempat tinggal (Urban / Rural) |
| `income_level` | Categorical | Tingkat pendapatan (Low, Medium, High, dll.) |
| `employment_status` | Categorical | Status pekerjaan (Employed, Self-Employed, Unemployed, dll.) |
| `education_level` | Categorical | Jenjang pendidikan terakhir |
| `relationship_status` | Categorical | Status perkawinan/hubungan |
| `has_children` | Boolean/Binary | Indikator kepemilikan anak |
| `household_size` | Integer | Jumlah anggota keluarga dalam satu rumah tangga |
| `occupation` | Categorical | Jenis pekerjaan/profesi |
| `ethnicity` | Categorical | Latar belakang etnis |
| `language_preference` | Categorical | Bahasa preferensi di platform |
| `device_type` | Categorical | Perangkat utama belanja (Mobile, Desktop, Tablet) |

## 2. Gaya Hidup & Finansial (15 Fitur)
| Nama Fitur | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `budgeting_style` | Categorical | Gaya pengelolaan anggaran belanja |
| `brand_loyalty_score` | Float | Skor loyalitas terhadap merek tertentu |
| `impulse_buying_score` | Float | Kecenderungan berbelanja secara impulsif |
| `environmental_consciousness` | Float | Tingkat kepedulian terhadap isu lingkungan |
| `health_conscious_shopping` | Float | Indeks pemilihan produk kesehatan |
| `travel_frequency` | Categorical/Int | Frekuensi bepergian per tahun |
| `hobby_count` | Integer | Jumlah hobi yang aktif ditekuni |
| `social_media_influence_score` | Float | Skor pengaruh paparan media sosial terhadap belanja |
| `reading_habits` | Categorical | Kebiasaan membaca |
| `exercise_frequency` | Categorical/Int | Frekuensi olahraga mingguan |
| `stress_from_financial_decisions`| Float | Tingkat stres saat mengambil keputusan finansial |
| `overall_stress_level` | Float | Tingkat stres umum pengguna |
| `sleep_quality` | Float | Kualitas tidur pengguna |
| `physical_activity_level` | Float | Tingkat keaktifan fisik |
| `mental_health_score` | Float | Indikator kondisi kesehatan mental |

## 3. Kebiasaan Berbelanja (14 Fitur)
| Nama Fitur | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `weekly_purchases` | Integer | Jumlah rata-rata transaksi per minggu |
| `monthly_spend` | Float | Total rata-rata pengeluaran bulanan |
| `average_order_value` | Float | Nilai rata-rata per transaksi pesanan (AOV) |
| `preferred_payment_method` | Categorical | Metode pembayaran favorit (Credit Card, E-Wallet, COD, dll.) |
| `coupon_usage_frequency` | Categorical/Int | Seberapa sering menggunakan kupon/diskon |
| `loyalty_program_member` | Boolean/Binary | Status keanggotaan program loyalitas |
| `referral_count` | Integer | Jumlah pengguna baru yang direferensikan |
| `product_category_preference` | Categorical | Kategori produk yang paling sering dibeli |
| `shopping_time_of_day` | Categorical | Waktu favorit berbelanja (Pagi, Siang, Malam, Dini hari) |
| `weekend_shopper` | Boolean/Binary | Apakah lebih aktif belanja di akhir pekan |
| `impulse_purchases_per_month` | Integer | Frekuensi pembelian impulsif dalam sebulan |
| `browse_to_buy_ratio` | Float | Rasio lama menjelajah dibandingkan transaksi nyata |
| `return_frequency` | Integer | Frekuensi pengembalian barang |
| `return_rate` | Float | Persentase barang yang diretur dari total pembelian |

## 4. Penggunaan Platform & Pembatalan (13 Fitur)
| Nama Fitur | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `daily_session_time_minutes` | Float | Durasi harian membuka aplikasi/web (menit) |
| `product_views_per_day` | Integer | Jumlah produk yang dilihat per hari |
| `ad_views_per_day` | Integer | Jumlah tayangan iklan yang dilihat |
| `ad_clicks_per_day` | Integer | Jumlah iklan yang diklik |
| `wishlist_items_count` | Integer | Jumlah item tersimpan di wishlist |
| `cart_items_average` | Float | Rata-rata jumlah barang yang dimasukkan ke keranjang |
| `checkout_abandonments_per_month`| Integer | **[TARGET]** Frekuensi pembatalan transaksi di tahap akhir per bulan |
| `cart_abandonment_rate` | Float | **[TARGET]** Rasio pembatalan keranjang belanja |
| `purchase_conversion_rate` | Float | Rasio konversi transaksi sukses |
| `app_usage_frequency` | Categorical | Frekuensi membuka aplikasi per hari/minggu |
| `notification_response_rate` | Float | Persentase respon terhadap notifikasi/push alert |
| `account_age_months` | Integer | Usia akun pengguna dalam bulan |
| `social_sharing_frequency` | Float | Seberapa sering membagikan tautan produk |
