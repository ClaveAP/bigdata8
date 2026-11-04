# Business Recommendations & Actionable Strategy

Dokumen ini memetakan temuan analisis teknis (Feature Importance Decision Tree dan Pola Perilaku Belanja) menjadi strategi intervensi bisnis konkret untuk menekan tingkat pembatalan keranjang (*Cart Abandonment*) pada platform e-commerce.

---

## 1. Ringkasan Eksekutif & Temuan Inti Model
Dari pelatihan model terdistribusi PySpark terhadap 1.000.000 baris data profil konsumen, ditemukan bahwa pembatalan transaksi di tahap akhir didominasi secara mutlak oleh 2 faktor psikologis utama:
1. **Tingkat Belanja Impulsif (`impulse_buying_score`)** menyumbang **94.99%** pengaruh pembatalan.
2. **Tingkat Stres Keputusan Finansial (`stress_from_financial_decisions`)** menyumbang **4.72%** pengaruh pembatalan.

### Interpretasi Perilaku Konsumen:
Konsumen e-commerce sangat mudah terpicu dorongan emosional saat melihat iklan atau promo (*impulse browsing*), sehingga memasukkan banyak barang ke dalam keranjang. Namun, ketika mereka tiba di tahap *checkout/cart review* dan melihat akumulasi total harga yang harus dibayar saat itu juga, muncul rasa cemas finansial (*financial anxiety & buyer's remorse*) yang menyebabkan mereka langsung menutup aplikasi tanpa menyelesaikan transaksi.

---

## 2. Strategi Intervensi Bisnis Berdasarkan Aturan IF-THEN Model

### Rekomendasi 1: Meredakan *Stres Finansial* dengan Opsi Cicilan Fleksibel (BNPL)
- **Target Aturan**: `If impulse_buying_score > 5.5 AND stress_from_financial_decisions > 4.5`
- **Tindakan Bisnis**:
  - Tampilkan opsi **Buy Now Pay Later (BNPL)** atau **Cicilan 0%** langsung di halaman rincian keranjang belanja (bukan disembunyikan di tahap akhir pembayaran).
  - Tampilkan simulasi cicilan ringan (contoh: *"Bisa dicicil Rp75.000/bulan"* daripada hanya menampilkan total Rp900.000). Hal ini secara psikologis mereduksi beban stres finansial seketika.

### Rekomendasi 2: Fitur "Simpan untuk Nanti" (*Save for Later / Wishlist Split*)
- **Target Aturan**: Pengguna impulsif dengan keranjang belanja bernilai tinggi.
- **Tindakan Bisnis**:
  - Daripada membiarkan konsumen membatalkan seluruh keranjang belanja, sediakan tombol satu-klik *"Beli Barang Utama Sekarang, Simpan Sisanya untuk Nanti"*.
  - Strategi ini memecah transaksi impulsif besar menjadi transaksi yang terjangkau secara psikologis, sehingga angka konversi tetap tercapai.

### Rekomendasi 3: Countdown Diskon Dinamis Anti-Idle (*Dynamic Exit-Intent Incentive*)
- **Target Aturan**: Sesi keranjang yang diam (*idle*) lebih dari 90 detik pada pengguna terindikasi risiko tinggi.
- **Tindakan Bisnis**:
  - Ketika kursor pengguna mengarah keluar jendela web atau aplikasi tidak aktif di laman checkout, luncurkan *micro-incentive* (misalnya: *"Voucher Gratis Ongkir Khusus 10 Menit Ini"*).
  - Insentif ini memvalidasi keputusan belanja impulsif mereka sebagai keputusan yang menguntungkan dan hemat, menetralkan rasa penyesalan belanja.

### Rekomendasi 4: Transparansi Biaya Sejak Awal (*Zero Hidden Fee Guarantee*)
- **Faktor Pendukung**: `preferred_payment_method` dan `product_category_preference`.
- **Tindakan Bisnis**:
  - Tampilkan estimasi ongkos kirim dan pajak secara transparan sejak barang dimasukkan ke keranjang. Lonjakan harga tiba-tiba di tahap akhir adalah pemicu terbesar stres finansial konsumen impulsif.

---

## 3. Matriks KPI Keberhasilan Implementasi

| Program Intervensi | Target Sasaran | Target Penurunan Abandonment | Metrik Keberhasilan (KPI) |
| :--- | :--- | :---: | :--- |
| **Integrasi Promosi BNPL / Cicilan 0%** | Pengguna dengan Stres Finansial $\ge 5$ | Turun 8 - 12% | Tingkat adopsi metode pembayaran BNPL |
| **Fitur Save-for-Later Satu Klik** | Pengguna Impulsif $\ge 6$ | Turun 10 - 15% | Peningkatan *Completed Checkout Rate* |
| **Exit-Intent Micro-Discount (Gratis Ongkir)** | Pengguna Idle di Checkout | Turun 5 - 7% | *Redemption Rate* voucher checkout darurat |
| **Transparansi Biaya & Estimasi Ongkir** | Seluruh Segmen Pengguna | Turun 6 - 9% | Penurunan rasio *bounce rate* di halaman checkout |

---

## 4. Profil Persona Pelanggan Berbasis Data Skala Besar (1.000.000 Baris)

Berdasarkan komparasi data statistik riil antara dua segmen pengguna, ditemukan profil perilaku kontras:

| Karakteristik Metrik | Persona 1: Konsumen Risiko Batal (Class 1) | Persona 2: Konsumen Loyal / Stabil (Class 0) | Analisis Kontras Perilaku |
| :--- | :---: | :---: | :--- |
| **Skor Belanja Impulsif** | **7,68 / 10** | **2,26 / 10** | **3,4x Lebih Tinggi** pada kelompok pembatal |
| **Stres Keputusan Finansial** | **7,59 / 10** | **2,35 / 10** | **3,2x Lebih Tinggi** pada kelompok pembatal |
| **Rata-rata Belanja Bulanan** | \$2.496,58 | \$2.501,02 | **Hampir Identik** (Bukan masalah daya beli) |
| **Average Order Value (AOV)** | \$255,01 | \$255,05 | **Identik** (Besar belanjaan per transaksi sama) |
| **Rata-rata Isi Keranjang** | 5,50 item | 5,49 item | **Identik** (Jumlah barang per transaksi sama) |
| **Durasi Sesi Aplikasi** | 60,01 menit | 60,02 menit | **Identik** (Waktu browsing sama-sama 1 jam) |

### 💡 Temuan Kunci untuk Manajemen E-Commerce:
1. **Bukan Masalah Kemampuan Finansial (Daya Beli)**: Kedua kelompok memiliki anggaran belanja bulanan yang sama persis (\$2.500) dan nilai keranjang yang sama (\$255).
2. **Bukan Masalah Desain Katalog atau Waktu Browsing**: Keduanya sama-sama meluangkan waktu 1 jam menjelajahi platform.
3. **Penyebab Utama Murni adalah Friksi Mental (*Buyer's Remorse Friction*)**: Konsumen impulsif mengalami lonjakan kecemasan di halaman pembayaran. Strategi intervensi harus difokuskan pada **reassurance (menenangkan kecemasan pembeli)** melalui fasilitas cicilan ringan dan pemecahan keranjang belanja (*Split Cart*).
