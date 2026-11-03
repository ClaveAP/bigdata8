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
