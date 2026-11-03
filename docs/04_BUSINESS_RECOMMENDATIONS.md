# Business Recommendations & Actionable Strategy

Dokumen ini memetakan temuan analisis teknis (Feature Importance / SHAP values dan Klaster Pengguna) menjadi strategi bisnis operasional untuk menurunkan angka *checkout abandonment*.

## Ringkasan Eksekutif
*(Akan disintesis pasca-pelatihan model & interpretasi SHAP)*

---

## 1. Analisis Faktor Pendorong Utama (Model Insights)
- **Top Pemicu Pembatalan**: *(Diisi setelah SHAP summary plot dibuat)*
- **Hubungan Stres Finansial & Kupon**: Analisis elastisitas harga dan diskon terhadap keranjang belanja.
- **Pengaruh Perangkat (UX Friction)**: Temuan apakah pengguna mobile mengalami tingkat pembatalan lebih tinggi dibanding desktop.

---

## 2. Strategi Berdasarkan Persona / Klaster Pelanggan

### Klaster 1: *(Nama Persona TBD)*
- **Karakteristik**: ...
- **Rekomendasi Strategi**:
  - Terapkan reminder notifikasi terpersonalisasi.
  - Tawarkan opsi pembayaran fleksibel (misal: Cicilan 0% / Buy Now Pay Later).

### Klaster 2: *(Nama Persona TBD)*
- **Karakteristik**: ...
- **Rekomendasi Strategi**:
  - Dynamic checkout discount saat pengguna idle di halaman pembayaran selama > 2 menit.
