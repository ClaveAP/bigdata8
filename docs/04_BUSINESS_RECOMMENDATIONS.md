# Business Recommendations, IF-THEN Rules, & Persona Strategy

Dokumen ini memetakan temuan empiris dari *pipeline* **PySpark MLlib** (`1.000.000` baris pengguna $\times$ `57` fitur prediktor independen), analisis **SHAP & Permutation Importance**, ekstraksi **IF-THEN Rules** terhumanisasi, serta segmentasi **K-Means Clustering** menjadi strategi intervensi bisnis operasional untuk menekan tingkat pembatalan keranjang (*Cart Abandonment*).

---

## 1. Ringkasan Eksekutif & Temuan Inti Model

Berdasarkan audit statistik populasi penuh ($N = 1.000.000$) dan evaluasi multi-model pada data uji ($N_{\text{test}} = 200.277$), diperoleh tiga kesimpulan utama:

1. **Klaster 4 Faktor Psikologis Kolinear sebagai Pendorong Tunggal Pembatalan**:
   - Pembatalan keranjang (`cart_abandonment_rate`) digerakkan oleh 4 variabel psikologis yang saling berkolinearitas tinggi:
     - `impulse_buying_score` ($r = +0,9615$; *DT Gini Importance* = **95,00%**, *RF Ensemble Gini (100 Pohon Audit)* = **35,14%**, *Mean* $|\text{SHAP}|_{\text{RF}}$ = **0,1804**)
     - `stress_from_financial_decisions` ($r = +0,9404$; *DT Gini Importance* = **4,72%**, *RF Ensemble Gini (100 Pohon Audit)* = **31,15%**, *Mean* $|\text{SHAP}|_{\text{RF}}$ = **0,1468**)
     - `overall_stress_level` ($r = +0,8635$; *RF Ensemble Gini (100 Pohon Audit)* = **17,08%**, *Mean* $|\text{SHAP}|_{\text{RF}}$ = **0,0523**)
     - `impulse_purchases_per_month` ($r = +0,8501$; *RF Ensemble Gini (100 Pohon Audit)* = **14,04%**, *Mean* $|\text{SHAP}|_{\text{RF}}$ = **0,0528**)
   - Pengacakan serentak (*Joint Cluster Permutation*) keempat fitur ini menjatuhkan $\text{ROC-AUC}$ sebesar **$-0,4835$ s.d. $-0,4899$** (menuju tebakan acak $\approx 0,50$), membuktikan bahwa **100% sinyal prediktif pembatalan berada pada klaster friksi psikologis ini**.
2. **Bukan Masalah Daya Beli atau Kendala Teknis Sesi**:
   - Kelompok pembatal tinggi (*High Abandonment*, $N = 505.469$) dan kelompok pembeli stabil (*Low Abandonment*, $N = 494.531$) memiliki pengeluaran bulanan yang identik (**\$2.496,58** vs **\$2.501,02**), nilai pesanan rata-rata (AOV) yang identik (**\$255,01** vs **\$255,05**), dan durasi sesi harian yang sama (**60,01 menit** vs **60,02 menit**).
3. **Mekanisme Perilaku (*Buyer's Remorse Friction*)**:
   - Konsumen impulsif sangat reaktif memasukkan barang ke keranjang saat melihat promo, namun ketika dihadapkan pada total tagihan akhir di laman *checkout*, lonjakan stres keputusan finansial memicu penyesalan seketika (*buyer's remorse*) dan penutupan aplikasi.

---

## 2. Evaluasi Empiris Aturan Keputusan 3-Tier (IF-THEN Rules) pada Data Uji ($N = 200.277$)

Pohon keputusan ringkas (`maxDepth=3`, hanya `7` simpul / `4` aturan daun) terbukti memiliki tingkat kesepakatan prediksi (**Fidelity**) sebesar **98,28%** (`196.825` dari `200.277` pengguna uji) terhadap *Champion Decision Tree* penuh (`maxDepth=10`, `473` simpul), dengan retensi akurasi **100,08%** (`94,06%` vs `93,99%`).

Aturan percabangan pohon tersebut menghasilkan **3 Tier Operasional** yang dapat ditanamkan langsung ke mesin *checkout real-time* tanpa biaya komputasi inferensi kompleks:

| Tier Operasional | Aturan Logika Eksplisit (*IF-THEN Condition*) | Populasi Uji ($N = 200.277$) | Proporsi Cakupan | Aktual *High Abandonment Rate* | Rata-rata `cart_abandonment_rate` | Tindakan Intervensi Otomatis (*Real-Time Action*) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: Risiko Kritis** | `IF impulse_buying_score > 5.5` **ATAU** `(impulse_buying_score == 5 AND stress_from_financial_decisions > 5.5)` | `98.341` | **49,10%** | **95,49%** | **62,25%** | Tampilkan simulasi **Cicilan 0% / BNPL** langsung di halaman keranjang & tombol **Split-Cart ("Beli 1 Sekarang, Simpan Sisanya di Wishlist")** |
| **Tier 2: Risiko Ambang (Moderat)** | `IF 4.5 < impulse_buying_score <= 5.5` **DAN** `stress_from_financial_decisions <= 5.5` | `10.943` | **5,46%** | **37,57%** | **36,95%** | Aktifkan pemantauan *idle* (`> 90 detik`), tampilkan lencana **Free Return Guarantee**, & pengingat keranjang non-intrusif |
| **Tier 3: Risiko Rendah (Aman)** | `IF impulse_buying_score <= 4.5` | `90.993` | **45,43%** | **3,67%** | **16,88%** | Sediakan jalur **1-Click Express Checkout**; **JANGAN** berikan pop-up diskon darurat agar margin keuntungan tidak tergerus (*anti-cannibalization*) |

---

## 3. Rincian 4 Program Intervensi Bisnis & UI/UX

### Rekomendasi 1: Dekompresi Stres Tagihan melalui Simulasi Cicilan 0% / BNPL di Keranjang (Target: Tier 1)
- **Masalah**: Pengguna Tier 1 memiliki rata-rata skor impulsif **7,82/10** dan stres finansial **7,71/10**, dengan risiko batal **95,49%**.
- **Tindakan UI/UX**:
  - Pindahkan tampilan simulasi **Buy Now Pay Later (BNPL) / Cicilan 0%** dari halaman akhir pembayaran ke **halaman tinjauan keranjang belanja (*Cart Review*)**, khususnya pada antarmuka **Mobile (60,02% pengguna)**.
  - Tampilkan nominal cicilan per bulan berdampingan dengan harga total untuk menurunkan persepsi beban kas keluar (*out-of-pocket shock*).

### Rekomendasi 2: Fitur Satu-Klik *Partial Checkout / Split-Cart* (Target: Tier 1)
- **Masalah**: Saat pengguna impulsif mengumpulkan rata-rata `5,50` item di keranjang (senilai ~\$255), rasa cemas membuat mereka membatalkan **seluruh** isi keranjang.
- **Tindakan UI/UX**:
  - Sediakan tombol **"Checkout Item Prioritas Sekarang & Pindahkan Sisanya ke Wishlist"** dengan jaminan harga terkunci (*Price-Lock 48 Jam*).
  - Menyelamatkan minimal 1–2 item per transaksi jauh lebih menguntungkan dibandingkan kehilangan 100% keranjang.

### Rekomendasi 3: *Reassurance Badge* & Intervensi Anti-Idle (Target: Tier 2 — Zona Ambang)
- **Masalah**: Pengguna Tier 2 (`5,46%` populasi, skor impulsif `5,0`, stres finansial `4,0`) berada tepat di ambang keraguan (`37,57%` *high abandonment*).
- **Tindakan UI/UX**:
  - Tampilkan lencana **"Jaminan 100% Gratis Pengembalian Barang (Free Return)"** dan rincian ongkos kirim transparan sejak awal (*Zero Hidden Fee*).
  - Hanya picu *voucher* gratis ongkir mikro apabila terdeteksi ketidakaktifan layar (*idle*) `> 90 detik`.

### Rekomendasi 4: Proteksi Margin melalui *1-Click Express Checkout* (Target: Tier 3)
- **Masalah**: Pemberian kupon diskon secara membabi buta kepada seluruh pengguna akan memotong margin pada `45,43%` pengguna yang sebenarnya sudah aman (hanya `3,67%` risiko batal tinggi).
- **Tindakan Bisnis**:
  - Hilangkan seluruh *pop-up* intervensi pada pengguna Tier 3 dan arahkan langsung ke **1-Click Instant Checkout** menggunakan metode pembayaran tersimpan (`Apple Pay`, `Google Pay`, `Credit Card`, `PayPal`).

---

## 4. Profil Persona Pelanggan: Komparasi Kelas Target & Segmentasi K-Means ($N = 1.000.000$)

### A. Komparasi Statistik Berdasarkan Kelas Target (`is_high_abandonment`)

| Karakteristik Metrik | Segmen High Abandonment (`Class 1`, $N = 505.469$) | Segmen Low Abandonment (`Class 0`, $N = 494.531$) | Kontras & Implikasi Bisnis |
| :--- | :---: | :---: | :--- |
| **`impulse_buying_score` (0–10)** | **7,68** | **2,26** | **3,40x lebih tinggi** pada kelompok pembatal |
| **`stress_from_financial_decisions` (0–10)** | **7,59** | **2,35** | **3,23x lebih tinggi** pada kelompok pembatal |
| **`overall_stress_level` (0–10)** | **7,45** | **2,50** | **2,98x lebih tinggi** pada kelompok pembatal |
| **`impulse_purchases_per_month` (0–7)** | **4,63 kali** | **1,92 kali** | **2,41x lebih sering** belanja impulsif |
| **Rata-rata Belanja Bulanan (`monthly_spend`)** | **\$2.496,58** | **\$2.501,02** | **Identik** ($\Delta = 0,18\%$, bukan faktor daya beli) |
| **Nilai Pesanan Rata-rata (`average_order_value`)** | **\$255,01** | **\$255,05** | **Identik** ($\Delta = 0,02\%$) |
| **Rata-rata Isi Keranjang (`cart_items_average`)** | **5,50 item** | **5,49 item** | **Identik** |
| **Durasi Sesi Harian (`daily_session_time_minutes`)** | **60,01 menit** | **60,02 menit** | **Identik** (sama-sama aktif menjelajah ~1 jam/hari) |

### B. Segmentasi Unsupervised PySpark K-Means Clustering (`src/clustering.py`)

Evaluasi *K-Means Clustering* pada 10 fitur perilaku dan psikologis terstandarisasi ($Z$-score) di seluruh `1.000.000` pengguna menunjukkan:
- **Struktur Alami Terbaik ($k=2$, *Silhouette Score* = `0,3921`, WSSSE = `7.306.470,89`)**: Memisahkan populasi secara simetris menjadi **Klaster Belanja Terencana (`50,11%`, rata-rata pembatalan `18,91%`)** dan **Klaster Impulsif-Cemas (`49,89%`, rata-rata pembatalan `61,60%`)**.
- **Segmentasi Operasional 3-Persona ($k=3$, *Silhouette Score* = `0,2653`, WSSSE = `6.911.854,61`)**:
  1. **Persona A — *Mindful & Low-Stress Shoppers*** (`469.105` pengguna / **`46,91%`**): Skor impulsif `2,13`, stres finansial `2,16`, rata-rata `cart_abandonment_rate` = **`17,57%`** (hanya `5,18%` berisiko tinggi).
  2. **Persona B — *High-Spend Impulsive Shoppers*** (`264.941` pengguna / **`26,49%`**): Skor impulsif `7,52`, stres finansial `7,49`, pengeluaran bulanan tertinggi (`$2.506,10`), rata-rata `cart_abandonment_rate` = **`60,08%`** (`90,45%` berisiko tinggi).
  3. **Persona C — *Conversion-Active Impulsive Browsers*** (`265.954` pengguna / **`26,60%`**): Skor impulsif `7,55`, stres finansial `7,53`, konversi historis `50,20%`, rata-rata `cart_abandonment_rate` = **`60,36%`** (`90,82%` berisiko tinggi).

---

## 5. Matriks Target KPI Implementasi

| Program Intervensi | Segmen Sasaran | Estimasi Penurunan *Abandonment* | Indikator Kinerja Utama (KPI) |
| :--- | :--- | :---: | :--- |
| **Promosi BNPL / Cicilan 0% di Keranjang Mobile** | Tier 1 (`49,10%` pengguna) | Turun `8% – 12%` | *BNPL Adoption Rate* & *Checkout Completion Rate* |
| **Fitur Satu-Klik Split-Cart ke Wishlist** | Tier 1 dengan `cart_items >= 5` | Turun `10% – 15%` | *Partial Cart Conversion Rate* |
| **Free Return Badge & Anti-Idle Trigger (>90s)** | Tier 2 (`5,46%` pengguna ambang) | Turun `5% – 8%` | Penurunan *Checkout Exit Rate* |
| **1-Click Express Checkout Tanpa Diskon** | Tier 3 (`45,43%` pengguna aman) | Penghematan biaya promo `100%` pada segmen aman | *Gross Margin Preservation* & *Checkout Latency* |
