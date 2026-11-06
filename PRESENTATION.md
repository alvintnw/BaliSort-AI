# Materi Presentasi BaliSort AI

## 1. Judul

**BaliSort AI: Smart Waste Recognition & Sorting Recommendation**

Membantu masyarakat mengenali kategori sampah dari gambar dan memahami cara
memilahnya dengan lebih baik.

## 2. Masalah

Pemilahan sampah sering terhambat oleh:

- Kurangnya pengetahuan tentang kategori sampah.
- Sampah yang tercampur sejak dari sumbernya.
- Kesulitan membedakan material yang dapat didaur ulang dan residu.
- Kebutuhan akan edukasi pemilahan yang sederhana dan mudah diakses.

## 3. Solusi

BaliSort AI menyediakan aplikasi web sederhana yang:

1. Menerima gambar sampah dari pengguna.
2. Mengklasifikasikan gambar ke dalam enam kategori.
3. Menampilkan confidence score.
4. Memberi peringatan jika hasil prediksi kurang yakin.
5. Memberikan rekomendasi wadah dan cara pemilahan.
6. Menyimpan riwayat prediksi secara lokal.

## 4. Kategori sampah

Model mengenali:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash / Residual

## 5. Arsitektur teknis

```text
Gambar pengguna
      |
      v
Validasi file dan konversi RGB
      |
      v
Preprocessing 224 x 224 + normalisasi ImageNet
      |
      v
MobileNetV3-Large (PyTorch)
      |
      v
Kategori + confidence score
      |
      +--> Rekomendasi pemilahan
      +--> Riwayat SQLite
```

Komponen utama:

- **Frontend:** Streamlit.
- **Model:** MobileNetV3-Large dengan PyTorch.
- **Input:** gambar RGB 224 × 224 piksel.
- **Dataset:** pembagian train 70%, validation 15%, test 15%.
- **Database:** SQLite untuk riwayat lokal.
- **Inference:** GPU CUDA jika tersedia, CPU sebagai fallback.

## 6. Training dan evaluasi

Konfigurasi training:

- AdamW, learning rate `0.0001`.
- Batch size 32.
- Class weights berdasarkan distribusi data training.
- Mixed precision saat CUDA tersedia.
- Maksimal 20 epoch.
- Early stopping patience 5.
- Random seed 42.

Dataset test tidak digunakan untuk training atau pemilihan model.

Hasil evaluasi test set saat ini:

| Metrik | Hasil |
|---|---:|
| Accuracy | 91,67% |
| Macro precision | 91,07% |
| Macro recall | 91,14% |
| Macro F1 | 91,06% |

Hasil berlaku untuk checkpoint dan dataset yang digunakan saat evaluasi, bukan
jaminan akurasi untuk seluruh kondisi gambar di dunia nyata.

## 7. Alur demo

1. Buka aplikasi Streamlit.
2. Unggah foto sampah JPG, JPEG, atau PNG.
3. Tunjukkan preview gambar.
4. Tampilkan kategori dan confidence.
5. Jelaskan rekomendasi wadah dan tindakan pemilahan.
6. Buka probabilitas semua kategori.
7. Simpan hasil ke riwayat.
8. Tunjukkan tab edukasi enam kategori.

Gunakan contoh gambar yang:

- Objeknya terlihat jelas.
- Tidak terlalu gelap atau buram.
- Memiliki satu objek utama.
- Mewakili beberapa kategori berbeda.

## 8. Nilai inovasi dan dampak

- Mengubah klasifikasi visual menjadi tindakan pemilahan yang praktis.
- Menggabungkan AI, edukasi lingkungan, dan antarmuka yang mudah digunakan.
- Relevan untuk peningkatan kesadaran pengelolaan sampah di Bali.
- Dapat dikembangkan untuk bank sampah, sekolah, komunitas, dan rumah tangga.

## 9. Keterbatasan

- Model hanya mengenali enam kategori.
- Gambar yang ambigu, tercampur, atau di luar distribusi dataset dapat salah
  diklasifikasikan.
- Confidence score bukan jaminan kebenaran.
- Sampah berbahaya tetap memerlukan pemeriksaan dan prosedur khusus.
- Riwayat saat ini tersimpan pada SQLite lokal.

## 10. Pengembangan berikutnya

- Feedback pengguna untuk mengumpulkan prediksi yang keliru.
- Dataset lokal Bali yang lebih beragam.
- Dukungan multi-objek dalam satu gambar.
- Dashboard agregat untuk komunitas atau bank sampah.
- Database terkelola untuk deployment publik.
- Integrasi panduan lokasi drop-off dan bank sampah.

## 11. Perintah demo lokal

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

## 12. Pesan penutup

**BaliSort AI menghubungkan teknologi AI dengan kebiasaan sederhana:
kenali, pilah, dan jaga Bali tetap bersih.**
