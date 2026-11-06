# Checklist Final BaliSort AI

## Fungsionalitas aplikasi

- [ ] Aplikasi berjalan dengan `streamlit run app.py`.
- [ ] Upload JPG, JPEG, dan PNG berhasil.
- [ ] File rusak atau non-gambar ditolak dengan pesan yang jelas.
- [ ] Model menampilkan kategori dan confidence.
- [ ] Confidence rendah ditandai sebagai hasil yang perlu diverifikasi.
- [ ] Rekomendasi pemilahan tampil sesuai kategori.
- [ ] Probabilitas semua kategori dapat dilihat.
- [ ] Prediksi dapat disimpan ke riwayat.
- [ ] Riwayat prediksi tampil setelah aplikasi dijalankan ulang.
- [ ] Tab edukasi enam kategori tampil dengan benar.

## Model dan data

- [ ] `models/best_model.pth` tersedia pada environment deployment.
- [ ] Urutan kelas tetap `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- [ ] Ukuran input tetap 224 × 224 RGB.
- [ ] Dataset test tidak digunakan saat training.
- [ ] Checkpoint dapat dimuat pada CPU.
- [ ] CUDA digunakan jika tersedia.

## Validasi teknis

Jalankan dari root proyek:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
.\venv\Scripts\python.exe -m py_compile app.py
.\venv\Scripts\python.exe src\evaluation.py
.\venv\Scripts\python.exe -m streamlit run app.py
```

- [ ] Semua test lulus.
- [ ] `reports/classification_report.txt` tersedia.
- [ ] `reports/confusion_matrix.png` tersedia.
- [ ] Streamlit health check mengembalikan `ok`.
- [ ] Tidak ada secret atau API key di repository.

## Demo kompetisi

- [ ] Siapkan minimal satu contoh gambar untuk setiap kategori.
- [ ] Siapkan contoh gambar buram atau ambigu untuk mendemonstrasikan
      uncertainty handling.
- [ ] Tunjukkan alur upload sampai rekomendasi.
- [ ] Tunjukkan riwayat prediksi.
- [ ] Jelaskan bahwa hasil AI adalah rekomendasi, bukan kepastian mutlak.
- [ ] Siapkan ringkasan teknis dari [`PRESENTATION.md`](./PRESENTATION.md).

## Deployment publik

- [ ] Repository memiliki `requirements.txt` yang lengkap.
- [ ] File utama deployment adalah `app.py`.
- [ ] `.streamlit/config.toml` tersedia.
- [ ] Ukuran checkpoint sesuai batas platform deployment.
- [ ] Jalur file menggunakan path relatif terhadap project root.
- [ ] Keterbatasan SQLite lokal sudah dipahami.
- [ ] Tidak ada data pribadi di gambar demo atau database yang dibagikan.
