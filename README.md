# BaliSort AI

## Description

BaliSort AI adalah aplikasi web berbasis AI untuk mengenali kategori sampah
dari gambar dan memberikan rekomendasi pemilahan yang sesuai.

## Problem & Target User

Masyarakat masih kesulitan membedakan sampah yang dapat didaur ulang dan
sampah residu. BaliSort AI ditujukan untuk masyarakat dan rumah tangga di
Bali, serta siswa, sekolah, komunitas lingkungan, dan bank sampah.

## Key Features

- Upload dan validasi gambar JPG, JPEG, dan PNG.
- Klasifikasi enam kategori sampah.
- Confidence score dan peringatan prediksi kurang yakin.
- Rekomendasi wadah dan cara pemilahan.
- Detail probabilitas semua kategori.
- Riwayat prediksi dan feedback pengguna.
- Statistik penggunaan dan ekspor feedback CSV.
- Panduan edukasi pemilahan sampah.

Kategori model:

`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

## Tech Stack

- Python 3.12
- Streamlit
- PyTorch 2.11.0
- Torchvision 0.26.0
- Pillow
- NumPy
- scikit-learn
- SQLite

## AI Implementation

Model menggunakan MobileNetV3-Large berbasis PyTorch dan Torchvision dengan
pretrained weights ImageNet. Gambar input diproses sebagai RGB berukuran
224 × 224 piksel, lalu diklasifikasikan ke enam kategori sampah. Output
meliputi kategori, confidence score, probabilitas, dan rekomendasi pemilahan.

Model dapat memberikan hasil kurang akurat pada gambar buram, objek yang
tercampur, atau gambar di luar distribusi dataset.

## Architecture / Data Flow

```text
User
  -> Streamlit Frontend
  -> Image Validation & Preprocessing
  -> MobileNetV3-Large / PyTorch
  -> Category, Confidence & Probabilities
  -> Sorting Recommendation
  -> SQLite History, Feedback & Statistics
```

## Installation / Run Instructions

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

Aplikasi berjalan di `http://localhost:8501`.

Untuk menjalankan pengujian:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

Checkpoint model harus tersedia di:

```text
models/best_model.pth
```

## Deployment URL

https://balisort-ai.streamlit.app

## Demo Account (jika diperlukan - jangan simpan password sensitif di public repo)

Tidak diperlukan. Aplikasi dapat digunakan tanpa login.

## Known Limitations

- Model hanya mengenali enam kategori.
- Prediksi bukan pengganti pemeriksaan manusia.
- Confidence score bukan jaminan kebenaran.
- Model dapat keliru pada gambar buram atau sampah yang tercampur.
- Riwayat dan feedback menggunakan SQLite lokal.
- Data SQLite dapat hilang pada hosting ephemeral.

## Third-party Resources / Credits

- PyTorch dan Torchvision untuk model dan inference.
- MobileNetV3-Large pretrained weights dari Torchvision/ImageNet.
- Streamlit untuk antarmuka web.
- Pillow, NumPy, scikit-learn, dan SQLite untuk pemrosesan, evaluasi, dan
  penyimpanan data.
- GitHub Copilot untuk membantu coding, debugging, testing, dan dokumentasi.

Tidak menggunakan API AI eksternal atau credential pihak ketiga.

## Team Members Sirkula

- Mecky Alvin Tanuwidjaya
- Universitas Tarumanagara
