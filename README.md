# BaliSort AI

BaliSort AI adalah aplikasi Streamlit untuk mengenali kategori sampah dari
gambar menggunakan MobileNetV3-Large dan memberikan rekomendasi pemilahan.
Aplikasi ini menggunakan enam kategori:

`cardboard`, `glass`, `metal`, `paper`, `plastic`, dan `trash`.

## Teknologi

- Python 3.12
- PyTorch 2.11.0
- Torchvision 0.26.0
- MobileNetV3-Large
- Streamlit
- Pillow
- SQLite

## Struktur proyek

```text
app.py                         # Aplikasi Streamlit
src/
  model.py                     # Pembuatan MobileNetV3-Large
  preprocessing.py             # Transformasi gambar
  prepare_dataset.py           # Validasi dan pembagian dataset
  train.py                     # Training dan checkpoint terbaik
  predict.py                   # Inference aplikasi
  evaluation.py                # Evaluasi dataset test
  recommendation.py            # Rekomendasi pemilahan
database/
  db.py                        # Penyimpanan riwayat prediksi
dataset/
  raw/                         # Dataset asli
  processed/                   # Dataset train, validation, dan test
models/
  best_model.pth               # Checkpoint hasil training
reports/
  classification_report.txt   # Laporan evaluasi
  confusion_matrix.png         # Confusion matrix
tests/                         # Pengujian
```

## Instalasi

Gunakan virtual environment Python 3.12:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Untuk instalasi PyTorch dengan CUDA, gunakan wheel yang sesuai dengan driver
NVIDIA pada mesin. Aplikasi tetap dapat berjalan menggunakan CPU jika CUDA
tidak tersedia.

## Menyiapkan dataset

Letakkan dataset asli dengan struktur kelas berikut:

```text
dataset/raw/extracted/dataset-resized/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Jalankan preparasi dataset:

```powershell
.\venv\Scripts\python.exe src\prepare_dataset.py
```

Script memvalidasi JPG, JPEG, dan PNG menggunakan Pillow, mengabaikan gambar
rusak, lalu menyalin data ke:

```text
dataset/processed/train/       # 70%
dataset/processed/validation/  # 15%
dataset/processed/test/        # 15%
```

Dataset asli tidak dihapus.

## Training

Pastikan dataset processed tersedia, lalu jalankan:

```powershell
.\venv\Scripts\python.exe src\train.py
```

Konfigurasi utama:

- MobileNetV3-Large
- batch size 32
- AdamW dengan learning rate `0.0001`
- class weights berdasarkan distribusi training
- mixed precision saat CUDA tersedia
- maksimal 20 epoch
- early stopping patience 5
- random seed 42

Checkpoint terbaik berdasarkan validation macro F1 disimpan ke
`models/best_model.pth`. Dataset test tidak digunakan selama training.

## Evaluasi

Evaluasi hanya menggunakan dataset test:

```powershell
.\venv\Scripts\python.exe src\evaluation.py
```

Hasil disimpan ke:

- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

Evaluasi terakhir menghasilkan:

- Accuracy: 91,67%
- Macro precision: 91,07%
- Macro recall: 91,14%
- Macro F1: 91,06%

Metrik tersebut berlaku untuk checkpoint dan dataset yang saat ini tersedia,
dan tidak menjamin performa pada gambar di luar distribusi dataset.

## Menjalankan aplikasi

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit, biasanya
`http://localhost:8501`.

Alur aplikasi:

1. Unggah gambar JPG, JPEG, atau PNG.
2. Aplikasi memvalidasi dan mengubah gambar menjadi RGB.
3. Model menghasilkan kategori dan confidence score.
4. Prediksi dengan confidence di bawah 60% ditandai sebagai tidak yakin.
5. Aplikasi menampilkan rekomendasi pemilahan dan probabilitas tiap kelas.
6. Prediksi dapat disimpan ke riwayat SQLite.
7. Pengguna dapat memberi feedback benar/salah dan kategori aktual.
8. Ringkasan penggunaan menampilkan distribusi prediksi dan akurasi feedback.

## Deployment Streamlit Community Cloud

Sebelum deployment, pastikan `models/best_model.pth` tersedia di repository.
File model ini diperlukan aplikasi untuk melakukan inference. Dataset training
tidak diperlukan saat aplikasi berjalan.

1. Push repository ke GitHub.
2. Buka Streamlit Community Cloud dan pilih **New app**.
3. Pilih repository, branch, dan file utama `app.py`.
4. Deploy aplikasi.
5. Tunggu instalasi dependency dari `requirements.txt`.

Jalankan pengujian lokal sebelum deploy:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
.\venv\Scripts\python.exe -m streamlit run app.py
```

Konfigurasi tampilan dan server berada di
[`.streamlit/config.toml`](.streamlit/config.toml). Jangan menyimpan API key
atau password di file tersebut. Jika konfigurasi rahasia diperlukan, gunakan
fitur **Secrets** pada platform deployment dan baca melalui
`st.secrets`.

Riwayat prediksi saat ini menggunakan SQLite lokal. Pada hosting ephemeral,
data tersebut dapat hilang ketika aplikasi di-restart atau di-redeploy.
Gunakan database terkelola jika riwayat harus persisten untuk pengguna publik.

Feedback dan statistik penggunaan juga disimpan secara lokal pada database yang
sama. File `*.db` diabaikan oleh Git dan tidak boleh dikomit karena dapat
berisi nama file unggahan atau data penggunaan. Untuk deployment publik:

- Jangan menyimpan gambar unggahan; aplikasi hanya menyimpan nama file,
  kategori, confidence, dan feedback.
- Hindari nama file yang memuat informasi pribadi.
- Beri tahu pengguna bahwa feedback digunakan untuk pemantauan kualitas model.
- Gunakan database terkelola dengan kontrol akses jika data harus persisten.
- Hapus atau batasi data lama sesuai kebijakan privasi aplikasi.

## Catatan dan batasan

- Model hanya mengenali enam kategori yang telah ditentukan.
- Prediksi bukan pengganti pemeriksaan manusia untuk sampah berbahaya atau
  benda yang tidak jelas.
- Confidence model bukan ukuran akurasi mutlak.
- Jangan mengunggah gambar yang mengandung data pribadi sensitif.
- File database lokal dibuat di `database/balisort.db`.

## Pengembangan berikutnya

- Memperluas pengujian otomatis untuk alur UI.
- Mengekspor feedback secara terkontrol untuk kurasi dataset.
- Menambahkan autentikasi dan penyimpanan terkelola sebelum deployment publik.