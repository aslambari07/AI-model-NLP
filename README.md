# AI: Klasifikasi Teks Berita + Aplikasi Web

Proyek ini dibuat seperti alur praktikum dari langkah 1 sampai akhir:

1. Menentukan masalah AI
2. Mengumpulkan data
3. Pra-pemrosesan data
4. Melatih model
5. Evaluasi akurasi
6. Menyimpan model
7. Deploy ke aplikasi web yang siap dipakai

Use case yang dipilih adalah **klasifikasi kategori berita** berbasis NLP dengan 4 kelas:

- `World`
- `Sports`
- `Business`
- `Sci/Tech`

Dataset yang digunakan adalah **AG News** dari Hugging Face Datasets.

## Target

- Akurasi minimal `80%`
- Web responsif dan langsung bisa digunakan secara lokal

## Struktur Proyek

```text
AI/
├── app/
│   ├── main.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── styles.css
├── artifacts/
│   ├── model.joblib
│   └── metrics.json
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Langkah 1. Definisi Masalah

Masalah yang diselesaikan:

> Bagaimana membuat sistem yang dapat mengenali kategori sebuah berita secara otomatis berdasarkan judul dan isi teks?

Jenis AI:

- NLP
- Klasifikasi teks

Output model:

- Salah satu dari 4 label kategori berita

## Langkah 2. Pengumpulan Data

Sumber data:

- AG News (`train` dan `test`)
- Diunduh otomatis melalui library `datasets`

Kelebihan dataset:

- Sudah berlabel
- Ukurannya cukup besar
- Cocok untuk klasifikasi teks
- Umumnya mudah mencapai akurasi di atas 80%

## Langkah 3. Pra-pemrosesan

Pra-pemrosesan yang dilakukan:

- Membersihkan kolom teks berita
- Membersihkan spasi berlebih
- Mengubah teks menjadi fitur `TF-IDF`
- Menggunakan `bigram` agar konteks kata lebih baik

## Langkah 4. Model

Model utama:

- `TfidfVectorizer`
- `LogisticRegression`

Pipeline dipilih karena:

- Cepat dilatih
- Ringan
- Akurasi bagus untuk tugas klasifikasi teks
- Mudah diintegrasikan ke web

## Langkah 5. Training

### Instalasi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Melatih model

```bash
python src/train.py
```

Hasil training akan disimpan ke folder `artifacts/`.

## Langkah 6. Menjalankan Web

```bash
uvicorn app.main:app --reload
```

Lalu buka:

```text
http://127.0.0.1:8000
```

## Langkah 7. Cara Pakai Web

1. Buka halaman web
2. Masukkan teks berita
3. Klik tombol prediksi
4. Sistem akan menampilkan:
   - kategori prediksi
   - confidence score
   - probabilitas semua kelas

## Endpoint API

### `POST /predict`

Contoh request:

```json
{
  "text": "Apple unveils a faster laptop chip and stronger quarterly earnings."
}
```

## Catatan

- Jika model belum dilatih, jalankan `python src/train.py` terlebih dahulu.
- Web ini siap untuk pengembangan lanjutan ke hosting, mobile wrapper, atau integrasi database.
- Akurasi hasil training saat ini tersimpan di `artifacts/metrics.json`.
