# Clustering Kondisi Sekolah Dasar Indonesia 2023-2024

Proyek Data Mining menggunakan teknik **Clustering** untuk menganalisis dan mengelompokkan provinsi di Indonesia berdasarkan kondisi pendidikan Sekolah Dasar (SD). Membandingkan dua algoritma: **KMeans** dan **Agglomerative Clustering**.

---

## Tujuan

Mengidentifikasi provinsi yang kondisi pendidikan SD-nya:
- **Baik** → Provinsi Maju
- **Menengah** → Provinsi Berkembang
- **Butuh perhatian** → Provinsi Perlu Perhatian

---

## Dataset

| Atribut | Nilai |
|---|---|
| File | `kelayakan-pendidikan-indonesia.csv` |
| Sumber | Data BPS / Kemendikbud 2023-2024 |
| Total baris | 39 (38 provinsi + 1 Luar Negeri, dihapus) |
| Total kolom | 14 |

### Kolom Dataset

| Kolom | Keterangan |
|---|---|
| `Provinsi` | Nama provinsi (label, tidak masuk model) |
| `Sekolah` | Jumlah sekolah dasar |
| `Siswa` | Jumlah total siswa |
| `Mengulang` | Jumlah siswa mengulang |
| `Putus Sekolah` | Jumlah siswa putus sekolah |
| `Kepala Sekolah dan Guru(<S1)` | Guru dengan pendidikan di bawah S1 |
| `Kepala Sekolah dan Guru(>S1)` | Guru dengan pendidikan S1 ke atas |
| `Tenaga Kependidikan(SM)` | Tendik lulusan SMA/sederajat |
| `Tenaga Kependidikan(>SM)` | Tendik lulusan di atas SMA |
| `Rombongan Belajar` | Jumlah kelas aktif |
| `Ruang kelas(baik)` | Ruang kelas kondisi baik |
| `Ruang kelas(rusak ringan)` | Ruang kelas rusak ringan |
| `Ruang kelas(rusak sedang)` | Ruang kelas rusak sedang |
| `Ruang kelas(rusak berat)` | Ruang kelas rusak berat |

### Fitur Turunan (Feature Engineering)

| Fitur | Formula |
|---|---|
| `rasio_putus_sekolah` | `Putus Sekolah / Siswa × 100` |
| `rasio_guru_berkualifikasi` | `Guru(>S1) / (Guru(<S1) + Guru(>S1)) × 100` |
| `rasio_kelas_rusak` | `(Rusak ringan + sedang + berat) / Total kelas × 100` |

---

## Teknologi

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- tabulate

### Instalasi Dependensi

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tabulate scipy
```

---

## Cara Menjalankan

```bash
python data_mining_sd.py
```

Pastikan file `kelayakan-pendidikan-indonesia.csv` berada di direktori yang sama.

---

## Alur Analisis

```
1. Data Loading & Inspeksi Awal
        ↓
2. Feature Engineering (3 fitur turunan)
        ↓
3. Handling Missing Value + StandardScaler
        ↓
4. PCA (2 komponen, hanya untuk visualisasi)
        ↓
5. Elbow Method → K Optimal
        ↓
6. Training KMeans
        ↓
7. Training Agglomerative Clustering
        ↓
8. Interpretasi & Labeling Cluster
        ↓
9. Evaluasi & Perbandingan Model
        ↓
10. Simpan Laporan (laporan.txt)
        ↓
11. Simpan Visualisasi (7 file PNG)
```

---

## Hasil

### Parameter Model

| Parameter | Nilai |
|---|---|
| Jumlah cluster (k) | 2 |
| KMeans init | k-means++ |
| KMeans max_iter | 300 |
| KMeans n_init | 10 |
| Agglomerative linkage | ward |
| random_state | 42 |

### PCA

| Komponen | Variansi |
|---|---|
| PC1 | 72.69% |
| PC2 | 14.19% |
| **Total** | **86.87%** |

### Evaluasi Model

| Metrik | KMeans | Agglomerative |
|---|---|---|
| Silhouette Score | 0.6389 | **0.6633** |
| Davies-Bouldin Score | 0.5447 | **0.4369** |
| Inertia | 266.05 | N/A |

**Model terbaik: Agglomerative Clustering**
- Silhouette Score lebih tinggi (0.6633 vs 0.6389)
- Davies-Bouldin Score lebih rendah (0.4369 vs 0.5447)

---

## Output File

| File | Keterangan |
|---|---|
| `data_mining_sd.py` | Script Python utama |
| `hasil_clustering.csv` | Data provinsi + label cluster kedua model |
| `laporan.txt` | Laporan evaluasi lengkap |
| `elbow_method.png` | Grafik Elbow Method |
| `scatter_kmeans.png` | Scatter plot PCA 2D hasil KMeans |
| `scatter_agglomerative.png` | Scatter plot PCA 2D hasil Agglomerative |
| `heatmap_kmeans.png` | Heatmap rata-rata fitur per cluster (KMeans) |
| `heatmap_agglomerative.png` | Heatmap rata-rata fitur per cluster (Agglomerative) |
| `bar_distribusi.png` | Bar chart distribusi provinsi per cluster |
| `dendrogram.png` | Dendrogram Agglomerative Clustering |

---

## Struktur Direktori

```
tugas_data_mining/
├── data_mining_sd.py
├── kelayakan-pendidikan-indonesia.csv
├── hasil_clustering.csv
├── laporan.txt
├── elbow_method.png
├── scatter_kmeans.png
├── scatter_agglomerative.png
├── heatmap_kmeans.png
├── heatmap_agglomerative.png
├── bar_distribusi.png
├── dendrogram.png
└── README.md
```
