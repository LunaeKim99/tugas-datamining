import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tabulate import tabulate
from .utils import cetak_tabel


def muat_data(filepath='kelayakan-pendidikan-indonesia.csv'):
    df = pd.read_csv(filepath)

    cetak_tabel(df, "5 Baris Pertama Dataset", jumlah=5)

    print("\n--- INFO DATASET ---")
    df.info()

    print("\n--- STATISTIK DESKRIPTIF ---")
    print(tabulate(df.describe(), headers='keys', tablefmt='grid', floatfmt='.2f'))

    print("\n--- MISSING VALUE PER KOLOM ---")
    missing = df.isnull().sum()
    print(tabulate([[col, val] for col, val in missing.items()],
                   headers=['Kolom', 'Missing Value'], tablefmt='grid'))

    df = df[df['Provinsi'] != 'Luar Negeri'].reset_index(drop=True)
    print(f"\n1. Dataset berhasil dimuat. {len(df)} provinsi, {df.shape[1]} kolom.")
    return df


def feature_engineering(df):
    labels = df['Provinsi'].reset_index(drop=True)
    X = df.drop(columns=['Provinsi']).copy()

    X['rasio_putus_sekolah'] = (X['Putus Sekolah'] / X['Siswa']) * 100

    guru_total = X['Kepala Sekolah dan Guru(<S1)'] + X['Kepala Sekolah dan Guru(>S1)']
    X['rasio_guru_berkualifikasi'] = (X['Kepala Sekolah dan Guru(>S1)'] / guru_total) * 100

    kelas_rusak = (X['Ruang kelas(rusak ringan)'] + X['Ruang kelas(rusak sedang)']
                   + X['Ruang kelas(rusak berat)'])
    kelas_total = X['Ruang kelas(baik)'] + kelas_rusak
    X['rasio_kelas_rusak'] = (kelas_rusak / kelas_total) * 100

    print("2. Feature engineering selesai. 3 fitur baru ditambahkan.")
    cetak_tabel(
        X[['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']]
        .assign(Provinsi=labels.values).set_index('Provinsi'),
        "Fitur Turunan (5 Baris Pertama)",
        jumlah=5
    )
    return X, labels


def scaling(X):
    fitur_numerik = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in fitur_numerik:
        X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[fitur_numerik])
    print(f"3. Scaling selesai. Total fitur: {X_scaled.shape[1]}")
    return X_scaled, fitur_numerik, scaler


def reduksi_pca(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    total_var = pc1_var + pc2_var

    print(f"4. PCA selesai. Variansi dijelaskan: {total_var:.2f}%")
    print(f"   PCA: PC1={pc1_var:.2f}%, PC2={pc2_var:.2f}%, Total={total_var:.2f}%")
    return pca, X_pca, pc1_var, pc2_var, total_var
