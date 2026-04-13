from datetime import datetime
from .utils import out


def simpan_laporan(df, fitur_numerik, pc1_var, pc2_var, total_var,
                   k_optimal, evaluasi, label_map_kmeans, label_map_agg):
    distribusi_kmeans = {}
    for c in sorted(df['Cluster_KMeans'].unique()):
        label = label_map_kmeans[c]
        provinsi_list = df[df['Cluster_KMeans'] == c]['Provinsi'].tolist()
        distribusi_kmeans[c] = (label, provinsi_list)

    distribusi_agg = {}
    for c in sorted(df['Cluster_Agglomerative'].unique()):
        label = label_map_agg[c]
        provinsi_list = df[df['Cluster_Agglomerative'] == c]['Provinsi'].tolist()
        distribusi_agg[c] = (label, provinsi_list)

    with open(out('laporan.txt'), 'w', encoding='utf-8') as f:
        f.write("================================================\n")
        f.write("LAPORAN HASIL CLUSTERING\n")
        f.write("Dataset: Pendidikan SD Indonesia 2023-2024\n")
        f.write(f"Tanggal: {datetime.now().strftime('%d %B %Y %H:%M:%S')}\n")
        f.write("================================================\n\n")

        f.write("[INFORMASI DATASET]\n")
        f.write(f"- Total provinsi: {len(df)}\n")
        f.write(f"- Total fitur: {len(fitur_numerik)}\n")
        f.write(f"- Fitur yang digunakan: {', '.join(fitur_numerik)}\n")
        f.write("- Fitur turunan: rasio_putus_sekolah, rasio_guru_berkualifikasi, rasio_kelas_rusak\n\n")

        f.write("[PCA]\n")
        f.write(f"- PC1 explained variance: {pc1_var:.2f}%\n")
        f.write(f"- PC2 explained variance: {pc2_var:.2f}%\n")
        f.write(f"- Total variance explained: {total_var:.2f}%\n\n")

        f.write("[PARAMETER MODEL]\n")
        f.write(f"- Jumlah Cluster (k): {k_optimal}\n")
        f.write("- KMeans: init=k-means++, max_iter=300, n_init=10, random_state=42\n")
        f.write("- Agglomerative: linkage=ward\n\n")

        f.write("[HASIL EVALUASI]\n")
        f.write(f"- KMeans Silhouette Score    : {evaluasi['sil_kmeans']:.4f}\n")
        f.write(f"- KMeans Davies-Bouldin Score: {evaluasi['db_kmeans']:.4f}\n")
        f.write(f"- KMeans Inertia             : {evaluasi['inertia_kmeans']:.2f}\n")
        f.write(f"- Agglomerative Silhouette   : {evaluasi['sil_agg']:.4f}\n")
        f.write(f"- Agglomerative Davies-Bouldin: {evaluasi['db_agg']:.4f}\n\n")

        f.write("[KESIMPULAN]\n")
        f.write(f"Model terbaik: {evaluasi['best_model']}\n")
        f.write(f"Alasan: {evaluasi['best_model']} {evaluasi['alasan']}.\n\n")

        f.write("[DISTRIBUSI PROVINSI PER CLUSTER — KMEANS]\n")
        for c, (label, provinsi_list) in distribusi_kmeans.items():
            f.write(f"Cluster {c} ({label}): {', '.join(provinsi_list)}\n")

        f.write("\n[DISTRIBUSI PROVINSI PER CLUSTER — AGGLOMERATIVE]\n")
        for c, (label, provinsi_list) in distribusi_agg.items():
            f.write(f"Cluster {c} ({label}): {', '.join(provinsi_list)}\n")

        f.write("\n================================================\n")

    print("10. Laporan berhasil disimpan ke laporan.txt")


def simpan_hasil_csv(df):
    hasil = df[['Provinsi', 'Cluster_KMeans', 'Label_KMeans',
                'Cluster_Agglomerative', 'Label_Agglomerative']].copy()
    hasil.to_csv(out('hasil_clustering.csv'), index=False, encoding='utf-8-sig')
