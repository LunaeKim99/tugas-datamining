import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tabulate import tabulate
from .utils import out, simpan_plot


def cari_k_optimal(X_scaled):
    inertia_list = []
    k_range = range(1, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        km.fit(X_scaled)
        inertia_list.append(km.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_list, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Jumlah Cluster (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method - Menentukan K Optimal', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    for k, inertia in zip(k_range, inertia_list):
        plt.annotate(f'{inertia:.0f}', (k, inertia), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)
    simpan_plot(out('elbow_method.png'))

    diffs = np.diff(inertia_list)
    diffs2 = np.diff(diffs)
    k_optimal = int(np.argmax(np.abs(diffs2)) + 2)
    if k_optimal < 2:
        k_optimal = 3

    print(f"5. K optimal yang disarankan: {k_optimal}")
    return k_optimal, inertia_list


def latih_kmeans(X_scaled, k_optimal):
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    kmeans_labels = kmeans.predict(X_scaled)
    print("6. Model KMeans berhasil dilatih.")
    return kmeans, kmeans_labels


def latih_agglomerative(X_scaled, k_optimal):
    agg = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
    agg_labels = agg.fit_predict(X_scaled)
    print("7. Model Agglomerative berhasil dilatih.")
    return agg, agg_labels


def beri_label_cluster(df_input, cluster_col):
    rasio_cols = ['rasio_putus_sekolah', 'rasio_kelas_rusak']
    kualifikasi_col = 'rasio_guru_berkualifikasi'
    cluster_means = df_input.groupby(cluster_col)[rasio_cols + [kualifikasi_col]].mean()

    skor = {}
    for c in cluster_means.index:
        skor_negatif = float(cluster_means.at[c, 'rasio_putus_sekolah']
                             + cluster_means.at[c, 'rasio_kelas_rusak']) / 2
        skor_positif = float(cluster_means.at[c, kualifikasi_col])
        skor[c] = skor_positif - skor_negatif

    sorted_clusters = sorted(skor.keys(), key=lambda c_key: skor[c_key], reverse=True)

    label_map = {}
    n = len(sorted_clusters)
    for i, c in enumerate(sorted_clusters):
        if i == 0:
            label_map[c] = "Provinsi Maju"
        elif i == n - 1:
            label_map[c] = "Provinsi Perlu Perhatian"
        else:
            label_map[c] = "Provinsi Berkembang"

    return label_map


def interpretasi_cluster(df, df_with_features, label_map_kmeans, label_map_agg):
    rasio_fitur = ['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']

    print("\n--- INTERPRETASI CLUSTER KMEANS ---")
    km_stats = df_with_features.groupby('Cluster_KMeans')[rasio_fitur].mean().round(4)
    km_stats.index = [f"Cluster {i} ({label_map_kmeans[i]})" for i in km_stats.index]
    print(tabulate(km_stats, headers='keys', tablefmt='grid', floatfmt='.4f'))

    print("\n--- DAFTAR PROVINSI (KMEANS) ---")
    km_list = df[['Provinsi', 'Cluster_KMeans', 'Label_KMeans']].copy()
    km_list.columns = ['Provinsi', 'Cluster', 'Label']
    print(tabulate(km_list, headers='keys', tablefmt='grid', showindex=False))

    print("\n--- INTERPRETASI CLUSTER AGGLOMERATIVE ---")
    agg_stats = df_with_features.groupby('Cluster_Agglomerative')[rasio_fitur].mean().round(4)
    agg_stats.index = [f"Cluster {i} ({label_map_agg[i]})" for i in agg_stats.index]
    print(tabulate(agg_stats, headers='keys', tablefmt='grid', floatfmt='.4f'))

    print("\n--- DAFTAR PROVINSI (AGGLOMERATIVE) ---")
    agg_list = df[['Provinsi', 'Cluster_Agglomerative', 'Label_Agglomerative']].copy()
    agg_list.columns = ['Provinsi', 'Cluster', 'Label']
    print(tabulate(agg_list, headers='keys', tablefmt='grid', showindex=False))


def evaluasi_model(X_scaled, kmeans, kmeans_labels, agg_labels):
    sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
    db_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
    inertia_kmeans = kmeans.inertia_

    sil_agg = silhouette_score(X_scaled, agg_labels)
    db_agg = davies_bouldin_score(X_scaled, agg_labels)

    print(f"\n9. Silhouette KMeans: {sil_kmeans:.4f} | Silhouette Agglomerative: {sil_agg:.4f}")

    tabel_eval = [
        ['Silhouette Score', f'{sil_kmeans:.4f}', f'{sil_agg:.4f}'],
        ['Davies-Bouldin Score', f'{db_kmeans:.4f}', f'{db_agg:.4f}'],
        ['Inertia', f'{inertia_kmeans:.2f}', 'N/A'],
    ]
    print("\n--- TABEL PERBANDINGAN MODEL ---")
    print(tabulate(tabel_eval,
                   headers=['Metrik', 'KMeans', 'Agglomerative'],
                   tablefmt='grid'))

    sil_winner = 'KMeans' if sil_kmeans >= sil_agg else 'Agglomerative'
    db_winner = 'KMeans' if db_kmeans <= db_agg else 'Agglomerative'

    if sil_winner == db_winner:
        best_model = sil_winner
        alasan = (f"memiliki Silhouette Score lebih tinggi ({max(sil_kmeans, sil_agg):.4f}) "
                  f"dan Davies-Bouldin Score lebih rendah ({min(db_kmeans, db_agg):.4f})")
    elif sil_winner == 'KMeans':
        best_model = 'KMeans'
        alasan = (f"memiliki Silhouette Score lebih tinggi ({sil_kmeans:.4f} vs {sil_agg:.4f}), "
                  f"meskipun Davies-Bouldin Score sedikit lebih tinggi")
    else:
        best_model = 'Agglomerative'
        alasan = (f"memiliki Silhouette Score lebih tinggi ({sil_agg:.4f} vs {sil_kmeans:.4f}), "
                  f"meskipun tidak memiliki inertia")

    kesimpulan = f"Kesimpulan: Model terbaik adalah {best_model} karena {alasan}."
    print(f"\n{kesimpulan}")
    print(f"9. Model terbaik: {best_model}")

    return {
        'sil_kmeans': sil_kmeans, 'db_kmeans': db_kmeans, 'inertia_kmeans': inertia_kmeans,
        'sil_agg': sil_agg, 'db_agg': db_agg,
        'best_model': best_model, 'alasan': alasan, 'kesimpulan': kesimpulan
    }
