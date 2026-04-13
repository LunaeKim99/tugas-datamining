import warnings
warnings.filterwarnings('ignore')

from modules.preprocessing import muat_data, feature_engineering, scaling, reduksi_pca
from modules.clustering import (cari_k_optimal, latih_kmeans, latih_agglomerative,
                                beri_label_cluster, interpretasi_cluster, evaluasi_model)
from modules.visualisasi import buat_semua_visualisasi
from modules.laporan import simpan_laporan, simpan_hasil_csv


def main():
    df = muat_data('kelayakan-pendidikan-indonesia.csv')

    X, labels = feature_engineering(df)

    X_scaled, fitur_numerik, scaler = scaling(X)

    pca, X_pca, pc1_var, pc2_var, total_var = reduksi_pca(X_scaled)

    k_optimal, _ = cari_k_optimal(X_scaled)

    kmeans, kmeans_labels = latih_kmeans(X_scaled, k_optimal)
    df['Cluster_KMeans'] = kmeans_labels

    agg, agg_labels = latih_agglomerative(X_scaled, k_optimal)
    df['Cluster_Agglomerative'] = agg_labels

    df_with_features = df.copy()
    for col in ['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']:
        df_with_features[col] = X[col].values

    label_map_kmeans = beri_label_cluster(df_with_features, 'Cluster_KMeans')
    label_map_agg = beri_label_cluster(df_with_features, 'Cluster_Agglomerative')

    df['Label_KMeans'] = df['Cluster_KMeans'].map(label_map_kmeans)
    df['Label_Agglomerative'] = df['Cluster_Agglomerative'].map(label_map_agg)

    interpretasi_cluster(df, df_with_features, label_map_kmeans, label_map_agg)

    evaluasi = evaluasi_model(X_scaled, kmeans, kmeans_labels, agg_labels)

    simpan_laporan(df, fitur_numerik, pc1_var, pc2_var, total_var,
                   k_optimal, evaluasi, label_map_kmeans, label_map_agg)

    buat_semua_visualisasi(df, df_with_features, X_pca, X_scaled,
                           kmeans_labels, agg_labels, kmeans,
                           labels, label_map_kmeans, label_map_agg,
                           pca, pc1_var, pc2_var)

    simpan_hasil_csv(df)

    print("\n--- SELESAI ---")
    print("File output:")
    print("  - hasil_clustering.csv")
    print("  - laporan.txt")
    print("  - elbow_method.png")
    print("  - scatter_kmeans.png")
    print("  - scatter_agglomerative.png")
    print("  - heatmap_kmeans.png")
    print("  - heatmap_agglomerative.png")
    print("  - bar_distribusi.png")
    print("  - dendrogram.png")


if __name__ == '__main__':
    main()
