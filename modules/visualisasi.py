import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from .utils import out, simpan_plot

COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']


def plot_scatter_kmeans(df, X_pca, kmeans_labels, kmeans, pca, labels, label_map_kmeans, pc1_var, pc2_var):
    fig, ax = plt.subplots(figsize=(14, 10))
    for c in sorted(df['Cluster_KMeans'].unique()):
        mask = kmeans_labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[c % len(COLORS)],
                   label=f"Cluster {c} ({label_map_kmeans[c]})",
                   s=100, alpha=0.8, edgecolors='white', linewidth=0.5)

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               marker='*', s=400, c='black', zorder=5, label='Centroid')

    for i, prov in enumerate(labels.values):
        ax.annotate(prov.replace('Prov. ', ''), (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=6.5, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
    ax.set_title('Hasil Clustering KMeans (PCA 2D)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    simpan_plot(out('scatter_kmeans.png'))


def plot_scatter_agglomerative(df, X_pca, agg_labels, labels, label_map_agg, pc1_var, pc2_var):
    fig, ax = plt.subplots(figsize=(14, 10))
    for c in sorted(df['Cluster_Agglomerative'].unique()):
        mask = agg_labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[c % len(COLORS)],
                   label=f"Cluster {c} ({label_map_agg[c]})",
                   s=100, alpha=0.8, edgecolors='white', linewidth=0.5)

    for c in sorted(df['Cluster_Agglomerative'].unique()):
        mask = agg_labels == c
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.scatter(cx, cy, marker='*', s=400, c='black', zorder=5)

    ax.scatter([], [], marker='*', s=400, c='black', label='Centroid (rata-rata)')

    for i, prov in enumerate(labels.values):
        ax.annotate(prov.replace('Prov. ', ''), (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=6.5, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
    ax.set_title('Hasil Agglomerative Clustering (PCA 2D)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    simpan_plot(out('scatter_agglomerative.png'))


def plot_heatmap_kmeans(df_with_features, label_map_kmeans):
    rasio_fitur = ['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']
    hm_km = df_with_features.groupby('Cluster_KMeans')[rasio_fitur].mean()
    hm_km.index = [f"Cluster {i}\n({label_map_kmeans[i]})" for i in hm_km.index]
    hm_km.columns = ['Rasio Putus\nSekolah (%)', 'Rasio Guru\nBerkualifikasi (%)', 'Rasio Kelas\nRusak (%)']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(hm_km, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Nilai'})
    ax.set_title('Heatmap Rata-rata Fitur Turunan per Cluster (KMeans)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fitur', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    plt.tight_layout()
    simpan_plot(out('heatmap_kmeans.png'))


def plot_heatmap_agglomerative(df_with_features, label_map_agg):
    rasio_fitur = ['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']
    hm_agg = df_with_features.groupby('Cluster_Agglomerative')[rasio_fitur].mean()
    hm_agg.index = [f"Cluster {i}\n({label_map_agg[i]})" for i in hm_agg.index]
    hm_agg.columns = ['Rasio Putus\nSekolah (%)', 'Rasio Guru\nBerkualifikasi (%)', 'Rasio Kelas\nRusak (%)']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(hm_agg, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Nilai'})
    ax.set_title('Heatmap Rata-rata Fitur Turunan per Cluster (Agglomerative)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fitur', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    plt.tight_layout()
    simpan_plot(out('heatmap_agglomerative.png'))


def plot_bar_distribusi(df):
    km_counts = df['Cluster_KMeans'].value_counts().sort_index()
    agg_counts = df['Cluster_Agglomerative'].value_counts().sort_index()
    all_clusters = sorted(set(km_counts.index) | set(agg_counts.index))
    km_vals = [km_counts.get(c, 0) for c in all_clusters]
    agg_vals = [agg_counts.get(c, 0) for c in all_clusters]

    x = np.arange(len(all_clusters))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, km_vals, width, label='KMeans',
                   color='#3498db', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, agg_vals, width, label='Agglomerative',
                   color='#e74c3c', edgecolor='white', linewidth=0.5)

    for bar in bars1:
        ax.annotate(f'{int(bar.get_height())}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.annotate(f'{int(bar.get_height())}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Nomor Cluster', fontsize=11)
    ax.set_ylabel('Jumlah Provinsi', fontsize=11)
    ax.set_title('Distribusi Provinsi per Cluster: KMeans vs Agglomerative', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {c}' for c in all_clusters])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    simpan_plot(out('bar_distribusi.png'))


def plot_dendrogram(X_scaled, labels):
    Z = linkage(X_scaled, method='ward')
    fig, ax = plt.subplots(figsize=(18, 8))
    dendrogram(Z, labels=labels.values, leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]), ax=ax)
    ax.set_title('Dendrogram Agglomerative Clustering (Ward Linkage)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Provinsi', fontsize=11)
    ax.set_ylabel('Jarak (Distance)', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    simpan_plot(out('dendrogram.png'))


def buat_semua_visualisasi(df, df_with_features, X_pca, X_scaled,
                            kmeans_labels, agg_labels, kmeans,
                            labels, label_map_kmeans, label_map_agg,
                            pca, pc1_var, pc2_var):
    plot_scatter_kmeans(df, X_pca, kmeans_labels, kmeans, pca, labels, label_map_kmeans, pc1_var, pc2_var)
    plot_scatter_agglomerative(df, X_pca, agg_labels, labels, label_map_agg, pc1_var, pc2_var)
    plot_heatmap_kmeans(df_with_features, label_map_kmeans)
    plot_heatmap_agglomerative(df_with_features, label_map_agg)
    plot_bar_distribusi(df)
    plot_dendrogram(X_scaled, labels)
    print("11. Semua visualisasi berhasil disimpan.")
