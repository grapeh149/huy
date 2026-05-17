import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Thư viện cho phân cụm và đánh giá
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                             davies_bouldin_score, adjusted_rand_score, 
                             normalized_mutual_info_score)

# ==========================================
# 0. KHỞI TẠO DỮ LIỆU GIẢ LẬP
# ==========================================
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Chuẩn hóa dữ liệu (Rất quan trọng trong phân cụm)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================
# 1. K-MEANS CLUSTERING
# ==========================================
print("--- K-MEANS ---")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

labels_km = kmeans.labels_
centroids = kmeans.cluster_centers_

# Trực quan hóa K-Means
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_km, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.title('K-Means Clustering')
plt.show()

print(f"Cluster labels: {np.unique(labels_km)}")
print(f"Points per cluster: {np.bincount(labels_km)}")
print(f"Inertia (Within-cluster sum of squares): {kmeans.inertia_:.2f}")

# Phương pháp Khuỷu tay (Elbow Method) để chọn K
inertias = []
K_range = range(1, 11)
for k in K_range:
    km_el = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_el.fit(X_scaled)
    inertias.append(km_el.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.axvline(x=4, color='r', linestyle='--', label='Elbow at K=4')
plt.legend()
plt.show()

# Tính Silhouette Score cho các K khác nhau
print("\nSilhouette Scores for different K:")
for k in range(2, 11):
    km_sil = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_sil = km_sil.fit_predict(X_scaled)
    score_sil = silhouette_score(X_scaled, labels_sil)
    print(f"K={k}: Silhouette Score = {score_sil:.4f}")


# ==========================================
# 2. HIERARCHICAL CLUSTERING (Phân cụm phân cấp)
# ==========================================
print("\n--- HIERARCHICAL CLUSTERING ---")
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)

# Trực quan hóa các cụm
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_hc, cmap='viridis', alpha=0.6)
plt.title('Hierarchical Clustering')
plt.show()

# Vẽ biểu đồ Dendrogram
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=5, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=5, color='r', linestyle='--', label='Cut threshold')
plt.legend()
plt.show()


# ==========================================
# 3. DBSCAN (Mật độ dựa trên cụm)
# ==========================================
print("\n--- DBSCAN ---")
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_db = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise_ = list(labels_db).count(-1)
print(f"Number of clusters found by DBSCAN: {n_clusters_db}")
print(f"Number of noise points: {n_noise_}")

# Trực quan hóa kết quả DBSCAN bao gồm nhiễu
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_db, cmap='viridis', alpha=0.6)
# Đánh dấu điểm nhiễu (noise) màu đỏ ký hiệu chữ X
noise_mask = (labels_db == -1)
plt.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], c='red', marker='x', s=100, label='Noise')
plt.legend()
plt.title('DBSCAN Clustering')
plt.colorbar(scatter)
plt.show()

# Tìm eps tối ưu bằng đồ thị K-Distance
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('K-Distance Graph (Elbow = eps)')
plt.axvline(x=280, color='g', linestyle=':') # Điểm uốn ước lượng
plt.axhline(y=0.3, color='r', linestyle='--', label='eps = 0.3')
plt.legend()
plt.show()


# ==========================================
# 4. ĐÁNH GIÁ CHẤT LƯỢNG PHÂN CỤM
# ==========================================
print("\n--- CLUSTERING EVALUATION (K-Means Results) ---")
# Chỉ số Nội bộ (Không cần nhãn thực tế)
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_km):.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, labels_km):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, labels_km):.4f}")

# Chỉ số Ngoại vi (Khi có sẵn nhãn gốc y_true)
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y_true, labels_km):.4f}")
print(f"Normalized Mutual Information (NMI): {normalized_mutual_info_score(y_true, labels_km):.4f}")


# ==========================================
# 5. SO SÁNH TRỰC QUAN BA THUẬT TOÁN
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# K-Means Subplot
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_km, cmap='viridis')
axes[0].set_title(f'K-Means\n(Sil: {silhouette_score(X_scaled, labels_km):.3f})')

# Hierarchical Subplot
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_hc, cmap='viridis')
axes[1].set_title(f'Hierarchical\n(Sil: {silhouette_score(X_scaled, labels_hc):.3f})')

# DBSCAN Subplot
axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_db, cmap='viridis')
axes[2].set_title(f'DBSCAN\n(Clusters: {n_clusters_db})')

plt.tight_layout()
plt.show()