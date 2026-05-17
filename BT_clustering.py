import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs, make_moons, make_circles

# =====================================================================
# Exercise 5.1: Customer Segmentation (Mall Customers Dataset)
# =====================================================================
print("==================================================")
print("EXERCISE 5.1: CUSTOMER SEGMENTATION")
print("==================================================")

# Khởi tạo tập dữ liệu Khách hàng giả lập (Cấu trúc tương tự dữ liệu Mall_Customers)
np.random.seed(42)
mall_df = pd.DataFrame({
    'CustomerID': range(1, 201),
    'Gender': np.random.choice(['Male', 'Female'], size=200),
    'Age': np.random.randint(18, 70, size=200),
    'Annual Income (k$)': np.random.randint(15, 137, size=200),
    'Spending Score (1-100)': np.random.randint(1, 100, size=200)
})

# Trích xuất 2 đặc tính cốt lõi thường dùng để phân khúc
X_mall = mall_df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler_m = StandardScaler()
X_mall_scaled = scaler_m.fit_transform(X_mall)

# Thử các giá trị K từ 2 tới 10 để chọn K thích hợp
inertias_m = []
sil_scores_m = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_mall_scaled)
    inertias_m.append(km.inertia_)
    sil_scores_m.append(silhouette_score(X_mall_scaled, labels))

# Vẽ biểu đồ Elbow và Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias_m, 'bo-')
ax1.set_title('Elbow Method (Mall Data)')
ax1.set_xlabel('Number of Clusters K')
ax1.set_ylabel('Inertia')

ax2.plot(K_range, sil_scores_m, 'ro-')
ax2.set_title('Silhouette Score (Mall Data)')
ax2.set_xlabel('Number of Clusters K')
ax2.set_ylabel('Silhouette Score')
plt.show()

# Giả sử K=5 là điểm uốn tối ưu thường thấy ở bài toán này
optimal_k_mall = 5
km_final = KMeans(n_clusters=optimal_k_mall, random_state=42, n_init=10)
mall_df['Cluster'] = km_final.fit_predict(X_mall_scaled)

print(f"\nPhân bố số lượng khách hàng trong {optimal_k_mall} nhóm mới:")
print(mall_df['Cluster'].value_counts())

# Vẽ đồ thị phân khúc khách hàng
plt.figure(figsize=(8, 6))
plt.scatter(X_mall.iloc[:, 0], X_mall.iloc[:, 1], c=mall_df['Cluster'], cmap='rainbow', alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments (K-Means)')
plt.show()


# =====================================================================
# Exercise 5.2: Image Compression (Nén ảnh bằng K-Means)
# =====================================================================
print("\n==================================================")
print("EXERCISE 5.2: IMAGE COMPRESSION USING K-MEANS")
print("==================================================")

# Khởi tạo một mảng ma trận ảnh màu giả lập kích thước 64x64 pixel gồm 3 kênh màu RGB
original_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

# 1. Biến đổi mảng (Reshape) ảnh về định dạng ma trận (n_pixels, 3)
sh = original_image.shape
X_img = original_image.reshape(sh[0] * sh[1], sh[2])

# Chia thang màu về khoảng [0, 1] giúp K-Means chạy nhanh, ổn định hơn
X_img_norm = X_img / 255.0

# 2. Tiến hành phân cụm K-Means với K = 16 màu chủ đạo
K_colors = 16
print(f"Đang thực hiện nén ma trận ảnh từ {X_img.shape[0]} màu xuống còn {K_colors} cụm màu chính...")
km_img = KMeans(n_clusters=K_colors, random_state=42, n_init=10)
labels_img = km_img.fit_predict(X_img_norm)
centroids_img = km_img.cluster_centers_

# 3. Thay thế giá trị màu ban đầu của từng pixel bằng màu của tâm cụm màu đại diện
compressed_img_flat = centroids_img[labels_img]
# Khôi phục lại kích thước ảnh gốc ban đầu
compressed_image = (compressed_img_flat.reshape(sh[0], sh[1], sh[2]) * 255).astype(np.uint8)

# 4. Hiển thị so sánh trực quan hai bức ảnh
fig, (ax_orig, ax_comp) = plt.subplots(1, 2, figsize=(10, 5))
ax_orig.imshow(original_image)
ax_orig.set_title("Original Image (24-bit Color)")
ax_orig.axis('off')

ax_comp.imshow(compressed_image)
ax_comp.set_title(f"Compressed Image ({K_colors} Colors)")
ax_comp.axis('off')
plt.show()


# =====================================================================
# Exercise 5.3: Comparing Algorithms on Different Shapes
# =====================================================================
print("\n==================================================")
print("EXERCISE 5.3: COMPARING CLUSTERING ALGORITHMS")
print("==================================================")

# Tạo 3 loại cấu trúc hình dáng phân cụm hình học kinh điển
n_samples_geo = 300
datasets = {
    'Blobs (Spherical)': make_blobs(n_samples=n_samples_geo, centers=3, cluster_std=0.5, random_state=42)[0],
    'Moons (Non-linear)': make_moons(n_samples=n_samples_geo, noise=0.05, random_state=42)[0],
    'Circles (Concentric)': make_circles(n_samples=n_samples_geo, factor=0.5, noise=0.05, random_state=42)[0]
}

# Khởi tạo các đại diện thuật toán đại diện cho 3 trường phái phân cụm khác nhau
algo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
algo_hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
algo_dbscan = DBSCAN(eps=0.25, min_samples=5) # Lưu ý: Cần chỉnh tinh chỉnh thông số hình học hình học phù hợp

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
row_idx = 0

for name, X_geo in datasets.items():
    X_geo_scaled = StandardScaler().fit_transform(X_geo)
    
    # 1. Chạy mô hình K-Means
    if 'Blobs' not in name: # Cấu trúc dữ liệu hình tròn hoặc hình bán nguyệt chỉ định 2 cụm nhãn gốc
        algo_kmeans.n_clusters = 2
    else:
        algo_kmeans.n_clusters = 3
    lbl_km = algo_kmeans.fit_predict(X_geo_scaled)
    axes[row_idx, 0].scatter(X_geo_scaled[:, 0], X_geo_scaled[:, 1], c=lbl_km, cmap='plasma', s=15)
    if row_idx == 0: axes[row_idx, 0].set_title('K-Means')
    axes[row_idx, 0].set_ylabel(name, fontsize=10, fontweight='bold')
    
    # 2. Chạy mô hình Phân cụm Phân cấp (Hierarchical Clustering)
    if 'Blobs' not in name:
        algo_hierarchical.n_clusters = 2
    else:
        algo_hierarchical.n_clusters = 3
    lbl_hc = algo_hierarchical.fit_predict(X_geo_scaled)
    axes[row_idx, 1].scatter(X_geo_scaled[:, 0], X_geo_scaled[:, 1], c=lbl_hc, cmap='plasma', s=15)
    if row_idx == 0: axes[row_idx, 1].set_title('Hierarchical')
    
    # 3. Chạy mô hình DBSCAN mật độ
    # Tinh chỉnh riêng eps động một chút giúp hình khối trơn dễ gom cụm hơn
    if 'Circles' in name:
        lbl_db = DBSCAN(eps=0.4, min_samples=5).fit_predict(X_geo_scaled)
    elif 'Moons' in name:
        lbl_db = DBSCAN(eps=0.4, min_samples=5).fit_predict(X_geo_scaled)
    else:
        lbl_db = algo_dbscan.fit_predict(X_geo_scaled)
        
    axes[row_idx, 2].scatter(X_geo_scaled[:, 0], X_geo_scaled[:, 1], c=lbl_db, cmap='plasma', s=15)
    if row_idx == 0: axes[row_idx, 2].set_title('DBSCAN')
    
    row_idx += 1

plt.tight_layout()
plt.show()

print("\nNhận xét đánh giá:")
print("- K-Means và Hierarchical (Ward) hoạt động cực tốt trên cụm dạng khối tròn cầu (Blobs) nhưng thất bại trên cấu trúc Moons và Circles.")
print("- DBSCAN nhận diện xuất sắc các hình khối phi tuyến tính đa dạng, chấp nhận dị biệt nhiễu mà không bắt buộc cấu trúc phân bổ cụm cố định đối xứng.")