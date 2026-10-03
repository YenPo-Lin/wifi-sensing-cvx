import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.lines as mlines

# ==========================================
# 1. 生成模擬的 3D 點雲資料 (包含群集與雜訊)
# ==========================================
centers = [[2, 2, 2], [-2, -2, -2], [2, -2, 2]]
X, _ = make_blobs(n_samples=600, centers=centers, cluster_std=0.5, random_state=42)

np.random.seed(42)
noise = np.random.uniform(low=-4, high=4, size=(100, 3))
X = np.vstack((X, noise))

# ==========================================
# 2. 執行 DBSCAN 分群
# ==========================================
dbscan = DBSCAN(eps=0.8, min_samples=10)
labels = dbscan.fit_predict(X)

# ==========================================
# 3. 視覺化 (Z軸上色 + 標示重心)
# ==========================================
z_min, z_max = X[:, 2].min(), X[:, 2].max()

fig = plt.figure(figsize=(16, 7))

# --- 分群前 (Before) ---
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], 
                  c=X[:, 2], cmap='bwr', vmin=z_min, vmax=z_max, 
                  marker='o', s=15, alpha=0.8)
ax1.set_title('Before: Raw Point Cloud (Colored by Z-axis)', fontsize=14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
fig.colorbar(sc1, ax=ax1, label='Z Elevation', shrink=0.6, pad=0.1)

# --- 分群後 (After) ---
ax2 = fig.add_subplot(122, projection='3d')
unique_labels = set(labels)

for k in unique_labels:
    class_member_mask = (labels == k)
    xyz = X[class_member_mask]
    
    if k == -1:
        # 雜訊 (Noise)
        ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                    c='black', marker='x', s=15, alpha=0.3)
    else:
        # 有效群集點 (依 Z 軸上色)
        sc2 = ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                          c=xyz[:, 2], cmap='bwr', vmin=z_min, vmax=z_max, 
                          marker='o', s=20, alpha=0.9)
        
        # ==========================================
        # 計算並畫出重心 (Centroid)
        # ==========================================
        # 沿著 axis=0 計算所有該群集點的 X, Y, Z 平均值
        centroid = np.mean(xyz, axis=0)
        
        # 使用金色大星星繪製重心，zorder=10 確保它顯示在最上層
        ax2.scatter(centroid[0], centroid[1], centroid[2],
                    c='gold', marker='*', s=300, edgecolor='black', 
                    linewidth=1.5, zorder=10)

ax2.set_title('After: DBSCAN with Centroids', fontsize=14)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
fig.colorbar(sc2, ax=ax2, label='Z Elevation', shrink=0.6, pad=0.1)

# 自訂 Legend (加入重心圖例)
cluster_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                               markersize=6, label='Clustered Points')
noise_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                             markersize=6, label='Noise (-1)', alpha=0.5)
centroid_marker = mlines.Line2D([], [], color='gold', markeredgecolor='black', marker='*', 
                                linestyle='None', markersize=15, label='Centroid')

ax2.legend(handles=[cluster_marker, centroid_marker, noise_marker], loc='upper left')

plt.tight_layout()
plt.show()