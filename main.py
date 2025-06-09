import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from fcm import FCM
from preprocess import _preprocess_cic2023_mini

print("=== ÁP DỤNG FCM TRÊN CIC 2023 MINI ===\n")

data_path = "/data/0.01percent_34classes.csv"
x, y_encoded = _preprocess_cic2023_mini(data_path)

# 1. FCM Euclidean trên dữ liệu đầy đủ (10 features)
print("1. FCM EUCLIDEAN - Full Features (10D):")
fcm_euc_full = FCM(n_clusters=5, m=2, max_iter=50, tol=1e-4, distance_metric='euclidean')
fcm_euc_full.fit(x)
labels_euc_full = fcm_euc_full.predict(x)

# 2. FCM Canberra trên dữ liệu đầy đủ (10 features)
print("\n2. FCM CANBERRA - Full Features (10D):")
fcm_can_full = FCM(n_clusters=5, m=2, max_iter=50, tol=1e-4, distance_metric='canberra')
fcm_can_full.fit(x)
labels_can_full = fcm_can_full.predict(x)

print("\n✅ Tất cả thuật toán FCM đã hoàn thành!")

# Đánh giá kết quả
print("\n=== ĐÁNH GIÁ KẾT QUẢ ===")

# Adjusted Rand Index
ari_euc_full = adjusted_rand_score(y_encoded, labels_euc_full)
ari_can_full = adjusted_rand_score(y_encoded, labels_can_full)

print("Adjusted Rand Index (so với ground truth):")
print(f"- FCM Euclidean (Full): {ari_euc_full:.4f}")
print(f"- FCM Canberra (Full):  {ari_can_full:.4f}")


# Silhouette Score
sil_euc_full = silhouette_score(x, labels_euc_full)
sil_can_full = silhouette_score(x, labels_can_full)

print("\nSilhouette Score (chất lượng clustering):")
print(f"- FCM Euclidean (Full): {sil_euc_full:.4f}")
print(f"- FCM Canberra (Full):  {sil_can_full:.4f}")

# Phân tích cluster distribution
print("\n=== PHÂN TÍCH CLUSTER DISTRIBUTION ===")
print("Ground Truth distribution:", np.bincount(y_encoded))
print("FCM Euclidean (Full):", np.bincount(labels_euc_full))
print("FCM Canberra (Full):", np.bincount(labels_can_full))

# Tìm thuật toán tốt nhất

print(f"\n🏆 THUẬT TOÁN TỐT NHẤT:")
algorithms = [
    ("FCM Euclidean (Full)", ari_euc_full, sil_euc_full),
    ("FCM Canberra (Full)", ari_can_full, sil_can_full),
]

best_ari_algo = max(algorithms, key=lambda x: x[1])
best_sil_algo = max(algorithms, key=lambda x: x[2])

print(f"   ARI: {best_ari_algo[0]} - {best_ari_algo[1]:.4f}")
print(f"   Silhouette: {best_sil_algo[0]} - {best_sil_algo[2]:.4f}")