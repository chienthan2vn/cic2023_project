import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

def _preprocess_cic2023_mini():
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('data/cic2023_mini.csv')

    # Preprocessing dữ liệu CIC 2023 mini
    print("=== PREPROCESSING DỮ LIỆU CIC 2023 MINI ===")

    # Tách features và labels
    X = df.drop('Label', axis=1).values
    y_true = df['Label'].values

    # Encode labels thành số
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_true)

    # print(f"Features shape: {X.shape}")
    # print(f"Labels: {label_encoder.classes_}")
    # print(f"Label distribution: {np.bincount(y_encoded)}")

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # print(f"Data scaled - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    return X_scaled, y_encoded

def _preprocess_cic2023_mini_for_pca(dim=2):
    X_scaled, y_encoded = _preprocess_cic2023_mini()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y_encoded

# # Sử dụng PCA để giảm chiều cho visualization
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
# print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# # Visualization dữ liệu gốc
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# colors = ['blue', 'red', 'green', 'orange', 'purple']
# for i, label in enumerate(label_encoder.classes_):
#     mask = y_encoded == i
#     plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
#                 c=colors[i], label=label, alpha=0.6, s=30)
# plt.title('CIC 2023 Mini Dataset (PCA)')
# plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
# plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.subplot(1, 2, 2)
# # Heatmap correlation
# feature_names = df.columns[:-1]
# correlation_matrix = df[feature_names].corr()
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
# plt.title('Feature Correlation Matrix')
# plt.tight_layout()
# plt.show()

# print("\n=== CHUẨN BỊ CHO FCM ===")
# print("✅ Dữ liệu đã được chuẩn hóa")
# print("✅ PCA đã được áp dụng cho visualization")
# print("✅ Sẵn sàng cho FCM clustering")