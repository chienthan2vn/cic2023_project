import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

def preprocess_dataframe(df):
    df_copy = df.copy()

    # Apply frequency encoding to 'protocol_type'
    protocol_type_counts = df_copy['protocol_type'].value_counts()
    df_copy['protocol_type'] = df_copy['protocol_type'].map(protocol_type_counts)

    # Apply RobustScaler to continuous data
    continuous_columns = df_copy.select_dtypes(include=[np.number]).columns
    scaler = RobustScaler()
    df_copy[continuous_columns] = scaler.fit_transform(df_copy[continuous_columns])

    # Binary encode boolean data
    bool_columns = df_copy.select_dtypes(include=['bool']).columns
    encoder = LabelEncoder()
    for col in bool_columns:
        df_copy[col] = encoder.fit_transform(df_copy[col])

    # Normalize all features except 'label' using StandardScaler
    feature_columns = [col for col in df_copy.columns if col != 'label']
    standard_scaler = StandardScaler()
    df_copy[feature_columns] = standard_scaler.fit_transform(df_copy[feature_columns])

    return df_copy

def _preprocess_cic2023_mini(path: str):
    df = pd.read_csv(path)
    df = preprocess_dataframe(df)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded

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