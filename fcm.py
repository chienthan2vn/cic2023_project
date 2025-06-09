import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Định nghĩa lại các class FCM
class FCM:
    def __init__(self, n_clusters=3, m=2, max_iter=100, tol=1e-4, random_state=42, distance_metric='euclidean'):
        self.n_clusters = n_clusters
        self.m = m  # fuzziness parameter
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.mapping = {
            'euclidean': self.euclidean_distance,
            'canberra': self.canberra_distance,
            'manhattan': self.manhattan_distance,
            'minkowski': self.minkowski_distance
        }
        self.distance_metric = self.mapping.get(distance_metric, self.euclidean_distance)
        
    def euclidean_distance(self, x, y):
        """Tính khoảng cách Euclidean"""
        return np.sqrt(np.sum((x - y) ** 2, axis=1))
    
    def canberra_distance(self, x, y):
        """Tính khoảng cách Canberra"""
        numerator = np.abs(x - y)
        denominator = np.abs(x) + np.abs(y) + 1e-10  # Tránh chia cho 0
        distance = np.sum(numerator / denominator, axis=1)
        return distance
    
    def manhattan_distance(self, x, y):
        """Tính khoảng cách Manhattan"""
        return np.sum(np.abs(x - y), axis=1)
    
    def minkowski_distance(self, x, y, p=3):
        """Tính khoảng cách Minkowski với tham số p"""
        return np.sum(np.abs(x - y) ** p, axis=1) ** (1 / p)
    
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Khởi tạo ma trận membership ngẫu nhiên
        self.u = np.random.rand(n_samples, self.n_clusters)
        self.u = self.u / np.sum(self.u, axis=1, keepdims=True)
        
        # Khởi tạo centroids
        self.centroids = np.zeros((self.n_clusters, n_features))
        
        for iteration in range(self.max_iter):
            # Cập nhật centroids
            old_centroids = self.centroids.copy()
            
            for i in range(self.n_clusters):
                numerator = np.sum((self.u[:, i] ** self.m).reshape(-1, 1) * X, axis=0)
                denominator = np.sum(self.u[:, i] ** self.m)
                self.centroids[i] = numerator / denominator
            
            # Cập nhật membership matrix
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    distances = []
                    for k in range(self.n_clusters):
                        dist = self.distance_metric(X[i:i+1], self.centroids[k:k+1])[0]
                        distances.append(dist)
                    
                    if distances[j] == 0:
                        self.u[i, j] = 1.0
                        for k in range(self.n_clusters):
                            if k != j:
                                self.u[i, k] = 0.0
                    else:
                        sum_term = 0
                        for k in range(self.n_clusters):
                            if distances[k] > 0:
                                sum_term += (distances[j] / distances[k]) ** (2 / (self.m - 1))
                        self.u[i, j] = 1 / sum_term
            
            # Kiểm tra hội tụ
            if np.allclose(old_centroids, self.centroids, atol=self.tol):
                print(f"FCM Euclidean hội tụ sau {iteration + 1} iterations")
                break
        
        return self
    
    def predict(self, X):
        """Dự đoán cluster cho dữ liệu mới"""
        labels = np.argmax(self.u, axis=1)
        return labels
