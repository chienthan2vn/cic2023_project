import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from preprocess import _preprocess_cic2023_mini
import warnings
warnings.filterwarnings('ignore')

class FCMEvaluator:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.data_path = "./data/0.01percent_34classes.csv"
        self.X = None
        self.original_labels = None
        self.evaluation_results = []
        
    def load_data(self):
        """Load và preprocess dữ liệu gốc"""
        print("🔄 Đang load và preprocess dữ liệu...")
        self.X, y_encoded = _preprocess_cic2023_mini(self.data_path)
        
        # Load lại để lấy original labels
        df = pd.read_csv(self.data_path)
        self.original_labels = df['label'].copy()
        
        print(f"✅ Đã load dữ liệu: {self.X.shape}")
        print(f"📊 Số lượng label gốc: {len(self.original_labels.unique())}")
        return self.X, self.original_labels
    
    def calculate_wcss(self, data, centroids, labels):
        """Tính Within-Cluster Sum of Squares (WCSS) cho Elbow Method"""
        wcss = 0
        for i in range(len(centroids)):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[i]) ** 2)
        return wcss
    
    def calculate_silhouette_score(self, data, labels):
        """Tính Silhouette Score"""
        if len(np.unique(labels)) < 2:
            return -1  # Không thể tính silhouette với ít hơn 2 clusters
        try:
            return silhouette_score(data, labels)
        except:
            return -1
    
    def calculate_partition_coefficient(self, membership_matrix):
        """Tính Partition Coefficient (PC) - Validity Index cho FCM"""
        n = membership_matrix.shape[0]
        pc = np.sum(membership_matrix ** 2) / n
        return pc
    
    def calculate_partition_entropy(self, membership_matrix):
        """Tính Partition Entropy (PE) - Validity Index cho FCM"""
        n = membership_matrix.shape[0]
        # Tránh log(0) bằng cách thêm epsilon nhỏ
        epsilon = 1e-10
        membership_matrix = np.clip(membership_matrix, epsilon, 1 - epsilon)
        pe = -np.sum(membership_matrix * np.log(membership_matrix)) / n
        return pe
    
    def calculate_xie_beni_index(self, data, membership_matrix, centroids, m=2):
        """Tính Xie-Beni Index (XB) - Validity Index cho FCM"""
        try:
            n, c = membership_matrix.shape
            
            # Tính numerator: tổng của (membership^m * distance^2)
            numerator = 0
            for i in range(n):
                for j in range(c):
                    dist_sq = np.sum((data[i] - centroids[j]) ** 2)
                    numerator += (membership_matrix[i, j] ** m) * dist_sq
            
            # Tính denominator: n * min distance giữa các centroids
            min_centroid_dist = float('inf')
            for i in range(c):
                for j in range(i+1, c):
                    dist = np.sum((centroids[i] - centroids[j]) ** 2)
                    if dist < min_centroid_dist:
                        min_centroid_dist = dist
            
            if min_centroid_dist == 0:
                return float('inf')
            
            xb = numerator / (n * min_centroid_dist)
            return xb
        except:
            return float('inf')
    
    def evaluate_single_experiment(self, experiment_folder):
        """Đánh giá một thử nghiệm cụ thể"""
        try:
            # Load experiment info
            info_path = os.path.join(experiment_folder, 'experiment_info.json')
            with open(info_path, 'r') as f:
                experiment_info = json.load(f)
            
            # Load results
            results_path = os.path.join(experiment_folder, 'results.pkl')
            with open(results_path, 'rb') as f:
                results_data = pickle.load(f)
            
            membership_matrix = results_data['membership_matrix']
            centroids = results_data['centroids']
            predicted_labels = results_data['predicted_labels']
            
            # Tính các metrics nhanh
            wcss = self.calculate_wcss(self.X, centroids, predicted_labels)
            silhouette = self.calculate_silhouette_score(self.X, predicted_labels)
            
            # Comment lại các thuật toán chạy lâu
            # pc = self.calculate_partition_coefficient(membership_matrix)
            # pe = self.calculate_partition_entropy(membership_matrix)
            # xb = self.calculate_xie_beni_index(self.X, membership_matrix, centroids, 
            #                                  experiment_info['parameters']['m'])
            
            evaluation = {
                'experiment_id': experiment_info['experiment_id'],
                'parameters': experiment_info['parameters'],
                'metrics': {
                    'wcss': float(wcss),
                    'silhouette_score': float(silhouette),
                    # 'partition_coefficient': float(pc),
                    # 'partition_entropy': float(pe),
                    # 'xie_beni_index': float(xb)
                },
                'training_time': experiment_info['results']['training_time_seconds']
            }
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {experiment_folder}: {str(e)}")
            return None
    
    def evaluate_all_experiments(self):
        """Đánh giá tất cả các thử nghiệm trong thư mục"""
        if self.X is None:
            self.load_data()
        
        # Tìm tất cả thư mục experiment
        experiment_folders = []
        for item in os.listdir(self.experiment_dir):
            if item.startswith('experiment_') and os.path.isdir(os.path.join(self.experiment_dir, item)):
                experiment_folders.append(os.path.join(self.experiment_dir, item))
        
        experiment_folders.sort()
        print(f"🔍 Tìm thấy {len(experiment_folders)} thử nghiệm để đánh giá")
        
        self.evaluation_results = []
        for i, folder in enumerate(experiment_folders, 1):
            print(f"⚙️  Đánh giá thử nghiệm {i}/{len(experiment_folders)}: {os.path.basename(folder)}")
            evaluation = self.evaluate_single_experiment(folder)
            if evaluation:
                self.evaluation_results.append(evaluation)
        
        print(f"✅ Hoàn thành đánh giá {len(self.evaluation_results)} thử nghiệm")
        return self.evaluation_results
    
    def create_elbow_plots(self):
        """Tạo Elbow plots cho từng distance metric và m value"""
        if not self.evaluation_results:
            print("⚠️  Chưa có kết quả đánh giá. Hãy chạy evaluate_all_experiments() trước.")
            return
        
        print("📈 Đang tạo Elbow plots...")
        
        # Tạo DataFrame từ kết quả
        df_results = []
        for result in self.evaluation_results:
            if 'error' not in result:
                row = {**result['parameters'], **result['metrics']}
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Tạo plots cho từng combination của distance_metric và m
        distance_metrics = df['distance_metric'].unique()
        m_values = df['m'].unique()
        
        fig, axes = plt.subplots(len(distance_metrics), len(m_values), 
                                figsize=(5*len(m_values), 4*len(distance_metrics)))
        
        if len(distance_metrics) == 1 and len(m_values) == 1:
            axes = [[axes]]
        elif len(distance_metrics) == 1:
            axes = [axes]
        elif len(m_values) == 1:
            axes = [[ax] for ax in axes]
        
        for i, distance_metric in enumerate(distance_metrics):
            for j, m in enumerate(m_values):
                subset = df[(df['distance_metric'] == distance_metric) & (df['m'] == m)]
                if len(subset) > 0:
                    # Group by n_clusters và tính mean WCSS
                    elbow_data = subset.groupby('n_clusters')['wcss'].mean().reset_index()
                    
                    axes[i][j].plot(elbow_data['n_clusters'], elbow_data['wcss'], 'bo-', linewidth=2, markersize=6)
                    axes[i][j].set_title(f'Elbow Method\n{distance_metric}, m={m}', fontsize=12, fontweight='bold')
                    axes[i][j].set_xlabel('Number of Clusters')
                    axes[i][j].set_ylabel('WCSS')
                    axes[i][j].grid(True, alpha=0.3)
                    axes[i][j].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        save_path = os.path.join(self.experiment_dir, 'elbow_plots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu Elbow plots tại: {save_path}")
    
    def create_evaluation_summary(self):
        """Tạo bảng tổng hợp kết quả đánh giá"""
        if not self.evaluation_results:
            print("⚠️  Chưa có kết quả đánh giá.")
            return
        
        print("📊 Đang tạo tổng hợp kết quả...")
        
        # Tạo DataFrame
        df_results = []
        for result in self.evaluation_results:
            if 'error' not in result:
                row = {
                    'experiment_id': result['experiment_id'],
                    'n_clusters': result['parameters']['n_clusters'],
                    'm': result['parameters']['m'],
                    'tol': result['parameters']['tol'],
                    'random_state': result['parameters']['random_state'],
                    'distance_metric': result['parameters']['distance_metric'],
                    'wcss': result['metrics']['wcss'],
                    'silhouette_score': result['metrics']['silhouette_score'],
                    'partition_coefficient': result['metrics']['partition_coefficient'],
                    'partition_entropy': result['metrics']['partition_entropy'],
                    'xie_beni_index': result['metrics']['xie_beni_index'],
                    'training_time': result['training_time']
                }
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Lưu kết quả chi tiết
        csv_path = os.path.join(self.experiment_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Đã lưu kết quả chi tiết tại: {csv_path}")
        
        # Tạo summary statistics
        summary_stats = {
            'total_experiments': len(df),
            'best_silhouette': {
                'score': float(df['silhouette_score'].max()),
                'experiment_id': int(df.loc[df['silhouette_score'].idxmax(), 'experiment_id']),
                'parameters': df.loc[df['silhouette_score'].idxmax(), 
                                   ['n_clusters', 'm', 'distance_metric', 'random_state']].to_dict()
            },
            'best_partition_coefficient': {
                'score': float(df['partition_coefficient'].max()),
                'experiment_id': int(df.loc[df['partition_coefficient'].idxmax(), 'experiment_id']),
                'parameters': df.loc[df['partition_coefficient'].idxmax(), 
                                   ['n_clusters', 'm', 'distance_metric', 'random_state']].to_dict()
            },
            'best_xie_beni': {
                'score': float(df['xie_beni_index'].min()),
                'experiment_id': int(df.loc[df['xie_beni_index'].idxmin(), 'experiment_id']),
                'parameters': df.loc[df['xie_beni_index'].idxmin(), 
                                   ['n_clusters', 'm', 'distance_metric', 'random_state']].to_dict()
            },
            'lowest_partition_entropy': {
                'score': float(df['partition_entropy'].min()),
                'experiment_id': int(df.loc[df['partition_entropy'].idxmin(), 'experiment_id']),
                'parameters': df.loc[df['partition_entropy'].idxmin(), 
                                   ['n_clusters', 'm', 'distance_metric', 'random_state']].to_dict()
            }
        }
        
        # Lưu summary
        summary_path = os.path.join(self.experiment_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Đã lưu tổng hợp đánh giá tại: {summary_path}")
        
        return df, summary_stats
    
    def create_metric_heatmaps(self):
        """Tạo heatmaps cho các metrics theo n_clusters và m"""
        if not self.evaluation_results:
            print("⚠️  Chưa có kết quả đánh giá.")
            return
        
        print("🔥 Đang tạo metric heatmaps...")
        
        # Tạo DataFrame
        df_results = []
        for result in self.evaluation_results:
            if 'error' not in result:
                row = {**result['parameters'], **result['metrics']}
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        metrics = ['silhouette_score', 'partition_coefficient', 'partition_entropy', 'xie_beni_index']
        metric_labels = ['Silhouette Score', 'Partition Coefficient (PC)', 'Partition Entropy (PE)', 'Xie-Beni Index (XB)']
        distance_metrics = df['distance_metric'].unique()
        
        fig, axes = plt.subplots(len(distance_metrics), len(metrics), 
                                figsize=(5*len(metrics), 4*len(distance_metrics)))
        
        if len(distance_metrics) == 1:
            axes = [axes]
        
        for i, distance_metric in enumerate(distance_metrics):
            subset = df[df['distance_metric'] == distance_metric]
            
            for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
                # Tạo pivot table
                pivot_data = subset.pivot_table(values=metric, index='m', columns='n_clusters', aggfunc='mean')
                
                # Tạo heatmap
                sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=axes[i][j], 
                           cmap='viridis', cbar_kws={'label': label})
                axes[i][j].set_title(f'{label}\n({distance_metric})', fontweight='bold')
                axes[i][j].set_xlabel('Number of Clusters')
                axes[i][j].set_ylabel('Fuzziness (m)')
        
        plt.tight_layout()
        save_path = os.path.join(self.experiment_dir, 'metric_heatmaps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu Metric heatmaps tại: {save_path}")

    def create_comparison_plots(self):
        """Tạo plots so sánh các metrics"""
        if not self.evaluation_results:
            print("⚠️  Chưa có kết quả đánh giá.")
            return
        
        print("📊 Đang tạo comparison plots...")
        
        # Tạo DataFrame
        df_results = []
        for result in self.evaluation_results:
            if 'error' not in result:
                row = {**result['parameters'], **result['metrics']}
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Tạo subplot cho từng metric theo số clusters
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['silhouette_score', 'partition_coefficient', 'partition_entropy', 'xie_beni_index']
        titles = ['Silhouette Score vs N_Clusters', 'Partition Coefficient vs N_Clusters', 
                 'Partition Entropy vs N_Clusters', 'Xie-Beni Index vs N_Clusters']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            for distance_metric in df['distance_metric'].unique():
                subset = df[df['distance_metric'] == distance_metric]
                grouped = subset.groupby('n_clusters')[metric].agg(['mean', 'std']).reset_index()
                
                axes[idx].errorbar(grouped['n_clusters'], grouped['mean'], 
                                 yerr=grouped['std'], label=distance_metric, 
                                 marker='o', capsize=5, capthick=2)
            
            axes[idx].set_title(title, fontweight='bold')
            axes[idx].set_xlabel('Number of Clusters')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.experiment_dir, 'metric_comparison_plots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu Comparison plots tại: {save_path}")

def evaluate_fcm_experiments(experiment_dir):
    """Function chính để đánh giá các thử nghiệm FCM"""
    print(f"🚀 Bắt đầu đánh giá thử nghiệm tại: {experiment_dir}")
    
    evaluator = FCMEvaluator(experiment_dir)
    
    # Load dữ liệu
    evaluator.load_data()
    
    # Đánh giá tất cả thử nghiệm
    evaluator.evaluate_all_experiments()
    
    # Tạo các báo cáo và visualizations
    evaluator.create_elbow_plots()
    evaluator.create_metric_heatmaps()
    evaluator.create_comparison_plots()
    df, summary = evaluator.create_evaluation_summary()
    
    print("\n" + "="*60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP")
    print("="*60)
    print(f"📈 Tổng số thử nghiệm đánh giá: {summary['total_experiments']}")
    print(f"🏆 Silhouette Score tốt nhất: {summary['best_silhouette']['score']:.4f} (experiment {summary['best_silhouette']['experiment_id']})")
    print(f"   Tham số: {summary['best_silhouette']['parameters']}")
    print(f"🏆 Partition Coefficient tốt nhất: {summary['best_partition_coefficient']['score']:.4f} (experiment {summary['best_partition_coefficient']['experiment_id']})")
    print(f"   Tham số: {summary['best_partition_coefficient']['parameters']}")
    print(f"🏆 Xie-Beni Index tốt nhất: {summary['best_xie_beni']['score']:.4f} (experiment {summary['best_xie_beni']['experiment_id']})")
    print(f"   Tham số: {summary['best_xie_beni']['parameters']}")
    print(f"🏆 Partition Entropy tốt nhất: {summary['lowest_partition_entropy']['score']:.4f} (experiment {summary['lowest_partition_entropy']['experiment_id']})")
    print(f"   Tham số: {summary['lowest_partition_entropy']['parameters']}")
    print("="*60)
    
    return evaluator, df, summary

def find_latest_experiment_dir():
    """Tìm thư mục thử nghiệm mới nhất"""
    test_cluster_dir = "./test_cluster"
    if not os.path.exists(test_cluster_dir):
        return None
    
    experiment_dirs = []
    for item in os.listdir(test_cluster_dir):
        if item.startswith('fcm_experiment_') and os.path.isdir(os.path.join(test_cluster_dir, item)):
            experiment_dirs.append(os.path.join(test_cluster_dir, item))
    
    if experiment_dirs:
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        experiment_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        return experiment_dirs[0]
    
    return None

if __name__ == "__main__":
    # Tự động tìm thư mục thử nghiệm mới nhất
    experiment_dir = find_latest_experiment_dir()
    
    if experiment_dir:
        print(f"🔍 Tìm thấy thư mục thử nghiệm: {experiment_dir}")
        evaluator, df, summary = evaluate_fcm_experiments(experiment_dir)
    else:
        print("❌ Không tìm thấy thư mục thử nghiệm nào!")
        print("📝 Hãy chạy fcm_experiment.py trước để tạo dữ liệu thử nghiệm.")
        
        # Hiển thị thư mục có sẵn
        test_cluster_dir = "./test_cluster"
        if os.path.exists(test_cluster_dir):
            print("\n📁 Các thư mục có sẵn trong test_cluster:")
            for item in os.listdir(test_cluster_dir):
                if os.path.isdir(os.path.join(test_cluster_dir, item)):
                    print(f"   - {item}")
