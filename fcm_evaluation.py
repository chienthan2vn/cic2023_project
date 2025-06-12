import pandas as pd
import numpy as np
import pickle
import json
import os
import time
import psutil
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from preprocess import _preprocess_cic2023_mini
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FCMEvaluator:
    def __init__(self, experiment_dir, data_path=None, max_samples_for_silhouette=10000):
        """
        Initialize FCM Evaluator with critical improvements
        
        Args:
            experiment_dir: Directory containing experiments
            data_path: Path to dataset (flexible)
            max_samples_for_silhouette: Max samples for silhouette calculation
        """
        self.experiment_dir = experiment_dir
        self.data_path = data_path or "./data/0.01percent_34classes.csv"
        self.max_samples_for_silhouette = max_samples_for_silhouette
        self.X = None
        self.original_labels = None
        self.evaluation_results = []
        self.dataset_size = 0
        self.memory_usage = {}
        
        logger.info(f"Initialized FCMEvaluator with data_path: {self.data_path}")
        
    def _get_memory_usage(self):
        """Monitor memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def _validate_data_path(self):
        """Validate data path exists"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        logger.info(f"Data file validated: {self.data_path}")
        
    def load_data(self):
        """Load và preprocess dữ liệu gốc - MEMORY OPTIMIZED"""
        try:
            self._validate_data_path()
            
            start_memory = self._get_memory_usage()
            logger.info("🔄 Loading and preprocessing data...")
            
            # Load data efficiently - avoid duplicate loading
            start_time = time.time()
            
            # First load to get size info
            df_info = pd.read_csv(self.data_path, nrows=1000)  # Sample to check structure
            total_rows = sum(1 for line in open(self.data_path)) - 1  # Count lines efficiently
            self.dataset_size = total_rows
            
            logger.info(f"Dataset size detected: {self.dataset_size:,} samples")
            
            # Load data based on size
            if self.dataset_size > 500000:  # Large dataset
                logger.warning("Large dataset detected. Consider using a sample for initial analysis.")
            
            # Preprocess data
            self.X, y_encoded = _preprocess_cic2023_mini(self.data_path)
            
            # Load original labels efficiently (only label column)
            df_labels = pd.read_csv(self.data_path, usecols=['label'])
            self.original_labels = df_labels['label'].copy()
            del df_labels  # Free memory immediately
            
            end_memory = self._get_memory_usage()
            load_time = time.time() - start_time
            
            self.memory_usage['data_loading'] = end_memory - start_memory
            
            logger.info(f"✅ Data loaded: {self.X.shape}")
            logger.info(f"📊 Unique labels: {len(self.original_labels.unique())}")
            logger.info(f"⏱️  Load time: {load_time:.2f}s")
            logger.info(f"💾 Memory usage: {self.memory_usage['data_loading']:.2f}MB")
            
            return self.X, self.original_labels
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_wcss(self, data, centroids, labels):
        """Tính Within-Cluster Sum of Squares (WCSS) cho Elbow Method - OPTIMIZED"""
        # Tối ưu hóa cho dữ liệu lớn bằng vectorization
        wcss = 0
        unique_labels = np.unique(labels)
        
        for i in unique_labels:
            if i < len(centroids):  # Đảm bảo index hợp lệ
                cluster_mask = labels == i
                if np.sum(cluster_mask) > 0:
                    cluster_points = data[cluster_mask]
                    # Vectorized calculation
                    distances_sq = np.sum((cluster_points - centroids[i]) ** 2, axis=1)
                    wcss += np.sum(distances_sq)
        return wcss
    
    def calculate_silhouette_score(self, data, labels):
        """Tính Silhouette Score - ADAPTIVE SAMPLING based on dataset size"""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                logger.warning("Cannot calculate silhouette score with less than 2 clusters")
                return -1
            
            n_samples = len(data)
            
            # Adaptive strategy based on dataset size
            if n_samples <= 5000:
                # Small dataset: full calculation
                logger.info("Small dataset: Computing full silhouette score")
                return silhouette_score(data, labels)
                
            elif n_samples <= self.max_samples_for_silhouette:
                # Medium dataset: sample calculation
                sample_size = min(5000, n_samples // 2)
                logger.info(f"Medium dataset: Sampling {sample_size} points for silhouette score")
                indices = np.random.choice(n_samples, sample_size, replace=False)
                return silhouette_score(data[indices], labels[indices])
                
            else:
                # Large dataset: disable or very small sample
                if n_samples > 100000:
                    logger.warning(f"Large dataset ({n_samples:,} samples): Silhouette score disabled for performance")
                    return -1
                else:
                    # Very small sample for datasets 10k-100k
                    sample_size = 2000
                    logger.info(f"Large dataset: Using small sample ({sample_size}) for silhouette score")
                    indices = np.random.choice(n_samples, sample_size, replace=False)
                    return silhouette_score(data[indices], labels[indices])
                    
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {str(e)}")
            return -1
    
    def calculate_partition_coefficient(self, membership_matrix):
        """Tính Partition Coefficient (PC) - FAST and efficient"""
        n = membership_matrix.shape[0]
        # Vectorized calculation
        pc = np.sum(membership_matrix ** 2) / n
        return pc
    
    def calculate_partition_entropy(self, membership_matrix):
        """Tính Partition Entropy (PE) - FAST and efficient"""
        n = membership_matrix.shape[0]
        # Tránh log(0) bằng cách thêm epsilon nhỏ
        epsilon = 1e-10
        membership_matrix_safe = np.clip(membership_matrix, epsilon, 1 - epsilon)
        # Vectorized calculation
        pe = -np.sum(membership_matrix_safe * np.log(membership_matrix_safe)) / n
        return pe
    
    def calculate_compactness_separation(self, data, membership_matrix, centroids, m=2):
        """Tính Compactness and Separation - ALTERNATIVE to slow metrics"""
        try:
            n, c = membership_matrix.shape
            
            # Compactness: average weighted distance to centroids
            compactness = 0
            for j in range(c):
                distances_sq = np.sum((data - centroids[j]) ** 2, axis=1)
                weighted_distances = (membership_matrix[:, j] ** m) * distances_sq
                compactness += np.sum(weighted_distances)
            compactness /= n
            
            # Separation: minimum distance between centroids
            min_separation = float('inf')
            for i in range(c):
                for j in range(i+1, c):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    if dist < min_separation:
                        min_separation = dist
            
            # Compactness/Separation ratio (lower is better)
            if min_separation == 0:
                return float('inf')
            
            cs_ratio = compactness / min_separation
            return compactness, min_separation, cs_ratio
        except:
            return float('inf'), 0, float('inf')
    
    def calculate_xie_beni_index(self, data, membership_matrix, centroids, m=2):
        """Tính Xie-Beni Index (XB) - CRITICALLY OPTIMIZED with early termination"""
        try:
            start_time = time.time()
            n, c = membership_matrix.shape
            
            # Validate inputs
            if n == 0 or c == 0:
                logger.error("Invalid membership matrix dimensions")
                return float('inf')
            
            if len(centroids) != c:
                logger.error(f"Centroids count ({len(centroids)}) doesn't match clusters ({c})")
                return float('inf')
            
            # Early termination for very large datasets
            if n > 200000:
                logger.warning(f"Very large dataset ({n:,}). Using sampling for XB calculation")
                sample_size = min(50000, n)
                indices = np.random.choice(n, sample_size, replace=False)
                data_sample = data[indices]
                membership_sample = membership_matrix[indices]
                return self._calculate_xb_core(data_sample, membership_sample, centroids, m)
            
            result = self._calculate_xb_core(data, membership_matrix, centroids, m)
            
            calc_time = time.time() - start_time
            logger.debug(f"XB calculation completed in {calc_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Xie-Beni index: {str(e)}")
            return float('inf')
    
    def _calculate_xb_core(self, data, membership_matrix, centroids, m):
        """Core XB calculation with vectorization"""
        n, c = membership_matrix.shape
        
        # Vectorized numerator calculation
        numerator = 0
        for j in range(c):
            # Compute all distances to centroid j at once
            distances_sq = np.sum((data - centroids[j]) ** 2, axis=1)
            # Weighted sum using broadcasting
            weighted_distances = (membership_matrix[:, j] ** m) * distances_sq
            numerator += np.sum(weighted_distances)
        
        # Compute minimum distance between centroids
        min_centroid_dist = float('inf')
        for i in range(c):
            for j in range(i+1, c):
                dist = np.sum((centroids[i] - centroids[j]) ** 2)
                if dist < min_centroid_dist:
                    min_centroid_dist = dist
        
        # Validate denominator
        if min_centroid_dist == 0 or min_centroid_dist == float('inf'):
            logger.warning("Invalid centroid distances for XB calculation")
            return float('inf')
        
        xb = numerator / (n * min_centroid_dist)
        
        # Sanity check
        if not np.isfinite(xb) or xb < 0:
            logger.warning(f"Invalid XB value: {xb}")
            return float('inf')
            
        return xb
    
    def evaluate_single_experiment(self, experiment_folder):
        """Đánh giá một thử nghiệm cụ thể - ROBUST with comprehensive error handling"""
        experiment_name = os.path.basename(experiment_folder)
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            logger.info(f"📊 Evaluating experiment: {experiment_name}")
            
            # Validate experiment folder structure
            info_path = os.path.join(experiment_folder, 'experiment_info.json')
            results_path = os.path.join(experiment_folder, 'results.pkl')
            
            if not os.path.exists(info_path):
                raise FileNotFoundError(f"Missing experiment_info.json in {experiment_folder}")
            if not os.path.exists(results_path):
                raise FileNotFoundError(f"Missing results.pkl in {experiment_folder}")
            
            # Load experiment info with validation
            with open(info_path, 'r') as f:
                experiment_info = json.load(f)
            
            required_keys = ['experiment_id', 'parameters', 'results']
            for key in required_keys:
                if key not in experiment_info:
                    raise ValueError(f"Missing required key '{key}' in experiment info")
            
            # Load results with validation
            with open(results_path, 'rb') as f:
                results_data = pickle.load(f)
            
            required_data = ['membership_matrix', 'centroids', 'predicted_labels']
            for key in required_data:
                if key not in results_data:
                    raise ValueError(f"Missing required data '{key}' in results")
            
            membership_matrix = results_data['membership_matrix']
            centroids = results_data['centroids']
            predicted_labels = results_data['predicted_labels']
            
            # Validate data shapes
            if len(predicted_labels) != len(self.X):
                raise ValueError(f"Label count mismatch: {len(predicted_labels)} vs {len(self.X)}")
            
            # Calculate metrics with progress logging
            metrics = {}
            
            # Fast metrics first
            logger.info(f"   ⚡ Computing PC and PE...")
            metrics['partition_coefficient'] = self.calculate_partition_coefficient(membership_matrix)
            metrics['partition_entropy'] = self.calculate_partition_entropy(membership_matrix)
            
            logger.info(f"   ⚡ Computing WCSS...")
            metrics['wcss'] = self.calculate_wcss(self.X, centroids, predicted_labels)
            
            logger.info(f"   ⚡ Computing Compactness-Separation...")
            compactness, separation, cs_ratio = self.calculate_compactness_separation(
                self.X, membership_matrix, centroids, experiment_info['parameters']['m']
            )
            metrics['compactness'] = compactness
            metrics['separation'] = separation
            metrics['compactness_separation_ratio'] = cs_ratio
            
            # Adaptive silhouette calculation
            logger.info(f"   🔄 Computing Silhouette Score (adaptive)...")
            metrics['silhouette_score'] = self.calculate_silhouette_score(self.X, predicted_labels)
            
            # XB index with optimization
            logger.info(f"   ⚡ Computing Xie-Beni Index...")
            metrics['xie_beni_index'] = self.calculate_xie_beni_index(
                self.X, membership_matrix, centroids, experiment_info['parameters']['m']
            )
            
            # Validate all metrics
            for metric_name, value in metrics.items():
                if not np.isfinite(value) and value != -1:  # -1 is valid for disabled metrics
                    logger.warning(f"Invalid metric {metric_name}: {value}")
                    metrics[metric_name] = float('inf')
            
            # Create evaluation result
            evaluation = {
                'experiment_id': experiment_info['experiment_id'],
                'parameters': experiment_info['parameters'],
                'metrics': {k: float(v) for k, v in metrics.items()},
                'training_time': experiment_info['results']['training_time_seconds'],
                'evaluation_time': time.time() - start_time,
                'memory_usage': self._get_memory_usage() - start_memory
            }
            
            logger.info(f"   ✅ Metrics computed: PC={metrics['partition_coefficient']:.4f}, "
                       f"PE={metrics['partition_entropy']:.4f}, XB={metrics['xie_beni_index']:.4f}")
            
            return evaluation
            
        except Exception as e:
            error_msg = f"Error evaluating {experiment_folder}: {str(e)}"
            logger.error(error_msg)
            
            # Return error result instead of None for better tracking
            return {
                'experiment_id': experiment_name,
                'error': str(e),
                'evaluation_time': time.time() - start_time
            }
    
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
        
        # Khởi tạo file lưu kết quả
        results_file = os.path.join(self.experiment_dir, 'evaluation_results_temp.json')
        
        # Load existing results nếu có
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    self.evaluation_results = json.load(f)
                print(f"📂 Đã load {len(self.evaluation_results)} kết quả có sẵn")
            except:
                self.evaluation_results = []
        else:
            self.evaluation_results = []
        
        # Lấy danh sách experiment đã đánh giá
        evaluated_ids = {result['experiment_id'] for result in self.evaluation_results}
        
        for i, folder in enumerate(experiment_folders, 1):
            experiment_name = os.path.basename(folder)
            
            # Kiểm tra xem đã đánh giá chưa
            try:
                info_path = os.path.join(folder, 'experiment_info.json')
                with open(info_path, 'r') as f:
                    experiment_info = json.load(f)
                
                if experiment_info['experiment_id'] in evaluated_ids:
                    print(f"⏭️  Bỏ qua thử nghiệm {i}/{len(experiment_folders)}: {experiment_name} (đã đánh giá)")
                    continue
            except:
                pass
            
            print(f"⚙️  Đánh giá thử nghiệm {i}/{len(experiment_folders)}: {experiment_name}")
            evaluation = self.evaluate_single_experiment(folder)
            if evaluation:
                self.evaluation_results.append(evaluation)
                
                # Lưu kết quả ngay sau mỗi lần đánh giá
                try:
                    with open(results_file, 'w') as f:
                        json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
                    print(f"💾 Đã lưu kết quả ({len(self.evaluation_results)} thử nghiệm)")
                except Exception as e:
                    print(f"⚠️  Lỗi khi lưu: {str(e)}")
        
        print(f"✅ Hoàn thành đánh giá {len(self.evaluation_results)} thử nghiệm")
        
        # Rename file cuối cùng
        final_file = os.path.join(self.experiment_dir, 'evaluation_results_complete.json')
        if os.path.exists(results_file):
            os.rename(results_file, final_file)
            print(f"📁 Đã lưu kết quả cuối cùng tại: {final_file}")
        
        return self.evaluation_results
    
    def create_elbow_plots(self):
        """Tạo Elbow plots cho từng distance metric và m value - IMPROVED with data filtering"""
        # Load results từ file nếu chưa có
        if not self.evaluation_results:
            # Thử load từ file complete trước
            complete_file = os.path.join(self.experiment_dir, 'evaluation_results_complete.json')
            temp_file = os.path.join(self.experiment_dir, 'evaluation_results_temp.json')
            
            if os.path.exists(complete_file):
                with open(complete_file, 'r') as f:
                    self.evaluation_results = json.load(f)
            elif os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    self.evaluation_results = json.load(f)
            else:
                logger.warning("⚠️  No evaluation results found. Run evaluate_all_experiments() first.")
                return
        
        logger.info("📈 Creating Elbow plots with data quality filtering...")
        
        # Tạo DataFrame từ kết quả với data filtering
        df_results = []
        for result in self.evaluation_results:
            if 'error' not in result:
                row = {**result['parameters'], **result['metrics']}
                # Filter out experiments with invalid WCSS
                if np.isfinite(row.get('wcss', float('inf'))) and row.get('wcss', 0) > 0:
                    df_results.append(row)
        
        if not df_results:
            logger.error("No valid data for plotting!")
            return
            
        df = pd.DataFrame(df_results)
        logger.info(f"Using {len(df)} valid experiments out of {len(self.evaluation_results)} total")
        
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
                    elbow_data = subset.groupby('n_clusters')['wcss'].agg(['mean', 'std', 'count']).reset_index()
                    elbow_data = elbow_data[elbow_data['count'] > 0]  # Ensure we have data
                    
                    if len(elbow_data) > 1:  # Need at least 2 points for a line
                        # Plot with error bars if we have multiple experiments per cluster count
                        if elbow_data['std'].notna().any():
                            axes[i][j].errorbar(elbow_data['n_clusters'], elbow_data['mean'], 
                                              yerr=elbow_data['std'], fmt='bo-', linewidth=2, 
                                              markersize=6, capsize=5, capthick=2)
                        else:
                            axes[i][j].plot(elbow_data['n_clusters'], elbow_data['mean'], 
                                          'bo-', linewidth=2, markersize=6)
                        
                        axes[i][j].set_title(f'Elbow Method\n{distance_metric}, m={m}', 
                                           fontsize=12, fontweight='bold')
                        axes[i][j].set_xlabel('Number of Clusters')
                        axes[i][j].set_ylabel('WCSS')
                        axes[i][j].grid(True, alpha=0.3)
                        
                        # Format y-axis for readability
                        axes[i][j].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                        
                        # Add data point count annotation
                        axes[i][j].text(0.02, 0.98, f'n={len(subset)}', 
                                       transform=axes[i][j].transAxes, 
                                       verticalalignment='top', fontsize=10, 
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    else:
                        axes[i][j].text(0.5, 0.5, 'Insufficient\nvalid data', 
                                       transform=axes[i][j].transAxes, 
                                       ha='center', va='center', fontsize=12)
                        axes[i][j].set_title(f'Elbow Method\n{distance_metric}, m={m}', 
                                           fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.experiment_dir, 'elbow_plots_filtered.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved filtered Elbow plots at: {save_path}")
    
    def create_evaluation_summary(self):
        """Tạo bảng tổng hợp kết quả đánh giá"""
        # Load results từ file nếu chưa có
        if not self.evaluation_results:
            # Thử load từ file complete trước
            complete_file = os.path.join(self.experiment_dir, 'evaluation_results_complete.json')
            temp_file = os.path.join(self.experiment_dir, 'evaluation_results_temp.json')
            
            if os.path.exists(complete_file):
                with open(complete_file, 'r') as f:
                    self.evaluation_results = json.load(f)
                print(f"📂 Đã load {len(self.evaluation_results)} kết quả từ file complete")
            elif os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    self.evaluation_results = json.load(f)
                print(f"📂 Đã load {len(self.evaluation_results)} kết quả từ file temp")
            else:
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
                    'compactness': result['metrics'].get('compactness', 0),
                    'separation': result['metrics'].get('separation', 0),
                    'compactness_separation_ratio': result['metrics'].get('compactness_separation_ratio', float('inf')),
                    'training_time': result['training_time']
                }
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Lưu kết quả chi tiết
        csv_path = os.path.join(self.experiment_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Đã lưu kết quả chi tiết tại: {csv_path}")
        
        # Tạo summary statistics với filtering cho giá trị hợp lệ
        summary_stats = {
            'total_experiments': len(df),
            'note': 'Metrics filtered for valid values. Infinite and invalid values excluded.',
        }
        
        # Helper function to find best valid metric
        def find_best_metric(df, metric_col, ascending=True, exclude_negative=True):
            """Find best experiment for a metric, excluding invalid values"""
            valid_df = df.copy()
            
            # Filter out infinite values
            valid_df = valid_df[np.isfinite(valid_df[metric_col])]
            
            # Filter out negative values if specified
            if exclude_negative:
                valid_df = valid_df[valid_df[metric_col] >= 0]
            
            # For silhouette score, also exclude -1 (disabled indicator)
            if metric_col == 'silhouette_score':
                valid_df = valid_df[valid_df[metric_col] > -0.5]
            
            if len(valid_df) == 0:
                return None
                
            if ascending:
                best_idx = valid_df[metric_col].idxmin()
            else:
                best_idx = valid_df[metric_col].idxmax()
                
            return {
                'score': float(valid_df.loc[best_idx, metric_col]),
                'experiment_id': int(valid_df.loc[best_idx, 'experiment_id']),
                'parameters': valid_df.loc[best_idx, 
                                           ['n_clusters', 'm', 'distance_metric', 'random_state']].to_dict(),
                'valid_experiments': len(valid_df)
            }
        
        # Find best for each metric with proper filtering
        metrics_to_evaluate = [
            ('partition_coefficient', False, True, 'Higher is better - measures crispness of clustering'),
            ('partition_entropy', True, True, 'Lower is better - measures uncertainty in cluster assignments'),
            ('xie_beni_index', True, True, 'Lower is better - ratio of compactness to separation'),
            ('compactness_separation_ratio', True, True, 'Lower is better - compact clusters with good separation'),
            ('wcss', True, True, 'Lower is better - within-cluster sum of squares'),
            ('silhouette_score', False, False, 'Higher is better - quality of cluster assignments')
        ]
        
        for metric, ascending, exclude_neg, description in metrics_to_evaluate:
            if metric in df.columns:
                best_result = find_best_metric(df, metric, ascending, exclude_neg)
                if best_result:
                    key_name = f'best_{metric}' if not ascending else f'lowest_{metric}'
                    summary_stats[key_name] = best_result
                    summary_stats[key_name]['description'] = description
                else:
                    logger.warning(f"No valid values found for metric: {metric}")
        
        # Add data quality statistics
        summary_stats['data_quality'] = {
            'total_experiments': len(df),
            'experiments_with_valid_xb': len(df[np.isfinite(df['xie_beni_index']) & (df['xie_beni_index'] > 0)]),
            'experiments_with_valid_cs': len(df[np.isfinite(df['compactness_separation_ratio']) & (df['compactness_separation_ratio'] > 0)]),
            'experiments_with_silhouette': len(df[df['silhouette_score'] > -0.5]),
            'infinite_xb_count': len(df[~np.isfinite(df['xie_beni_index'])]),
            'infinite_cs_count': len(df[~np.isfinite(df['compactness_separation_ratio'])])
        }
        
        # Lưu summary
        summary_path = os.path.join(self.experiment_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Đã lưu tổng hợp đánh giá tại: {summary_path}")
        
        return df, summary_stats
    
    def create_metric_heatmaps(self):
        """Tạo heatmaps cho các metrics theo n_clusters và m"""
        # Load results từ file nếu chưa có
        if not self.evaluation_results:
            # Thử load từ file complete trước
            complete_file = os.path.join(self.experiment_dir, 'evaluation_results_complete.json')
            temp_file = os.path.join(self.experiment_dir, 'evaluation_results_temp.json')
            
            if os.path.exists(complete_file):
                with open(complete_file, 'r') as f:
                    self.evaluation_results = json.load(f)
            elif os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    self.evaluation_results = json.load(f)
            else:
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
        
        metrics = ['partition_coefficient', 'partition_entropy', 'xie_beni_index', 'compactness_separation_ratio']
        metric_labels = ['Partition Coefficient (PC)', 'Partition Entropy (PE)', 'Xie-Beni Index (XB)', 'Compactness/Separation Ratio']
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
        """Tạo plots so sánh các metrics - IMPROVED with data filtering and validation"""
        # Load results từ file nếu chưa có
        if not self.evaluation_results:
            # Thử load từ file complete trước
            complete_file = os.path.join(self.experiment_dir, 'evaluation_results_complete.json')
            temp_file = os.path.join(self.experiment_dir, 'evaluation_results_temp.json')
            
            if os.path.exists(complete_file):
                with open(complete_file, 'r') as f:
                    self.evaluation_results = json.load(f)
            elif os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    self.evaluation_results = json.load(f)
            else:
                logger.warning("⚠️  No evaluation results found.")
                return
        
        logger.info("📊 Creating comparison plots with data quality filtering...")
        
        # Tạo DataFrame với data filtering
        df_results = []
        for result in self.evaluation_results:
            if 'error' not in result:
                row = {**result['parameters'], **result['metrics']}
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Define metrics with their filtering criteria
        metrics_info = [
            ('partition_coefficient', 'Partition Coefficient vs N_Clusters', 
             lambda x: np.isfinite(x) & (x >= 0) & (x <= 1)),
            ('partition_entropy', 'Partition Entropy vs N_Clusters',
             lambda x: np.isfinite(x) & (x >= 0)),
            ('xie_beni_index', 'Xie-Beni Index vs N_Clusters',
             lambda x: np.isfinite(x) & (x > 0) & (x < 1e10)),  # Exclude extreme values
            ('compactness_separation_ratio', 'Compactness/Separation vs N_Clusters',
             lambda x: np.isfinite(x) & (x > 0) & (x < 1e6))  # Exclude extreme values
        ]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (metric, title, filter_func) in enumerate(metrics_info):
            if metric not in df.columns:
                axes[idx].text(0.5, 0.5, f'Metric {metric}\nnot available', 
                             transform=axes[idx].transAxes, ha='center', va='center')
                axes[idx].set_title(title, fontweight='bold')
                continue
            
            # Filter valid data for this metric
            valid_mask = filter_func(df[metric])
            df_valid = df[valid_mask].copy()
            
            if len(df_valid) == 0:
                axes[idx].text(0.5, 0.5, f'No valid data\nfor {metric}', 
                             transform=axes[idx].transAxes, ha='center', va='center')
                axes[idx].set_title(title, fontweight='bold')
                continue
            
            # Plot for each distance metric
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            distance_metrics = df_valid['distance_metric'].unique()
            
            for i, distance_metric in enumerate(distance_metrics):
                subset = df_valid[df_valid['distance_metric'] == distance_metric]
                if len(subset) == 0:
                    continue
                    
                # Group by n_clusters and calculate statistics
                grouped = subset.groupby('n_clusters')[metric].agg(['mean', 'std', 'count']).reset_index()
                
                # Only plot if we have reasonable data
                grouped = grouped[grouped['count'] > 0]
                
                if len(grouped) > 0:
                    color = colors[i % len(colors)]
                    
                    # Use error bars only if we have multiple experiments per cluster
                    if grouped['std'].notna().any() and (grouped['count'] > 1).any():
                        axes[idx].errorbar(grouped['n_clusters'], grouped['mean'], 
                                         yerr=grouped['std'], label=distance_metric, 
                                         marker='o', capsize=5, capthick=2, color=color,
                                         linewidth=2, markersize=6)
                    else:
                        axes[idx].plot(grouped['n_clusters'], grouped['mean'], 
                                     label=distance_metric, marker='o', 
                                     linewidth=2, markersize=6, color=color)
            
            axes[idx].set_title(title, fontweight='bold')
            axes[idx].set_xlabel('Number of Clusters')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Add data quality information
            total_experiments = len(df)
            valid_experiments = len(df_valid)
            axes[idx].text(0.02, 0.98, f'Valid: {valid_experiments}/{total_experiments}', 
                         transform=axes[idx].transAxes, verticalalignment='top', 
                         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Use log scale for metrics with wide ranges
            if metric in ['xie_beni_index', 'compactness_separation_ratio']:
                try:
                    if df_valid[metric].max() / df_valid[metric].min() > 100:
                        axes[idx].set_yscale('log')
                        axes[idx].set_ylabel(f'{metric.replace("_", " ").title()} (log scale)')
                except:
                    pass  # Continue with linear scale if log fails
        
        plt.tight_layout()
        save_path = os.path.join(self.experiment_dir, 'metric_comparison_plots_filtered.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved filtered comparison plots at: {save_path}")

    def resume_evaluation(self):
        """Tiếp tục đánh giá từ nơi đã dừng"""
        print("🔄 Tiếp tục đánh giá từ nơi đã dừng...")
        return self.evaluate_all_experiments()
    
    def estimate_evaluation_time(self):
        """Ước tính thời gian đánh giá - IMPROVED with real data analysis"""
        try:
            # Count experiments
            experiment_folders = []
            for item in os.listdir(self.experiment_dir):
                if item.startswith('experiment_') and os.path.isdir(os.path.join(self.experiment_dir, item)):
                    experiment_folders.append(item)
            
            total_experiments = len(experiment_folders)
            
            # Adaptive time estimation based on dataset size
            base_times = {
                'small': {'PC_PE': 0.1, 'WCSS': 0.5, 'XB': 2, 'CS': 0.5, 'Silhouette': 1, 'overhead': 0.5},
                'medium': {'PC_PE': 0.5, 'WCSS': 2, 'XB': 8, 'CS': 2, 'Silhouette': 10, 'overhead': 1},
                'large': {'PC_PE': 1, 'WCSS': 5, 'XB': 15, 'CS': 5, 'Silhouette': 0, 'overhead': 2}  # Silhouette disabled
            }
            
            # Determine dataset category
            if self.dataset_size <= 10000:
                category = 'small'
                silhouette_note = "Full calculation"
            elif self.dataset_size <= 100000:
                category = 'medium'
                silhouette_note = "Sampled calculation"
            else:
                category = 'large'
                silhouette_note = "Disabled (too slow)"
            
            times = base_times[category]
            
            # Scale by actual dataset size
            size_factor = max(1, self.dataset_size / 10000)
            for key in times:
                if key != 'overhead':
                    times[key] *= np.log10(size_factor)
            
            estimated_time_per_exp = sum(times.values())
            total_estimated_time = total_experiments * estimated_time_per_exp
            
            logger.info("📊 Evaluation Time Estimation:")
            logger.info(f"   Dataset size: {self.dataset_size:,} samples ({category})")
            logger.info(f"   Experiments: {total_experiments}")
            logger.info(f"   Time/experiment: ~{estimated_time_per_exp:.1f}s")
            logger.info(f"   Total estimated: ~{total_estimated_time:.0f}s ({total_estimated_time/60:.1f}min)")
            logger.info(f"   Silhouette score: {silhouette_note}")
            
            # Breakdown
            logger.info("   Metric breakdown:")
            for metric, time_val in times.items():
                if time_val > 0:
                    logger.info(f"     - {metric}: ~{time_val:.1f}s")
            
            return total_estimated_time
            
        except Exception as e:
            logger.error(f"Error estimating evaluation time: {str(e)}")
            return 0

    def get_progress(self):
        """Kiểm tra tiến độ đánh giá"""
        # Đếm tổng số experiment
        experiment_folders = []
        for item in os.listdir(self.experiment_dir):
            if item.startswith('experiment_') and os.path.isdir(os.path.join(self.experiment_dir, item)):
                experiment_folders.append(os.path.join(self.experiment_dir, item))
        
        total_experiments = len(experiment_folders)
        
        # Đếm số experiment đã đánh giá
        temp_file = os.path.join(self.experiment_dir, 'evaluation_results_temp.json')
        complete_file = os.path.join(self.experiment_dir, 'evaluation_results_complete.json')
        
        evaluated_count = 0
        if os.path.exists(complete_file):
            with open(complete_file, 'r') as f:
                evaluated_count = len(json.load(f))
        elif os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                evaluated_count = len(json.load(f))
        
        progress = (evaluated_count / total_experiments * 100) if total_experiments > 0 else 0
        
        print(f"📊 Tiến độ: {evaluated_count}/{total_experiments} ({progress:.1f}%)")
        return evaluated_count, total_experiments, progress
    
    def create_performance_report(self):
        """Tạo báo cáo hiệu suất chi tiết"""
        try:
            if not self.evaluation_results:
                logger.warning("No evaluation results available for performance report")
                return
            
            # Filter out error results
            valid_results = [r for r in self.evaluation_results if 'error' not in r]
            error_results = [r for r in self.evaluation_results if 'error' in r]
            
            logger.info("📊 Performance Report:")
            logger.info(f"   Total experiments: {len(self.evaluation_results)}")
            logger.info(f"   Successful: {len(valid_results)}")
            logger.info(f"   Failed: {len(error_results)}")
            
            if valid_results:
                # Timing analysis
                eval_times = [r.get('evaluation_time', 0) for r in valid_results]
                train_times = [r.get('training_time', 0) for r in valid_results]
                
                logger.info(f"   Avg evaluation time: {np.mean(eval_times):.2f}s")
                logger.info(f"   Max evaluation time: {np.max(eval_times):.2f}s")
                logger.info(f"   Avg training time: {np.mean(train_times):.2f}s")
                
                # Memory analysis
                memory_usage = [r.get('memory_usage', 0) for r in valid_results]
                if any(memory_usage):
                    logger.info(f"   Avg memory per evaluation: {np.mean(memory_usage):.2f}MB")
                
                # Metric quality analysis
                metrics_summary = {}
                for result in valid_results:
                    for metric, value in result['metrics'].items():
                        if metric not in metrics_summary:
                            metrics_summary[metric] = []
                        if np.isfinite(value) and value != -1:
                            metrics_summary[metric].append(value)
                
                logger.info("   Metric ranges:")
                for metric, values in metrics_summary.items():
                    if values:
                        logger.info(f"     {metric}: [{np.min(values):.4f}, {np.max(values):.4f}]")
            
            # Error analysis
            if error_results:
                logger.warning("❌ Failed experiments:")
                error_types = {}
                for result in error_results:
                    error = result['error']
                    error_type = error.split(':')[0] if ':' in error else error
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                for error_type, count in error_types.items():
                    logger.warning(f"     {error_type}: {count} occurrences")
            
        except Exception as e:
            logger.error(f"Error creating performance report: {str(e)}")
    
    def create_data_quality_report(self):
        """Tạo báo cáo chất lượng dữ liệu và đề xuất cải thiện"""
        try:
            if not self.evaluation_results:
                logger.warning("No evaluation results available for data quality report")
                return
            
            # Create DataFrame for analysis
            df_results = []
            for result in self.evaluation_results:
                if 'error' not in result:
                    row = {**result['parameters'], **result['metrics']}
                    df_results.append(row)
            
            if not df_results:
                logger.warning("No valid results for data quality analysis")
                return
                
            df = pd.DataFrame(df_results)
            
            logger.info("\n" + "="*60)
            logger.info("📊 DATA QUALITY ANALYSIS REPORT")
            logger.info("="*60)
            
            # Overall statistics
            total_experiments = len(df)
            logger.info(f"Total experiments analyzed: {total_experiments}")
            
            # Metric quality analysis
            metrics_analysis = {
                'partition_coefficient': {
                    'valid_range': (0, 1),
                    'description': 'Should be between 0 and 1'
                },
                'partition_entropy': {
                    'valid_range': (0, float('inf')),
                    'description': 'Should be positive, lower is better'
                },
                'xie_beni_index': {
                    'valid_range': (0, 1e6),
                    'description': 'Should be positive and finite, lower is better'
                },
                'compactness_separation_ratio': {
                    'valid_range': (0, 1e4),
                    'description': 'Should be positive and finite, lower is better'
                },
                'wcss': {
                    'valid_range': (0, float('inf')),
                    'description': 'Should be positive, lower is better'
                }
            }
            
            logger.info("\n📈 Metric Quality Analysis:")
            for metric, info in metrics_analysis.items():
                if metric in df.columns:
                    values = df[metric]
                    
                    # Count valid values
                    finite_count = np.isfinite(values).sum()
                    inf_count = np.isinf(values).sum()
                    nan_count = np.isnan(values).sum()
                    
                    # Count values in valid range
                    min_val, max_val = info['valid_range']
                    in_range = ((values >= min_val) & (values <= max_val) & np.isfinite(values)).sum()
                    
                    logger.info(f"  {metric}:")
                    logger.info(f"    Valid/Total: {in_range}/{total_experiments} ({in_range/total_experiments*100:.1f}%)")
                    logger.info(f"    Finite: {finite_count}, Infinite: {inf_count}, NaN: {nan_count}")
                    if finite_count > 0:
                        finite_values = values[np.isfinite(values)]
                        logger.info(f"    Range: [{finite_values.min():.4f}, {finite_values.max():.4f}]")
                    logger.info(f"    Description: {info['description']}")
            
            # Cluster analysis
            logger.info("\n🎯 Clustering Configuration Analysis:")
            cluster_counts = df['n_clusters'].value_counts().sort_index()
            logger.info(f"  Cluster counts tested: {list(cluster_counts.index)}")
            logger.info(f"  Experiments per cluster count: {dict(cluster_counts)}")
            
            # Distance metric analysis
            distance_metrics = df['distance_metric'].value_counts()
            logger.info(f"  Distance metrics: {dict(distance_metrics)}")
            
            # m value analysis
            m_values = df['m'].value_counts().sort_index()
            logger.info(f"  Fuzziness values (m): {dict(m_values)}")
            
            # Problem identification
            logger.info("\n⚠️  Potential Issues Identified:")
            
            # Check for too many infinite XB values
            if 'xie_beni_index' in df.columns:
                inf_xb_ratio = np.isinf(df['xie_beni_index']).mean()
                if inf_xb_ratio > 0.3:
                    logger.warning(f"  - High proportion of infinite Xie-Beni values ({inf_xb_ratio:.1%})")
                    logger.warning("    Possible causes: Identical centroids, numerical instability")
            
            # Check for too many infinite CS values
            if 'compactness_separation_ratio' in df.columns:
                inf_cs_ratio = np.isinf(df['compactness_separation_ratio']).mean()
                if inf_cs_ratio > 0.3:
                    logger.warning(f"  - High proportion of infinite Compactness/Separation values ({inf_cs_ratio:.1%})")
                    logger.warning("    Possible causes: Very close centroids, poor separation")
            
            # Check for silhouette score issues
            if 'silhouette_score' in df.columns:
                disabled_silhouette = (df['silhouette_score'] == -1).mean()
                if disabled_silhouette > 0.5:
                    logger.warning(f"  - Silhouette score disabled for {disabled_silhouette:.1%} of experiments")
                    logger.warning("    Reason: Large dataset optimization")
            
            # Recommendations
            logger.info("\n💡 Recommendations:")
            logger.info("  1. Filter out experiments with infinite metric values for final analysis")
            logger.info("  2. Focus on Partition Coefficient and Partition Entropy as primary metrics")
            logger.info("  3. Use WCSS for elbow method analysis")
            logger.info("  4. Consider XB and CS metrics only for experiments with finite values")
            logger.info("  5. Validate clustering results with domain knowledge")
            
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error creating data quality report: {str(e)}")

def evaluate_fcm_experiments(experiment_dir, data_path=None):
    """Function chính để đánh giá các thử nghiệm FCM - IMPROVED"""
    logger.info(f"🚀 Starting FCM experiment evaluation at: {experiment_dir}")
    
    try:
        evaluator = FCMEvaluator(experiment_dir, data_path)
        
        # Load data with monitoring
        evaluator.load_data()
        
        # Estimate time before starting
        evaluator.estimate_evaluation_time()
        
        # Evaluate all experiments
        evaluator.evaluate_all_experiments()
        
        # Create performance report
        evaluator.create_performance_report()
        
        # Create data quality report
        evaluator.create_data_quality_report()
        
        # Create visualizations and summary
        evaluator.create_elbow_plots()
        evaluator.create_metric_heatmaps()
        evaluator.create_comparison_plots()
        df, summary = evaluator.create_evaluation_summary()
        
        # Enhanced final report
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL EVALUATION RESULTS (CRITICALLY REVIEWED)")
        logger.info("="*60)
        logger.info(f"📈 Total experiments evaluated: {summary['total_experiments']}")
        
        if 'note' in summary:
            logger.info(f"⚠️  Note: {summary['note']}")
        
        # Display best results for each metric
        metrics_to_show = [
            ('best_partition_coefficient', 'Partition Coefficient', 'higher'),
            ('best_xie_beni', 'Xie-Beni Index', 'lower'),
            ('lowest_partition_entropy', 'Partition Entropy', 'lower'),
            ('best_compactness_separation', 'Compactness/Separation', 'lower'),
            ('lowest_wcss', 'WCSS', 'lower')
        ]
        
        for key, name, direction in metrics_to_show:
            if key in summary:
                score = summary[key]['score']
                exp_id = summary[key]['experiment_id']
                params = summary[key]['parameters']
                direction_text = "↑" if direction == 'higher' else "↓"
                logger.info(f"🏆 Best {name} {direction_text}: {score:.4f} (experiment {exp_id})")
                logger.info(f"   Parameters: {params}")
        
        logger.info("="*60)
        
        return evaluator, df, summary
        
    except Exception as e:
        logger.error(f"Critical error in evaluation: {str(e)}")
        raise

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
    """Main execution with improved error handling and flexibility"""
    try:
        # Auto-discover latest experiment directory
        experiment_dir = find_latest_experiment_dir()
        
        if experiment_dir:
            logger.info(f"🔍 Found experiment directory: {experiment_dir}")
            
            # Check if custom data path should be used
            data_path = None
            if len(os.sys.argv) > 1:
                data_path = os.sys.argv[1]
                logger.info(f"Using custom data path: {data_path}")
            
            evaluator, df, summary = evaluate_fcm_experiments(experiment_dir, data_path)
            
        else:
            logger.error("❌ No experiment directories found!")
            logger.info("📝 Please run fcm_experiment.py first to generate experiment data.")
            
            # Show available directories
            test_cluster_dir = "./test_cluster"
            if os.path.exists(test_cluster_dir):
                logger.info("\n📁 Available directories in test_cluster:")
                for item in os.listdir(test_cluster_dir):
                    if os.path.isdir(os.path.join(test_cluster_dir, item)):
                        logger.info(f"   - {item}")
            
    except KeyboardInterrupt:
        logger.info("⚠️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Critical error: {str(e)}")
        raise
