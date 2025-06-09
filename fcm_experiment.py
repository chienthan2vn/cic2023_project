import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from fcm import FCM
from preprocess import _preprocess_cic2023_mini
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    X, y_encoded = _preprocess_cic2023_mini(file_path)
    
    # Load lại để lấy original labels
    df = pd.read_csv(file_path)
    original_labels = df['label'].copy()
    
    print(f"Dữ liệu sau khi xử lý: {X.shape}")
    print(f"Số lượng label gốc: {len(original_labels.unique())}")
    
    return X, original_labels, y_encoded

def run_fcm_experiment():
    """Chạy thử nghiệm FCM với các tham số khác nhau"""
    
    # Load dữ liệu
    data_path = "/data/0.01percent_34classes.csv"
    X, original_labels, y_encoded = load_and_preprocess_data(data_path)
    
    # Tham số thử nghiệm
    m_values = [2.0, 2.5, 3.0]
    tol_values = [0.01, 0.001]
    n_clusters_values = [3, 5, 8, 10, 12, 15]
    random_state_values = [42, 64, 88]
    distance_metrics = ["euclidean", "canberra"]  # Sửa tên distance metric
    
    # Tạo thư mục lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"/test_cluster/fcm_experiment_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    experiment_id = 0
    all_results = []
    
    total_experiments = len(m_values) * len(tol_values) * len(n_clusters_values) * len(random_state_values) * len(distance_metrics)
    print(f"Tổng số thử nghiệm: {total_experiments}")
    
    for m in m_values:
        for tol in tol_values:
            for n_clusters in n_clusters_values:
                for random_state in random_state_values:
                    for distance_metric in distance_metrics:
                        
                        experiment_id += 1
                        print(f"\nThử nghiệm {experiment_id}/{total_experiments}")
                        print(f"m={m}, tol={tol}, n_clusters={n_clusters}, random_state={random_state}, distance_metric={distance_metric}")
                        
                        try:
                            # Tạo và huấn luyện mô hình FCM
                            fcm = FCM(
                                n_clusters=n_clusters,
                                m=m,
                                max_iter=100,
                                tol=tol,
                                random_state=random_state,
                                distance_metric=distance_metric
                            )
                            
                            # Fit model
                            start_time = datetime.now()
                            fcm.fit(X)
                            end_time = datetime.now()
                            training_time = (end_time - start_time).total_seconds()
                            
                            # Dự đoán clusters
                            predicted_labels = fcm.predict(X)
                            
                            # Lưu thông tin thử nghiệm
                            experiment_info = {
                                'experiment_id': experiment_id,
                                'parameters': {
                                    'm': m,
                                    'tol': tol,
                                    'n_clusters': n_clusters,
                                    'random_state': random_state,
                                    'distance_metric': distance_metric,
                                    'max_iter': 100
                                },
                                'results': {
                                    'training_time_seconds': training_time,
                                    'convergence_achieved': True,
                                    'data_shape': X.shape,
                                    'unique_predicted_clusters': len(np.unique(predicted_labels))
                                },
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Lưu kết quả vào file
                            experiment_folder = os.path.join(result_dir, f"experiment_{experiment_id:03d}")
                            os.makedirs(experiment_folder, exist_ok=True)
                            
                            # Lưu ma trận membership (u) và centroids vào file pkl
                            results_data = {
                                'membership_matrix': fcm.u,
                                'centroids': fcm.centroids,
                                'predicted_labels': predicted_labels,
                                'original_labels': original_labels,
                                'encoded_labels': y_encoded
                            }
                            
                            with open(os.path.join(experiment_folder, 'results.pkl'), 'wb') as f:
                                pickle.dump(results_data, f)
                            
                            # Lưu thông tin thử nghiệm vào file json
                            with open(os.path.join(experiment_folder, 'experiment_info.json'), 'w') as f:
                                json.dump(experiment_info, f, indent=2, ensure_ascii=False)
                            
                            all_results.append(experiment_info)
                            print(f"✓ Thành công - Thời gian huấn luyện: {training_time:.2f}s")
                            
                        except Exception as e:
                            print(f"✗ Lỗi trong thử nghiệm {experiment_id}: {str(e)}")
                            error_info = {
                                'experiment_id': experiment_id,
                                'parameters': {
                                    'm': m,
                                    'tol': tol,
                                    'n_clusters': n_clusters,
                                    'random_state': random_state,
                                    'distance_metric': distance_metric
                                },
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            }
                            all_results.append(error_info)
    
    # Lưu tổng hợp tất cả kết quả
    with open(os.path.join(result_dir, 'all_experiments_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Hoàn thành tất cả thử nghiệm!")
    print(f"📁 Kết quả được lưu tại: {result_dir}")
    print(f"📊 Tổng số thử nghiệm: {len(all_results)}")
    
    return result_dir, all_results

if __name__ == "__main__":
    result_dir, results = run_fcm_experiment()
