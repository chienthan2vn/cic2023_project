import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Tạo dataset CIC 2023 mini
np.random.seed(42)

def create_cic2023_mini():
    """Tạo dataset CIC 2023 mini mô phỏng"""
    n_samples = 1000
    attack_types = ['BENIGN', 'DDoS', 'PortScan', 'BruteForce', 'WebAttack']
    
    data = []
    labels = []
    
    for i, attack in enumerate(attack_types):
        n_attack_samples = n_samples // len(attack_types)
        
        if attack == 'BENIGN':
            flow_duration = np.random.normal(5000, 1000, n_attack_samples)
            total_fwd_packets = np.random.normal(50, 15, n_attack_samples)
            total_bwd_packets = np.random.normal(45, 12, n_attack_samples)
            flow_bytes_s = np.random.normal(1000, 300, n_attack_samples)
            flow_packets_s = np.random.normal(10, 3, n_attack_samples)
        elif attack == 'DDoS':
            flow_duration = np.random.normal(1000, 200, n_attack_samples)
            total_fwd_packets = np.random.normal(200, 50, n_attack_samples)
            total_bwd_packets = np.random.normal(5, 2, n_attack_samples)
            flow_bytes_s = np.random.normal(5000, 1000, n_attack_samples)
            flow_packets_s = np.random.normal(100, 20, n_attack_samples)
        elif attack == 'PortScan':
            flow_duration = np.random.normal(100, 50, n_attack_samples)
            total_fwd_packets = np.random.normal(5, 2, n_attack_samples)
            total_bwd_packets = np.random.normal(2, 1, n_attack_samples)
            flow_bytes_s = np.random.normal(100, 50, n_attack_samples)
            flow_packets_s = np.random.normal(2, 1, n_attack_samples)
        elif attack == 'BruteForce':
            flow_duration = np.random.normal(2000, 500, n_attack_samples)
            total_fwd_packets = np.random.normal(20, 5, n_attack_samples)
            total_bwd_packets = np.random.normal(15, 5, n_attack_samples)
            flow_bytes_s = np.random.normal(500, 100, n_attack_samples)
            flow_packets_s = np.random.normal(5, 2, n_attack_samples)
        else:  # WebAttack
            flow_duration = np.random.normal(3000, 800, n_attack_samples)
            total_fwd_packets = np.random.normal(80, 20, n_attack_samples)
            total_bwd_packets = np.random.normal(70, 15, n_attack_samples)
            flow_bytes_s = np.random.normal(2000, 500, n_attack_samples)
            flow_packets_s = np.random.normal(25, 8, n_attack_samples)
        
        # Tạo thêm features
        fwd_packet_length_max = total_fwd_packets * np.random.uniform(10, 100, n_attack_samples)
        bwd_packet_length_max = total_bwd_packets * np.random.uniform(10, 100, n_attack_samples)
        flow_iat_mean = flow_duration / (total_fwd_packets + total_bwd_packets + 1)
        fwd_iat_mean = flow_duration / (total_fwd_packets + 1)
        bwd_iat_mean = flow_duration / (total_bwd_packets + 1)
        
        for j in range(n_attack_samples):
            sample = [
                flow_duration[j], total_fwd_packets[j], total_bwd_packets[j],
                flow_bytes_s[j], flow_packets_s[j], fwd_packet_length_max[j],
                bwd_packet_length_max[j], flow_iat_mean[j], fwd_iat_mean[j], bwd_iat_mean[j]
            ]
            data.append(sample)
            labels.append(attack)
    
    columns = [
        'Flow_Duration', 'Total_Fwd_Packets', 'Total_Bwd_Packets',
        'Flow_Bytes_s', 'Flow_Packets_s', 'Fwd_Packet_Length_Max',
        'Bwd_Packet_Length_Max', 'Flow_IAT_Mean', 'Fwd_IAT_Mean', 'Bwd_IAT_Mean'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df['Label'] = labels
    return df