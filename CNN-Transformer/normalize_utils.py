import os
import numpy as np
import pandas as pd


def compute_mlp_normalization_stats(track_df):
    """
    计算用于 MLP 输入的特征的均值和标准差
    """
    mlp_columns = [
        'x', 'y', 'parameter', 'maneuvering_parameter',
        'WTC', 'cas', 'heading', 'altitude', 'drct', 'sknt', 'skyl1',
        'flight level', 'head', 'velocity',
        'distance_to_changi', 'bearing_to_changi',
        'is_peakhour', 'num_other_plane', 'is_planroute', 'nearest_wp_dist_km'
    ]

    mlp_data = track_df[mlp_columns].copy()
    mlp_data = mlp_data.fillna(0.0)

    means = mlp_data.mean().values
    stds = mlp_data.std().replace(0, 1e-6).values  # 防止除以0

    return means.tolist(), stds.tolist()
# def compute_mlp_normalization_stats(track_df):
#     """
#     计算用于 MLP 数值输入的特征的均值和标准差（不包括 flight_mode）
#     """
#     mlp_columns = [
#         'x', 'y', 'parameter', 'maneuvering_parameter',
#         'WTC', 'cas', 'heading', 'altitude', 'drct', 'sknt', 'skyl1',
#         'distance_to_changi', 'bearing_to_changi',
#         'is_peakhour', 'num_other_plane', 'is_planroute', 'nearest_wp_dist_km',
#     ]
#
#     mlp_data = track_df[mlp_columns].fillna(0.0)
#
#     means = mlp_data.mean().values
#     stds = mlp_data.std().replace(0, 1e-6).values
#
#     return means.tolist(), stds.tolist()



def compute_traj_normalization_stats(trajectory_dir):
    """
    计算 Transformer 输入的轨迹特征的均值和标准差
    """
    traj_features = []

    for filename in os.listdir(trajectory_dir):
        if filename.endswith(".csv"):
            path = os.path.join(trajectory_dir, filename)
            try:
                df = pd.read_csv(path)
                data = df[['CAS', 'derived_heading', 'altitude', 'latitude', 'longitude']] \
                         .fillna(0.0).values
                traj_features.append(data)
            except Exception as e:
                print(f"[跳过] 文件 {filename} 读取失败：{e}")

    if not traj_features:
        raise ValueError("未能加载任何轨迹特征数据")

    all_data = np.concatenate(traj_features, axis=0)
    means = np.mean(all_data, axis=0)
    stds = np.std(all_data, axis=0)
    stds[stds == 0] = 1e-6  # 防止除以0

    return means.tolist(), stds.tolist()
