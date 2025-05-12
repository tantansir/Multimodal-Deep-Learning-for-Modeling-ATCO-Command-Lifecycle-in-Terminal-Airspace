import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FlightDataset(Dataset):
    def __init__(self, track_df, trajectory_dir, plot_dir, airspace_dir,
                 mlp_feature_means=None, mlp_feature_stds=None,
                 traj_feature_means=None, traj_feature_stds=None,
                 transform=None):
        self.df = track_df.reset_index(drop=True)
        self.trajectory_dir = trajectory_dir
        self.plot_dir = plot_dir
        self.airspace_dir = airspace_dir

        # 图像增强器
        self.transform = transform or A.Compose([
            A.Resize(224, 224),
            #A.HorizontalFlip(p=0.5),
            #A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.1),
            A.Normalize(),
            ToTensorV2()
        ])

        # 标准化参数
        self.mlp_feature_means = torch.tensor(mlp_feature_means or [0]*20, dtype=torch.float32)
        self.mlp_feature_stds = torch.tensor(mlp_feature_stds or [1]*20, dtype=torch.float32)
        self.traj_feature_means = torch.tensor(traj_feature_means or [0]*5, dtype=torch.float32)
        self.traj_feature_stds = torch.tensor(traj_feature_stds or [1]*5, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ------- MLP 输入 -------
        mlp_raw = torch.tensor([
            row['x'], row['y'], row['parameter'], row['maneuvering_parameter'],
            row['WTC'], row['cas'], row['heading'], row['altitude'], row['drct'], row['sknt'], row['skyl1'],
            row['flight level'], row['head'], row['velocity'],
            row['distance_to_changi'], row['bearing_to_changi'],
            float(row['is_peakhour']), row['num_other_plane'],
            float(row['is_planroute']), row['nearest_wp_dist_km'],
        ], dtype=torch.float32)

        # NaN替换
        mlp_raw_nan_mask = torch.isnan(mlp_raw)
        mlp_raw[mlp_raw_nan_mask] = self.mlp_feature_means[mlp_raw_nan_mask]
        mlp_features = (mlp_raw - self.mlp_feature_means) / self.mlp_feature_stds

        # 加噪声增强
        mlp_features += torch.normal(mean=0.0, std=0.01, size=mlp_features.shape)

        # ------- Transformer 输入 -------
        callsign = row['callsign']
        time = str(row['time'])
        parameter = row['parameter']
        traj_file = f"{time}_{callsign}_{parameter}.csv"
        traj_path = os.path.join(self.trajectory_dir, traj_file)

        if os.path.exists(traj_path):
            traj_df = pd.read_csv(traj_path)
            traj_array = traj_df[['CAS', 'derived_heading', 'altitude', 'latitude', 'longitude']].values[-60:]
            traj_array = np.nan_to_num(traj_array, nan=0.0)
            traj_seq = torch.tensor(traj_array, dtype=torch.float32)
            traj_seq = (traj_seq - self.traj_feature_means) / self.traj_feature_stds
            traj_seq += torch.normal(mean=0.0, std=0.01, size=traj_seq.shape)
        else:
            traj_seq = torch.zeros((60, 5))

        # ------- 图像输入 -------
        img_file = f"{time}_{callsign}_{parameter}.jpg"
        plot_img_path = os.path.join(self.plot_dir, img_file)
        airspace_img_path = os.path.join(self.airspace_dir, img_file)

        hist_img = self.load_image(plot_img_path)
        airspace_img = self.load_image(airspace_img_path)

        # ------- 输出目标 -------
        target = torch.tensor([row['time_offset'], row['duration']], dtype=torch.float32)

        return mlp_features, traj_seq, hist_img, airspace_img, target

    def load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img = np.array(img)
            return self.transform(image=img)["image"]
        except:
            return torch.zeros((3, 224, 224), dtype=torch.float32)
