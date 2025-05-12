# run_experiments.py（支持 CNN 类型 + 模态组合消融）
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from model_cnn_ablation import MultimodalModel # 消融实验专用
from dataset import FlightDataset
from normalize_utils import compute_mlp_normalization_stats, compute_traj_normalization_stats
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAE_offset": mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
        "MAE_duration": mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
        "RMSE_offset": np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])),
        "RMSE_duration": np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])),
        "R2_offset": r2_score(y_true[:, 0], y_pred[:, 0]),
        "R2_duration": r2_score(y_true[:, 1], y_pred[:, 1]),
    }


def train_model(model, train_loader, val_loader, device):
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    best_val_loss = float("inf")
    best_metrics = {}

    for epoch in range(100): #100
        model.train()
        for mlp, traj, hist_img, air_img, y in train_loader:
            mlp, traj, hist_img, air_img, y = mlp.to(device), traj.to(device), hist_img.to(device), air_img.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(mlp, traj, hist_img, air_img)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for mlp, traj, hist_img, air_img, y in val_loader:
                mlp, traj, hist_img, air_img, y = mlp.to(device), traj.to(device), hist_img.to(device), air_img.to(device), y.to(device)
                output = model(mlp, traj, hist_img, air_img)
                loss = criterion(output, y)
                val_loss += loss.item()
                preds.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            preds = np.vstack(preds)
            targets = np.vstack(targets)
            best_metrics = regression_metrics(targets, preds)

        scheduler.step()

    return best_metrics


def run_all_experiments():
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("Track_used_for_train.csv")
    mlp_means, mlp_stds = compute_mlp_normalization_stats(df)
    traj_means, traj_stds = compute_traj_normalization_stats("flight_extracts")

    dataset = FlightDataset(
        track_df=df,
        trajectory_dir="flight_extracts",
        plot_dir="flight_plots",
        airspace_dir="airspace_snapshot",
        mlp_feature_means=mlp_means,
        mlp_feature_stds=mlp_stds,
        traj_feature_means=traj_means,
        traj_feature_stds=traj_stds,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    cnn_types = ["efficientnet", "resnet", "custom"]
    configs = [
        {"use_mlp": True, "use_trans": True, "use_hist": True, "use_air": True},
        {"use_mlp": True, "use_trans": True, "use_hist": False, "use_air": True},
        {"use_mlp": True, "use_trans": True, "use_hist": True, "use_air": False},
        {"use_mlp": True, "use_trans": False, "use_hist": True, "use_air": True},
        {"use_mlp": False, "use_trans": True, "use_hist": True, "use_air": True},
        {"use_mlp": True, "use_trans": True, "use_hist": False, "use_air": False},
    ]

    results = []
    for cnn_type in cnn_types:
        for config in configs:
            print(f"\nRunning CNN={cnn_type} | config={config}")
            model = MultimodalModel(
                cnn_type=cnn_type,
                use_mlp=config["use_mlp"],
                use_transformer=config["use_trans"],
                use_hist_img=config["use_hist"],
                use_airspace_img=config["use_air"]
            ).to(device)
            metrics = train_model(model, train_loader, val_loader, device)
            entry = {"cnn_type": cnn_type, **config, **metrics}
            results.append(entry)

    df_result = pd.DataFrame(results)
    df_result.to_csv("full_ablation_results.csv", index=False)
    print("\nSaved full_ablation_results.csv")


if __name__ == "__main__":
    run_all_experiments()
