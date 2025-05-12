import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

from model import MultimodalModel
from dataset import FlightDataset
from normalize_utils import compute_mlp_normalization_stats, compute_traj_normalization_stats
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==== 回归指标 ====
def regression_metrics(y_true, y_pred):
    metrics = {
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
    return metrics


# ==== 参数配置 ====
set_seed(3407)
BATCH_SIZE = 16
EPOCHS = 200 # 200
LR = 1e-5 # 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

track_path = "Track_used_for_train.csv"
traj_dir = "flight_extracts"
plot_dir = "flight_plots"
airspace_dir = "airspace_snapshot"

df = pd.read_csv(track_path)
mlp_means, mlp_stds = compute_mlp_normalization_stats(df)
traj_means, traj_stds = compute_traj_normalization_stats(traj_dir)

dataset = FlightDataset(
    track_df=df,
    trajectory_dir=traj_dir,
    plot_dir=plot_dir,
    airspace_dir=airspace_dir,
    mlp_feature_means=mlp_means,
    mlp_feature_stds=mlp_stds,
    traj_feature_means=traj_means,
    traj_feature_stds=traj_stds,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

model = MultimodalModel(mlp_dim=20).to(DEVICE)

# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
all_val_preds, all_val_targets = [], []
best_loss = float("inf")

# 初始化每轮的回归指标记录
metric_history = {
    "MAE": [], "RMSE": [], "R2": [],
    "MAE_offset": [], "MAE_duration": [],
    "RMSE_offset": [], "RMSE_duration": [],
    "R2_offset": [], "R2_duration": []
}

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_epoch = -1

# ==== 训练 ====
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for mlp, traj, hist_img, air_img, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        mlp, traj, hist_img, air_img, y = mlp.to(DEVICE), traj.to(DEVICE), hist_img.to(DEVICE), air_img.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        output = model(mlp, traj, hist_img, air_img)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for mlp, traj, hist_img, air_img, y in val_loader:
            mlp, traj, hist_img, air_img, y = mlp.to(DEVICE), traj.to(DEVICE), hist_img.to(DEVICE), air_img.to(DEVICE), y.to(DEVICE)
            output = model(mlp, traj, hist_img, air_img)
            loss = criterion(output, y)

            val_loss += loss.item()
            preds.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    # 记录当前epoch的metrics
    epoch_metrics = regression_metrics(targets, preds)
    for key in metric_history:
        metric_history[key].append(epoch_metrics[key])

    print(f"[Epoch {epoch + 1} Metrics] {epoch_metrics}")

    #if (epoch+1) % 10 == 0:
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model.pth")
        all_val_preds = preds
        all_val_targets = targets

    scheduler.step()

# ==== 可视化 Loss ====
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Training/Validation Loss")
plt.savefig("loss_curve.png")

# 打印最优模型信息
print(f"\n最优模型出现在第 {best_epoch} 轮，验证集 Loss = {best_loss:.4f}")
metrics = regression_metrics(all_val_targets, all_val_preds)
print("回归结果指标（最优验证模型）:", metrics)

with open("best_model_info.txt", "w") as f:
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Val Loss: {best_loss:.4f}\n")
    for key, value in metrics.items():
        f.write(f"{key}: {value:.4f}\n")


# ==== 可视化预测对比 ====
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(all_val_targets[:, 0], all_val_preds[:, 0], alpha=0.5)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel("True Offset")
plt.ylabel("Predicted Offset")
plt.title("Offset Prediction")

plt.subplot(1, 2, 2)
plt.scatter(all_val_targets[:, 1], all_val_preds[:, 1], alpha=0.5)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel("True Duration")
plt.ylabel("Predicted Duration")
plt.title("Duration Prediction")
plt.tight_layout()
plt.savefig("best_model_regression_scatter.png")


# ==== 绘制柱状图展示 MAE / RMSE / R² ====
bar_metrics = ["MAE_offset", "MAE_duration", "RMSE_offset", "RMSE_duration", "R2_offset", "R2_duration"]
values = [metrics[m] for m in bar_metrics]

plt.figure(figsize=(10, 5))
plt.bar(bar_metrics, values, color='skyblue')
plt.xticks(rotation=45)
plt.title("Evaluation Metrics per Output")
plt.tight_layout()
plt.savefig("regression_metrics_bar.png")


# ==== 可视化 Metrics 曲线 ====
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(metric_history["MAE"], label="MAE")
plt.plot(metric_history["RMSE"], label="RMSE")
plt.legend()
plt.title("MAE & RMSE over Epochs")

plt.subplot(3, 1, 2)
plt.plot(metric_history["MAE_offset"], label="MAE_offset")
plt.plot(metric_history["MAE_duration"], label="MAE_duration")
plt.legend()
plt.title("MAE per Output over Epochs")

plt.subplot(3, 1, 3)
plt.plot(metric_history["R2_offset"], label="R2_offset")
plt.plot(metric_history["R2_duration"], label="R2_duration")
plt.legend()
plt.title("R² per Output over Epochs")

plt.tight_layout()
plt.savefig("metrics_curve.png")

# ==== 保存每个epoch的指标到CSV ====
metrics_df = pd.DataFrame(metric_history)
metrics_df["Epoch"] = range(1, EPOCHS + 1)
metrics_df.to_csv("epoch_metrics.csv", index=False)
print("每轮指标已保存为 epoch_metrics.csv")