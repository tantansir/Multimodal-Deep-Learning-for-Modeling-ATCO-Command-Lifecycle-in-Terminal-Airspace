import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import MultimodalModel
from dataset import FlightDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from normalize_utils import compute_mlp_normalization_stats, compute_traj_normalization_stats
import random

# ============ 配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
SEED = 3407

# ============ 设置随机种子 ============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# 12个checkpoint路径
CHECKPOINT_PATHS = [
    "try_ensemble_learning/duration_train19_checkpoint_epoch39.pth",
    "try_ensemble_learning/offset_train19_checkpoint_epoch124.pth",
    "try_ensemble_learning/overall_train19_checkpoint_epoch81.pth",
    "try_ensemble_learning/duration_train19_checkpoint_epoch40.pth",
    "try_ensemble_learning/offset_train19_checkpoint_epoch143.pth",
    "try_ensemble_learning/overall_train19_checkpoint_epoch98.pth",
    "try_ensemble_learning/duration_train20_checkpoint_epoch98.pth",
    "try_ensemble_learning/offset_train20_checkpoint_epoch124.pth",
    "try_ensemble_learning/overall_train20_checkpoint_epoch116.pth",
    "try_ensemble_learning/duration_train20_checkpoint_epoch122.pth",
    "try_ensemble_learning/offset_train20_checkpoint_epoch141.pth",
    "try_ensemble_learning/overall_train20_checkpoint_epoch142.pth"
]

# ============ 加载验证集 ============
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

# 固定验证集划分
val_ratio = 0.2
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size

train_set, val_set = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ============ 获取 Ground Truth ============
targets = []
with torch.no_grad():
    for _, _, _, _, y in val_loader:
        targets.append(y.numpy())
targets = np.vstack(targets)

# ============ 模型推理 ============
all_offset_preds = []
all_duration_preds = []

for path in CHECKPOINT_PATHS:
    print(f"\nLoading checkpoint: {path}")
    model = MultimodalModel(mlp_dim=20).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for mlp, traj, hist_img, air_img, _ in val_loader:
            mlp, traj, hist_img, air_img = mlp.to(DEVICE), traj.to(DEVICE), hist_img.to(DEVICE), air_img.to(DEVICE)
            output = model(mlp, traj, hist_img, air_img)
            preds.append(output.cpu().numpy())

    preds_all = np.vstack(preds)
    all_offset_preds.append(preds_all[:, 0])
    all_duration_preds.append(preds_all[:, 1])

# ============ 融合所有模型预测 ============
all_offset_preds = np.stack(all_offset_preds, axis=0)
all_duration_preds = np.stack(all_duration_preds, axis=0)

# 加权策略：offset偏重 offset-checkpoints, duration偏重 duration-checkpoints
# 这里假设 offset 模型是索引 [1, 4, 7, 10]，duration 模型是 [0, 3, 6, 9]，其余为 overall
# offset_weights = np.array([0, 0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0])
# duration_weights = np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0])

offset_weights = np.array([0, 0.5, 0.1, 0, 0.5, 0.1, 0, 0.3, 0.1, 0, 0.3, 0.1])
duration_weights = np.array([0.3, 0, 0.1, 0.3, 0, 0.1, 0.5, 0, 0.1, 0.5, 0, 0.1])

offset_weights = offset_weights / offset_weights.sum()
duration_weights = duration_weights / duration_weights.sum()

ensemble_offset = np.average(all_offset_preds, axis=0, weights=offset_weights)
ensemble_duration = np.average(all_duration_preds, axis=0, weights=duration_weights)
ensemble_all = np.stack([ensemble_offset, ensemble_duration], axis=1)

# ============ 评估函数 ============
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


metrics = regression_metrics(targets, ensemble_all)
print("\n✅ 集成模型评估指标（加权融合）:")
with open("metric_info.txt", "w") as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value:.4f}\n")
        print(f"{key}: {value:.4f}")

# ============ 保存预测结果 ============
pd.DataFrame(ensemble_all, columns=["offset_pred", "duration_pred"]).to_csv("ensemble_predictions.csv", index=False)
print("\n✅ 已保存 ensemble_predictions.csv")

# ============ 绘制预测散点图 ============
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(targets[:, 0], ensemble_offset, alpha=0.5)
plt.plot([0, 60], [0, 60], 'r--')
plt.xlabel("True Offset")
plt.ylabel("Predicted Offset")
plt.title("Offset Prediction")

plt.subplot(1, 2, 2)
plt.scatter(targets[:, 1], ensemble_duration, alpha=0.5)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel("True Duration")
plt.ylabel("Predicted Duration")
plt.title("Duration Prediction")
plt.tight_layout()
plt.savefig("ensemble_scatter.png")
print("✅ 已保存散点图 ensemble_scatter.png")

# ============ 集成 vs 单模型 R² 柱状图 ============
# 遍历所有模型，记录各自的 R²、R²_offset、R²_duration
r2_records = []

for path in CHECKPOINT_PATHS:
    model = MultimodalModel(mlp_dim=20).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for mlp, traj, hist_img, air_img, _ in val_loader:
            mlp, traj, hist_img, air_img = mlp.to(DEVICE), traj.to(DEVICE), hist_img.to(DEVICE), air_img.to(DEVICE)
            output = model(mlp, traj, hist_img, air_img)
            preds.append(output.cpu().numpy())

    preds_all = np.vstack(preds)
    r2 = r2_score(targets, preds_all)
    r2_offset = r2_score(targets[:, 0], preds_all[:, 0])
    r2_duration = r2_score(targets[:, 1], preds_all[:, 1])

    r2_records.append({
        "model": path,
        "R2": r2,
        "R2_offset": r2_offset,
        "R2_duration": r2_duration
    })

# 集成模型评估指标（重新定义）
ensemble_metrics = {
    "R2": r2_score(targets, ensemble_all),
    "R2_offset": r2_score(targets[:, 0], ensemble_all[:, 0]),
    "R2_duration": r2_score(targets[:, 1], ensemble_all[:, 1])
}

# 转为 DataFrame，排序并提取每个最高 R² 对应模型
r2_df = pd.DataFrame(r2_records)
best_r2_model = r2_df.sort_values("R2", ascending=False).iloc[0]
best_offset_model = r2_df.sort_values("R2_offset", ascending=False).iloc[0]
best_duration_model = r2_df.sort_values("R2_duration", ascending=False).iloc[0]

# 准备绘图数据
groups = ["Overall R2", "Offset R2", "Duration R2"]
single_values = [
    best_r2_model["R2"],
    best_offset_model["R2_offset"],
    best_duration_model["R2_duration"]
]
ensemble_values = [
    ensemble_metrics["R2"],
    ensemble_metrics["R2_offset"],
    ensemble_metrics["R2_duration"]
]

x = np.arange(len(groups))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, single_values, width, label='Best Single')
plt.bar(x + width/2, ensemble_values, width, label='Ensemble')
plt.xticks(x, groups)
plt.ylabel("R²")
plt.title("R² Comparison: Best Single Models vs Ensemble")
plt.legend()
plt.tight_layout()
plt.savefig("r2_comparison_bar.png")
print("✅ 已保存柱状图 r2_comparison_bar.png")