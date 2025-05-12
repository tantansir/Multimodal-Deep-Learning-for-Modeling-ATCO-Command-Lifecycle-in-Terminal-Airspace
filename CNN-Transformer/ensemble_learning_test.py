import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import MultimodalModel
from dataset import FlightDataset
from torch.utils.data import DataLoader
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

# ============ 12个checkpoint路径 ============
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

# ============ 加载完整数据集 ============
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

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        for mlp, traj, hist_img, air_img, _ in data_loader:
            mlp, traj, hist_img, air_img = mlp.to(DEVICE), traj.to(DEVICE), hist_img.to(DEVICE), air_img.to(DEVICE)
            output = model(mlp, traj, hist_img, air_img)
            preds.append(output.cpu().numpy())

    preds_all = np.vstack(preds)
    all_offset_preds.append(preds_all[:, 0])
    all_duration_preds.append(preds_all[:, 1])

# ============ 加权融合 ============
all_offset_preds = np.stack(all_offset_preds, axis=0)
all_duration_preds = np.stack(all_duration_preds, axis=0)

offset_weights = np.array([0, 0.5, 0.1, 0, 0.5, 0.1, 0, 0.3, 0.1, 0, 0.3, 0.1])
duration_weights = np.array([0.3, 0, 0.1, 0.3, 0, 0.1, 0.5, 0, 0.1, 0.5, 0, 0.1])

offset_weights = offset_weights / offset_weights.sum()
duration_weights = duration_weights / duration_weights.sum()

ensemble_offset = np.average(all_offset_preds, axis=0, weights=offset_weights)
ensemble_duration = np.average(all_duration_preds, axis=0, weights=duration_weights)

# ============ 输出预测结果 ============
output_df = df.copy()
output_df["predicted_offset"] = ensemble_offset
output_df["predicted_duration"] = ensemble_duration
output_df.to_csv("ensemble_predictions.csv", index=False)

print("\n✅ 推理完成，预测结果已保存到 ensemble_predictions.csv")
