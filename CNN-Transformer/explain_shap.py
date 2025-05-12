import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import MultimodalModel
from dataset import FlightDataset
from normalize_utils import compute_mlp_normalization_stats, compute_traj_normalization_stats

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU 运行，防止显存不足

# ---- 加载数据 ----
track_path = "Track_used_for_train.csv"
df = pd.read_csv(track_path)

mlp_means, mlp_stds = compute_mlp_normalization_stats(df)
traj_means, traj_stds = compute_traj_normalization_stats("flight_extracts")

# MLP 特征名
feature_names = [
    'x', 'y', 'parameter', 'maneuvering_parameter', 'WTC', 'cas', 'heading', 'altitude',
    'drct', 'sknt', 'skyl1', 'flight level', 'head', 'velocity',
    'distance_to_changi', 'bearing_to_changi',
    'is_peakhour', 'num_other_plane', 'is_planroute', 'nearest_wp_dist_km'
]

# 加载 Dataset（仅结构化特征）
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

mlp_samples = []
for i in range(min(100, len(dataset))):
    mlp, traj, hist, air, y = dataset[i]
    mlp_samples.append(mlp.numpy())
mlp_samples = np.array(mlp_samples)

# ---- 加载模型（只解释结构化输入） ----
path = "try_ensemble_learning/overall_train20_checkpoint_epoch142.pth"
DEVICE = torch.device("cpu")
model = MultimodalModel()
model.load_state_dict(torch.load(path, map_location=DEVICE))
model.eval()
model.to(DEVICE)


# ---- 只用结构分支和 mock 其他部分 ----
def model_mlp_only(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        mlp_out = model.mlp(x_tensor)
        traj_out = torch.zeros((x_tensor.size(0), 128), device=DEVICE)
        cnn_hist_out = torch.zeros((x_tensor.size(0), model.cnn_hist.out_dim), device=DEVICE)
        cnn_air_out = torch.zeros((x_tensor.size(0), model.cnn_airspace.out_dim), device=DEVICE)
        fused = torch.cat([mlp_out, traj_out, cnn_hist_out, cnn_air_out], dim=1)
        output = model.fusion(fused)
        return output.cpu().numpy()


# ---- SHAP解释器（Permutation） ----
explainer = shap.Explainer(model_mlp_only, mlp_samples[:20], feature_names=feature_names)
shap_values = explainer(mlp_samples)

# ---- 多输出兼容处理 ----
offset_values = shap_values.values[:, :, 0]
duration_values = shap_values.values[:, :, 1]

offset_exp = shap.Explanation(
    values=offset_values,
    base_values=shap_values.base_values[:, 0],
    data=shap_values.data,
    feature_names=feature_names
)

duration_exp = shap.Explanation(
    values=duration_values,
    base_values=shap_values.base_values[:, 1],
    data=shap_values.data,
    feature_names=feature_names
)

# ---- 可视化并保存 ----
print("SHAP for Time Offset Prediction:")
plt.figure()
shap.plots.beeswarm(offset_exp, max_display=20, show=False)
plt.title("SHAP Beeswarm - Time Offset")
plt.savefig("shap_beeswarm_offset.png", bbox_inches='tight')
plt.close()

plt.figure()
shap.plots.bar(offset_exp, max_display=20, show=False)
plt.title("SHAP Bar - Time Offset")
plt.savefig("shap_bar_offset.png", bbox_inches='tight')
plt.close()

print("SHAP for Duration Prediction:")
plt.figure()
shap.plots.beeswarm(duration_exp, max_display=20, show=False)
plt.title("SHAP Beeswarm - Duration")
plt.savefig("shap_beeswarm_duration.png", bbox_inches='tight')
plt.close()

plt.figure()
shap.plots.bar(duration_exp, max_display=20, show=False)
plt.title("SHAP Bar - Duration")
plt.savefig("shap_bar_duration.png", bbox_inches='tight')
plt.close()
