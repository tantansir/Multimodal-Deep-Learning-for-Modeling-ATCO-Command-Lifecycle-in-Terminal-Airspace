import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import MultimodalModel
from dataset import FlightDataset
from normalize_utils import compute_mlp_normalization_stats, compute_traj_normalization_stats
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

# ---- 数据准备 ----
df = pd.read_csv("Track_used_for_train.csv")
mlp_means, mlp_stds = compute_mlp_normalization_stats(df)
traj_means, traj_stds = compute_traj_normalization_stats("flight_extracts")

# Dataset
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

path = "try_ensemble_learning/overall_train20_checkpoint_epoch142.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel()
model.load_state_dict(torch.load(path, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# GradCAM 初始化
hist_cam = GradCAM(model=model.cnn_hist, target_layers=[model.cnn_hist.cnn[0]])
air_cam = GradCAM(model=model.cnn_airspace, target_layers=[model.cnn_airspace.cnn[0]])

# 输出目录
os.makedirs("gradcam_output", exist_ok=True)

# 批量生成图像
for i in range(len(dataset)):
    mlp, traj, hist_img, air_img, y = dataset[i]
    hist_img = hist_img.unsqueeze(0).to(DEVICE)
    air_img = air_img.unsqueeze(0).to(DEVICE)

    # GradCAM for hist
    hist_heatmap = hist_cam(input_tensor=hist_img, targets=[ClassifierOutputTarget(0)])
    original_hist = hist_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_hist = (original_hist - original_hist.min()) / (original_hist.max() - original_hist.min())
    plt.imshow(show_cam_on_image(original_hist, hist_heatmap[0], use_rgb=True))
    plt.title("GradCAM - Historical Trajectory")
    plt.axis('off')
    plt.savefig(f"gradcam_output/gradcam_hist_{i}.png")
    plt.close()

    # GradCAM for airspace
    air_heatmap = air_cam(input_tensor=air_img, targets=[ClassifierOutputTarget(1)])
    original_air = air_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_air = (original_air - original_air.min()) / (original_air.max() - original_air.min())
    plt.imshow(show_cam_on_image(original_air, air_heatmap[0], use_rgb=True))
    plt.title("GradCAM - Airspace Snapshot")
    plt.axis('off')
    plt.savefig(f"gradcam_output/gradcam_airspace_{i}.png")
    plt.close()

    print(f"Saved GradCAM images for sample {i}")

# 清理
del hist_cam
del air_cam