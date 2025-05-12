import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import MultimodalModel
from dataset import FlightDataset
from normalize_utils import compute_mlp_normalization_stats, compute_traj_normalization_stats
import os

# ======= 准备数据 =======
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

# ======= 加载模型 =======
path = "try_ensemble_learning/overall_train20_checkpoint_epoch142.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel()
model.load_state_dict(torch.load(path, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ======= 可视化函数 =======
def plot_attention_map(attn_matrix, layer_idx, sample_idx=0):
    """
    attn_matrix: shape [B, T, T] from layer.attn_weights
    """
    attn = attn_matrix[sample_idx].numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn, cmap='viridis')
    plt.title(f"Attention Map (Layer {layer_idx}, Sample {sample_idx})")
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.tight_layout()
    os.makedirs("attn_maps", exist_ok=True)
    plt.savefig(f"attn_maps/attn_layer{layer_idx}_sample{sample_idx}.png")
    plt.close()

# ======= 运行样本并保存注意力图 =======
sample_index = 0  # 可改成其他样本
_, traj_seq, _, _, _ = dataset[sample_index]
traj_seq = traj_seq.unsqueeze(0).to(DEVICE)  # [1, 60, 5]

with torch.no_grad():
    _ = model.trans(traj_seq)

for i, attn in enumerate(model.trans.attn_maps):
    plot_attention_map(attn, i, sample_idx=sample_index)

print("✅ Transformer 注意力图已保存到 attn_maps/")
