import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn.functional as F


# ----------------- MLP 分支（结构化信息） -----------------
class MLPBranch(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- 自定义 Transformer Layer（用于注意力可视化） -----------------
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropouts
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn_weights = None

    def forward(self, src):
        # ---- Multi-head Self-Attention Block ----
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        self.attn_weights = attn_weights.detach().cpu()  # Optional for visualization
        src = src + self.dropout2(attn_output)
        src = self.norm1(src)

        # ---- Feedforward Network Block ----
        ff_output = self.linear2(self.dropout1(F.relu(self.linear1(src))))
        src = src + self.dropout3(ff_output)
        src = self.norm2(src)

        return src


# ----------------- Transformer 分支（历史轨迹） -----------------
class TransformerBranch(nn.Module):
    def __init__(self, input_dim=5, seq_len=60, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        self.encoder_layers = nn.ModuleList([
            CustomEncoderLayer(hidden_dim, nhead) for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attn_maps = []

    def forward(self, x):  # x: [B, T, C]
        x = self.input_proj(x) + self.pos_embedding[:, :x.size(1), :]
        self.attn_maps = []
        for layer in self.encoder_layers:
            x = layer(x)
            self.attn_maps.append(layer.attn_weights)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return x


# ----------------- CNN 分支（图像输入） -----------------

class CNNBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None # EfficientNet_B0
        base_model = efficientnet_b0(weights=weights)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])  # 移除分类头
        self.out_dim = base_model.classifier[1].in_features

    def forward(self, img):  # [B, 3, H, W]
        x = self.cnn(img)
        return x.view(x.size(0), -1)


# ----------------- 多模态融合主模型 -----------------
class MultimodalModel(nn.Module):
    def __init__(self, mlp_dim=20, trans_dim=128, cnn_dim=512, fusion_dim=256):
        super().__init__()
        self.mlp = MLPBranch(input_dim=mlp_dim)
        self.trans = TransformerBranch()
        self.cnn_hist = CNNBranch()
        self.cnn_airspace = CNNBranch()

        total_input_dim = 128 + 128 + self.cnn_hist.out_dim + self.cnn_airspace.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3), #
            nn.Linear(fusion_dim, 2)  # 输出 time_offset 和 duration
        )

    def forward(self, mlp_input, traj_seq, hist_img, airspace_img):
        mlp_out = self.mlp(mlp_input)
        trans_out = self.trans(traj_seq)
        hist_img_out = self.cnn_hist(hist_img)
        airspace_img_out = self.cnn_airspace(airspace_img)
        fused = torch.cat([mlp_out, trans_out, hist_img_out, airspace_img_out], dim=1)
        return self.fusion(fused)


# Cross-Attention Test
# class MultimodalModel(nn.Module):
#     def __init__(self, mlp_dim=20, trans_dim=128, cnn_dim=512, fusion_dim=256, nhead=4):
#         super().__init__()
#         self.mlp = MLPBranch(input_dim=mlp_dim)
#         self.trans = TransformerBranch()
#         self.cnn_hist = CNNBranch()
#         self.cnn_airspace = CNNBranch()
#
#         # 输出维度
#         self.mlp_dim = trans_dim
#         self.trans_dim = trans_dim
#         self.cnn_dim = self.cnn_hist.out_dim  # efficientnet_b0 = 1280
#
#         # Cross-Attention 层：MLP output 作为 Query
#         self.query_proj = nn.Linear(self.mlp_dim, fusion_dim)
#         self.kv_proj = nn.Linear(self.trans_dim + self.cnn_dim * 2, fusion_dim)
#
#         self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=nhead, batch_first=True)
#
#         # 后续融合层
#         self.fusion = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(fusion_dim, 2)  # 输出 time_offset 和 duration
#         )
#
#     def forward(self, mlp_input, traj_seq, hist_img, airspace_img):
#         # 各模态编码
#         mlp_out = self.mlp(mlp_input)          # [B, D]
#         trans_out = self.trans(traj_seq)       # [B, D]
#         hist_img_out = self.cnn_hist(hist_img)     # [B, C]
#         airspace_img_out = self.cnn_airspace(airspace_img)  # [B, C]
#
#         # 拼接 KV 组合
#         kv = torch.cat([trans_out, hist_img_out, airspace_img_out], dim=1)  # [B, D_total]
#         kv = self.kv_proj(kv).unsqueeze(1)  # [B, 1, fusion_dim]
#
#         # MLP 为 Query
#         query = self.query_proj(mlp_out).unsqueeze(1)  # [B, 1, fusion_dim]
#
#         # Cross-Attention
#         attn_out, _ = self.cross_attn(query, kv, kv)  # [B, 1, fusion_dim]
#         attn_out = attn_out.squeeze(1)  # [B, fusion_dim]
#
#         return self.fusion(attn_out)
