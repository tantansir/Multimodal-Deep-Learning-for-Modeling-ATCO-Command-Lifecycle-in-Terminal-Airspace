from model import MLPBranch, TransformerBranch, CNNBranch
import torch.nn as nn
import torch


class MultimodalModel(nn.Module):
    def __init__(self, mlp_dim=20, fusion_dim=256):
        super().__init__()
        self.mlp = MLPBranch(input_dim=mlp_dim)
        self.trans = TransformerBranch()

        total_input_dim = 128 + 128
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2)
        )

    def forward(self, mlp_input, traj_seq, hist_img, airspace_img):
        mlp_out = self.mlp(mlp_input)
        trans_out = self.trans(traj_seq)
        fused = torch.cat([mlp_out, trans_out], dim=1)
        return self.fusion(fused)
