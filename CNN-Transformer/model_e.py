from model import MLPBranch, TransformerBranch, CNNBranch
import torch.nn as nn
import torch


class MultimodalModel(nn.Module):
    def __init__(self, fusion_dim=256):
        super().__init__()
        self.trans = TransformerBranch()
        self.cnn_hist = CNNBranch()
        self.cnn_airspace = CNNBranch()

        total_input_dim = 128 + self.cnn_hist.out_dim + self.cnn_airspace.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2)
        )

    def forward(self, mlp_input, traj_seq, hist_img, airspace_img):
        trans_out = self.trans(traj_seq)
        hist_out = self.cnn_hist(hist_img)
        air_out = self.cnn_airspace(airspace_img)
        fused = torch.cat([trans_out, hist_out, air_out], dim=1)
        return self.fusion(fused)