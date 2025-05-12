# model_cnn_ablation.py（支持模态控制 + CNN 类型）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from model import MLPBranch, TransformerBranch


class CNNBranch(nn.Module):
    def __init__(self, cnn_type='efficientnet', pretrained=True):
        super().__init__()
        if cnn_type == 'resnet':
            base_model = models.resnet18(pretrained=pretrained)
            self.cnn = nn.Sequential(*list(base_model.children())[:-1])
            self.out_dim = base_model.fc.in_features
        elif cnn_type == 'custom':
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.out_dim = 64
        else:  # efficientnet
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = efficientnet_b0(weights=weights)
            self.cnn = nn.Sequential(*list(base_model.children())[:-1])
            self.out_dim = base_model.classifier[1].in_features

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)


class MultimodalModel(nn.Module):
    def __init__(self, mlp_dim=20, trans_dim=128, fusion_dim=256,
                 cnn_type='efficientnet',
                 use_mlp=True, use_transformer=True, use_hist_img=True, use_airspace_img=True):
        super().__init__()

        self.use_mlp = use_mlp
        self.use_transformer = use_transformer
        self.use_hist_img = use_hist_img
        self.use_airspace_img = use_airspace_img

        if self.use_mlp:
            self.mlp = MLPBranch(input_dim=mlp_dim)
        if self.use_transformer:
            self.trans = TransformerBranch()
        if self.use_hist_img:
            self.cnn_hist = CNNBranch(cnn_type=cnn_type)
        if self.use_airspace_img:
            self.cnn_airspace = CNNBranch(cnn_type=cnn_type)

        total_input_dim = 0
        if self.use_mlp:
            total_input_dim += 128
        if self.use_transformer:
            total_input_dim += 128
        if self.use_hist_img:
            total_input_dim += self.cnn_hist.out_dim
        if self.use_airspace_img:
            total_input_dim += self.cnn_airspace.out_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2)
        )

    def forward(self, mlp_input, traj_seq, hist_img, airspace_img):
        features = []
        if self.use_mlp:
            features.append(self.mlp(mlp_input))
        if self.use_transformer:
            features.append(self.trans(traj_seq))
        if self.use_hist_img:
            features.append(self.cnn_hist(hist_img))
        if self.use_airspace_img:
            features.append(self.cnn_airspace(airspace_img))

        fused = torch.cat(features, dim=1)
        return self.fusion(fused)
