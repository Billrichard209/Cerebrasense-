"""Multimodal architecture combining 3D CNNs with tabular clinical features."""

import torch
import torch.nn as nn

from src.utils.monai_utils import load_monai_network_symbols

class OASISMultimodalDenseNet(nn.Module):
    """
    Fuses DenseNet121 3D MRI features with clinical tabular data (Age, Sex, MMSE).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float = 0.2,
        tabular_features: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        dense_net_cls = load_monai_network_symbols()["DenseNet121"]
        self.densenet = dense_net_cls(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )
        
        # In MONAI's DenseNet121, the class_layers module ends with an 'out' Linear layer.
        # It takes 1024 features to `out_channels`. We'll replace it.
        self.cnn_feature_dim = self.densenet.class_layers.out.in_features
        self.densenet.class_layers.out = nn.Identity()
        
        # Process tabular data (e.g. Age, Sex, MMSE)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Fuse CNN and Tabular features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, img: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        # Extract image features (B, 1024)
        cnn_features = self.densenet(img)
        
        # Extract tabular features (B, 32)
        tab_features = self.tabular_mlp(tabular)
        
        # Concatenate and classify (B, out_channels)
        fused = torch.cat((cnn_features, tab_features), dim=1)
        logits = self.fusion_mlp(fused)
        
        return logits
