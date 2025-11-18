"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/models
@File         : cnneef.py

@Credits      : Code adapted from "End-to-End Musical Key Estimation Using a Convolutional Neural Network"
@URL          : https://arxiv.org/abs/1706.02921
"""

import torch.nn as nn
import torch.nn.functional as F

class CNNEEF(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(CNNEEF, self).__init__()

        # 5 capas convolucionales (cada una 8 filtros 5x5 con padding)
        self.convs = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ELU(),
        )

        self.fc_freq = nn.Linear(8*120, 48)  # 8 filtros * 120 bins
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.convs(x)
        B, C, Freq, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C*Freq)
        x = self.fc_freq(x)
        x = F.elu(x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x
