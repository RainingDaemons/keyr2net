"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : models
@File         : cnn14.py

@Credits      : Code adapted from "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
@URL          : https://arxiv.org/abs/1912.10211
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class CNN14(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(CNN14, self).__init__()

        # 6 bloques convolucionales con pooling
        self.block1 = CNNBlock(1, 64)
        self.block2 = CNNBlock(64, 128)
        self.block3 = CNNBlock(128, 256)
        self.block4 = CNNBlock(256, 512)
        self.block5 = CNNBlock(512, 1024)
        self.block6 = CNNBlock(1024, 2048)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))

        # Fully connected
        self.fc1 = nn.Linear(2048*2, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.drop_fc1 = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        avg_p = self.avgpool(x).squeeze(-1).squeeze(-1)
        max_p = self.maxpool(x).squeeze(-1).squeeze(-1)
        x = torch.cat((avg_p, max_p), dim=1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        out = self.fc_out(x)
        return out