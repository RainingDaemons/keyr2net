"""
@Date         : 13-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : extra_models
@File         : cnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1) 
        self.bn1 = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        # Dropout + Adaptive pooling
        self.dropout_cnn = nn.Dropout2d(p=dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 32))

        # Clasificación
        self.fc = nn.Linear(64 * 8 * 32, num_classes)

    def forward(self, x):
        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.dropout_cnn(x)
        x = self.adaptive_pool(x) # [B, 64, freq', 32]

        # Dropout y clasificación
        b, c, f, t = x.size()
        x = x.view(b, c * f * t)

        out = self.fc(x)
        return out
