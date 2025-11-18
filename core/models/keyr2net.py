"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/models
@File         : keyr2net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.res_conv(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class KeyR2Net(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(KeyR2Net, self).__init__()
        self.res1 = ResidualBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.res2 = ResidualBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.res3 = ResidualBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Dropout + Adaptive pooling
        self.dropout_cnn = nn.Dropout2d(p=dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 32))

        # RNN
        self.gru_hidden = 256
        self.gru_input_size = 960   # [C*F] = 64 canales * 15 freq bins
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden,
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate
        )

        # Dropout antes de fc
        self.dropout_fc = nn.Dropout(p=dropout_rate)

        # Clasificación
        self.fc = nn.Linear(self.gru_hidden*2, num_classes)  # *2 por bidireccional

    def forward(self, x):
        # CNN
        x = self.res1(x)
        x = self.pool1(x)

        x = self.res2(x)
        x = self.pool2(x)

        x = self.res3(x)
        x = self.pool3(x)

        x = self.dropout_cnn(x)
        x = self.adaptive_pool(x)

        # Reorganizar para RNN
        b, c, freq, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F]
        x = x.view(b, t, c * freq)              # [B, T, 512]

        # RNN
        x, h_n = self.gru(x)  # h_n: [num_layers*2, B, hidden]
        h_n = h_n.view(self.gru.num_layers, 2, b, self.gru_hidden) # Concatenar últimos hidden states de ambas direcciones
        last_hidden = torch.cat((h_n[-1,0], h_n[-1,1]), dim=1)  # [B, 512]

        # Dropout y clasificación
        x = self.dropout_fc(last_hidden)
        out = self.fc(x)
        return out