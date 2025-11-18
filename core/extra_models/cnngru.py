"""
@Date         : 13-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/extra_models
@File         : cnngru.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGRU(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(CNNGRU, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1) 
        self.bn1   = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        self.dropout_cnn = nn.Dropout2d(p=dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 32))

        # RNN
        self.gru_hidden = 256
        self.gru_input_size = 960 # [C*F] = 64 canales * 15 freq bins
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
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