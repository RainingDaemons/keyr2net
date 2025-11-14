"""
@Date         : 13-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : extra_models
@File         : lstm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(LSTM, self).__init__()
        
        # LSTM
        self.num_layers = 2
        self.hidden_size = 256
        self.input_size = 120 # total bins
        self.lstm_out = None
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Dropout antes de fc
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        
        # Clasificación
        self.fc = nn.Linear(self.hidden_size * 2, num_classes) # *2 por bidireccional

    def forward(self, x):
        x = x.squeeze(1)       # [B, 120, 937]
        x = x.permute(0, 2, 1).contiguous()  # [B, 937, 120]

        # Secuencia de salidas y estados finales
        out, (h_n, c_n) = self.lstm(x)  # h_n: [num_layers*2, B, hidden]
        
        # Tomar los últimos hidden states de ambas direcciones
        h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size)
        last_hidden = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=1)  # [B, hidden*2]
        
        # Clasificación
        out = self.dropout_fc(last_hidden)
        out = self.fc(out)
        return out
