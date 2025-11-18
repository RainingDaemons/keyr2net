"""
@Date         : 18-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/models
@File         : ast.py

@Credits      : Code adapted from "AST: Audio Spectrogram Transformer"
@URL          : https://arxiv.org/abs/2104.01778
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, input_shape, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        num_patches = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1])
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x) # (B, embed_dim, F', T')
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        return x

class AST(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(AST, self).__init__()
        # Parametros base
        self.num_heads=8
        self.num_layers=3
        self.d_model=320
        self.mlp_dim=640
        self.input_shape=(120, 937)
        self.patch_size=(16, 16)

        # Patch embedding
        self.patch_embed = PatchEmbed(self.input_shape, self.patch_size, 1, self.d_model)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.d_model))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(self.d_model, self.num_heads, self.mlp_dim, dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)

        # Classifier
        self.head = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, num_patches, d_model)

        # Add cls token
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer layers
        for blk in self.blocks:
            x = blk(x)

        # Classification on cls token
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits