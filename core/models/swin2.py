"""
@Date         : 18-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/models
@File         : swin2.py

@Credits      : Code adapted from "Swin Transformer V2: Scaling Up Capacity and Resolution"
@URL          : https://arxiv.org/abs/2111.09883
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_input_for_swin(x, patch_size, window_size):
    """
    Corrige padding asegurando que:
        (H_out % window_size == 0)
        (W_out % window_size == 0)
    después del patch embedding.
    """

    B, C, H, W = x.shape

    # Pad to multiple of patch_size
    pad_h1 = (patch_size - (H % patch_size)) % patch_size
    pad_w1 = (patch_size - (W % patch_size)) % patch_size

    H2 = H + pad_h1
    W2 = W + pad_w1

    # After patch embedding
    Hp = H2 // patch_size
    Wp = W2 // patch_size

    # Pad so Hp, Wp are divisible by window_size
    pad_h2 = (window_size - (Hp % window_size)) % window_size
    pad_w2 = (window_size - (Wp % window_size)) % window_size

    # Convert back to pixel units
    pad_h2 *= patch_size
    pad_w2 *= patch_size

    # Final padding
    pad_h = pad_h1 + pad_h2
    pad_w = pad_w1 + pad_w2

    x = F.pad(x, (0, pad_w, 0, pad_h))

    # New shapes after patch embed
    Hp_final = (H + pad_h) // patch_size
    Wp_final = (W + pad_w) // patch_size

    assert Hp_final % window_size == 0, f"Hp={Hp_final} not divisible by {window_size}"
    assert Wp_final % window_size == 0, f"Wp={Wp_final} not divisible by {window_size}"

    return x, Hp_final, Wp_final

def pad_hw_for_windows(H, W, window_size):
    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size
    return pad_h, pad_w

def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C
    )
    return windows

def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size, W // window_size,
        window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        B, H, W, -1
    )
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (num_windows * B, Wh * Ww, C)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, heads, N, dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # window attention
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, dropout, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, H, W):
        """
        x: (B, H * W, C)
        """

        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # shift
        if self.shift:
            s = self.window_size // 2
            x = torch.roll(x, shifts=(-s, -s), dims=(1, 2))
        
        # partition windows
        windows = window_partition(x, self.window_size)  # (B*nW, Ws, Ws, C)
        windows = windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(windows)

        # reverse windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # undo shift
        if self.shift:
            s = self.window_size // 2
            x = torch.roll(x, shifts=(s, s), dims=(1, 2))
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape

        assert L == H * W, "Input features don't match H*W"

        # reshape
        x = x.view(B, H, W, C)

        # Fix pad when H or W is odd
        pad_h = H % 2
        pad_w = W % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w

        # do merging
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x, H // 2, W // 2

class SwinV2(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.3):
        super(SwinV2, self).__init__()
        # Parametros base
        self.patch_size=4
        self.window_size=7
        self.num_heads=[3, 6, 12]
        self.img_size=(120, 937)
        self.embed_dim=48 # adapted for 8GB GPUs

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, 
            self.embed_dim,
            self.patch_size,
            self.patch_size
        )

        # Stages
        self.stage1 = nn.ModuleList([
            SwinBlock(self.embed_dim, self.num_heads[0], self.window_size, dropout_rate),
            SwinBlock(self.embed_dim, self.num_heads[0], self.window_size, dropout_rate, shift=True)
        ])

        self.merge1 = PatchMerging(self.embed_dim)
        self.embed_dim *= 2

        self.stage2 = nn.ModuleList([
            SwinBlock(self.embed_dim, self.num_heads[1], self.window_size, dropout_rate),
            SwinBlock(self.embed_dim, self.num_heads[1], self.window_size, dropout_rate, shift=True)
        ])

        self.merge2 = PatchMerging(self.embed_dim)
        self.embed_dim *= 2

        self.stage3 = nn.ModuleList([
            SwinBlock(self.embed_dim, self.num_heads[2], self.window_size, dropout_rate),
            SwinBlock(self.embed_dim, self.num_heads[2], self.window_size, dropout_rate, shift=True)
        ])

        # Head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # --- pad to multiple of window size ---
        x, H, W = pad_input_for_swin(x, self.patch_size, self.window_size)

        # --- Patch embedding ---
        x = self.patch_embed(x)  # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H * W, C)

        # ---- Stage 1 ----
        for blk in self.stage1:
            x = blk(x, H, W)

        # ---- Merge 1 ----
        x, H, W = self.merge1(x, H, W)

        # Volver a padear para el siguiente stage 2
        pad_h, pad_w = pad_hw_for_windows(H, W, self.window_size)
        if pad_h or pad_w:
            x_map = x.view(B, H, W, -1)
            x_map = F.pad(x_map, (0, 0, 0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w
            x = x_map.view(B, H * W, -1)

        # ---- Stage 2 ----
        for blk in self.stage2:
            x = blk(x, H, W)

        # ---- Merge 2 ----
        x, H, W = self.merge2(x, H, W)

        # Volver a padear para el siguiente stage 3
        pad_h, pad_w = pad_hw_for_windows(H, W, self.window_size)
        if pad_h or pad_w:
            x_map = x.view(B, H, W, -1)
            x_map = F.pad(x_map, (0, 0, 0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w
            x = x_map.view(B, H * W, -1)

        # ---- Stage 3 ----
        for blk in self.stage3:
            x = blk(x, H, W)

        # ---- HEAD ----
        x = self.norm(x)
        x = x.mean(dim=1) # global average pooling
        x = self.fc(x)

        return x