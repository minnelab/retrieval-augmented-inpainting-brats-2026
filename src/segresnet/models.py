"""Custom architectures for the inpainting track.

UNetAttn3D: a plain 3D U-Net backbone (matching the 2025 winner: base-32 doubling, InstanceNorm,
PReLU, dropout 0.2) with a TRANSFORMER SELF-ATTENTION bottleneck. Vanilla U-Nets blur the centre
of large holes because their receptive field can't span the hole; self-attention at the deepest
level (small spatial size -> few tokens, cheap) gives a global receptive field there. This is our
bet to beat the winner's ordinary U-Net while keeping their training recipe.
"""
from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.blocks.transformerblock import TransformerBlock


def _double_conv(ci, co, dropout):
    return nn.Sequential(
        Convolution(3, ci, co, kernel_size=3, act="PRELU", norm="INSTANCE", dropout=dropout),
        Convolution(3, co, co, kernel_size=3, act="PRELU", norm="INSTANCE", dropout=dropout),
    )


class UNetAttn3D(nn.Module):
    """U-Net with a transformer bottleneck.

    feats = channels per level (len = #levels; downsamples = len-1). With feats of length 5 and a
    208x208x144 input the bottleneck is 13x13x9 (=1521 tokens). bottleneck_size must equal
    input_dims / 2**(len(feats)-1) and is fixed (train and infer use the same ROI).
    """
    def __init__(self, in_channels=2, out_channels=1, feats: Sequence[int] = (32, 64, 128, 256, 512),
                 dropout=0.2, attn_layers=4, attn_heads=8, bottleneck_size=(13, 13, 9)):
        super().__init__()
        self.feats = tuple(feats)
        self.bottleneck_size = tuple(bottleneck_size)

        self.in_block = _double_conv(in_channels, feats[0], dropout)
        self.down_samp = nn.ModuleList()   # strided conv (halve)
        self.down_block = nn.ModuleList()  # double conv at the new resolution
        for i in range(len(feats) - 1):
            self.down_samp.append(Convolution(3, feats[i], feats[i + 1], strides=2, kernel_size=3,
                                              act="PRELU", norm="INSTANCE", dropout=dropout))
            self.down_block.append(_double_conv(feats[i + 1], feats[i + 1], dropout))

        # transformer bottleneck over the deepest feature map (MONAI TransformerBlock = pre-norm
        # MHSA + MLP, the same blocks used inside UNETR/ViT). attn_layers=0 -> plain conv U-Net
        # (the faithful winner architecture, no attention).
        self.use_attn = attn_layers > 0
        if self.use_attn:
            dim = feats[-1]
            ntok = int(self.bottleneck_size[0] * self.bottleneck_size[1] * self.bottleneck_size[2])
            self.pos = nn.Parameter(torch.zeros(1, ntok, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.transformer = nn.ModuleList(
                [TransformerBlock(hidden_size=dim, mlp_dim=dim * 4, num_heads=attn_heads,
                                  dropout_rate=dropout) for _ in range(attn_layers)])
            self.tr_norm = nn.LayerNorm(dim)

        # decoder: transpose-conv upsample + skip concat + double conv
        self.up_samp = nn.ModuleList()
        self.up_block = nn.ModuleList()
        for i in range(len(feats) - 1, 0, -1):
            self.up_samp.append(nn.ConvTranspose3d(feats[i], feats[i - 1], kernel_size=2, stride=2))
            self.up_block.append(_double_conv(feats[i - 1] * 2, feats[i - 1], dropout))
        self.out_conv = nn.Conv3d(feats[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        h = self.in_block(x)
        skips.append(h)
        for samp, blk in zip(self.down_samp, self.down_block):
            h = blk(samp(h))
            skips.append(h)
        # h: (B, C, D, H, W) at bottleneck. attention over flattened tokens (if enabled).
        if self.use_attn:
            b, c, d, hh, w = h.shape
            t = h.flatten(2).transpose(1, 2) + self.pos   # (B, N, C)
            for blk in self.transformer:
                t = blk(t)
            t = self.tr_norm(t)
            h = t.transpose(1, 2).reshape(b, c, d, hh, w)
        # decode (skips[-1] is the bottleneck itself; use the shallower ones for concat)
        for j, (samp, blk) in enumerate(zip(self.up_samp, self.up_block)):
            h = samp(h)
            skip = skips[-(j + 2)]
            h = blk(torch.cat([h, skip], dim=1))
        return self.out_conv(h)
