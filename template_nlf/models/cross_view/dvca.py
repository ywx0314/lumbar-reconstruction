# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
# Image-encoder design reference: https://github.com/ShilongHong/DAGNet
# Project copyright and SPDX notice cover project-owned contributions only.
# Architecture reference only; DAGNet source code is not bundled.
# See THIRD_PARTY_NOTICES.md for the architectural acknowledgment.
import torch
from torch import nn
from .activations import get_activation
from typing import Optional, Tuple


class DWLayer(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, bias: bool = False, act: Optional[str] = None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.dconv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=ch_in, bias=bias)
        self.wconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.norm(self.wconv(self.dconv(x)))))


class SinCosPositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, h: int, w: int, device: str = "cpu") -> torch.Tensor:
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = self.embed_dim // 4
        omega = 1.0 / (self.temperature ** (torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim))
        out_w = torch.einsum("i,j->ij", grid_w.flatten(), omega)
        out_h = torch.einsum("i,j->ij", grid_h.flatten(), omega)
        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1).unsqueeze(0)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, pos1: Optional[torch.Tensor] = None, pos2: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x2
        q = self.with_pos_embed(x1, pos1)
        k = self.with_pos_embed(x1, pos1)
        v = self.with_pos_embed(x2, pos2)
        x2, _ = self.self_attn(q, k, v, need_weights=False)
        return self.norm(residual + self.dropout(x2))


class DVCA(nn.Module):
    def __init__(self, ch_in: int, ch_next: Optional[int] = None, num_heads: int = 8):
        super().__init__()
        self.cross_attention = MultiHeadCrossAttention(d_model=ch_in, nhead=num_heads)
        self.norm = nn.BatchNorm2d(ch_in)
        self.pos_encoder_ol = SinCosPositionalEncoding2D(embed_dim=ch_in)
        self.pos_encoder_sd = SinCosPositionalEncoding2D(embed_dim=ch_in)
        self.ch_next = ch_next
        if self.ch_next is not None:
            self.ffn = DWLayer(ch_in, ch_next)
            self.spatial = nn.Conv2d(2, 1, 7, stride=1, padding=3)
            self.sigmoid = nn.Sigmoid()

    def forward(self, ap_features: torch.Tensor, lateral_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = ap_features.size()
        ap_tokens = ap_features.flatten(2).permute(0, 2, 1)
        lateral_tokens = lateral_features.flatten(2).permute(0, 2, 1)
        ap_positions = self.pos_encoder_ol(h=h, w=w, device=ap_features.device)
        lateral_positions = self.pos_encoder_sd(h=h, w=w, device=ap_features.device)
        updated_lateral_tokens = self.cross_attention(ap_tokens, lateral_tokens, ap_positions, lateral_positions)
        updated_ap_tokens = self.cross_attention(lateral_tokens, ap_tokens, lateral_positions, ap_positions)
        ap_features = self.norm(updated_ap_tokens.permute(0, 2, 1).view(b, c, h, w)) + ap_features
        lateral_features = self.norm(updated_lateral_tokens.permute(0, 2, 1).view(b, c, h, w)) + lateral_features
        ap_spatial_gate = lateral_spatial_gate = torch.ones(b, 1, h // 2, w // 2, device=ap_features.device)
        if self.ch_next is not None:
            ffn_ol = self.ffn(ap_features)
            ffn_sd = self.ffn(lateral_features)
            ap_spatial_gate = self.sigmoid(self.spatial(torch.cat((torch.max(ffn_ol, 1)[0].unsqueeze(1), torch.mean(ffn_ol, 1).unsqueeze(1)), dim=1)))
            lateral_spatial_gate = self.sigmoid(self.spatial(torch.cat((torch.max(ffn_sd, 1)[0].unsqueeze(1), torch.mean(ffn_sd, 1).unsqueeze(1)), dim=1)))
        return ap_features, lateral_features, ap_spatial_gate, lateral_spatial_gate
