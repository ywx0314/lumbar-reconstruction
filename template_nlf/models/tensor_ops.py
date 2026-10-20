# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import torch
from typing import List, Optional

def linspace(start: float, stop: float, num: int, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, endpoint: bool = True):
    start = torch.as_tensor(start, device=device, dtype=dtype)
    stop = torch.as_tensor(stop, device=device, dtype=dtype)
    if endpoint:
        if num == 1:
            return torch.mean(torch.stack([start, stop], dim=0), dim=0, keepdim=True)
        return torch.linspace(start, stop, num, device=device, dtype=dtype)
    if num > 1:
        step = (stop - start) / num
        return torch.linspace(start, stop - step, num, device=device, dtype=dtype)
    return torch.linspace(start, stop, num, device=device, dtype=dtype)


def softmax(target: torch.Tensor, dim: List[int] = (-1,)):
    dim = [d if d >= 0 else d + target.ndim for d in dim]
    dim_min = min(dim)
    dim_max = max(dim)
    target_flattened = target.flatten(start_dim=dim_min, end_dim=dim_max)
    softmaxed_flattened = torch.softmax(target_flattened, dim=dim_min)
    return softmaxed_flattened.reshape(target.shape)


def decode_heatmap(inp: torch.Tensor, dim: List[int] = (-1,), output_coord_dim: int = -1):
    result = []
    dim = [d if d >= 0 else d + inp.ndim for d in dim]
    for d in dim:
        other_dims = list(dim)
        other_dims.remove(d)
        summed = torch.sum(inp, dim=other_dims, keepdim=True)
        coords = linspace(0.0, 1.0, inp.shape[d], dtype=inp.dtype, device=summed.device)
        decoded = torch.tensordot(summed, coords, dims=([d], [0]))
        x = torch.unsqueeze(decoded, d)
        for hd in sorted(dim)[::-1]:
            x = x.squeeze(hd)
        result.append(x)
    return torch.stack(result, dim=output_coord_dim)


def decode_heatmap2d(heatmap2d: torch.Tensor, num_views: int = 2) -> torch.Tensor:
    coords2d_flat = decode_heatmap(heatmap2d, dim=[3, 2], output_coord_dim=-1)
    nv, p, _ = coords2d_flat.shape
    n = nv // num_views
    return coords2d_flat.view(n, num_views, p, 2)


def charbonnier(x: torch.Tensor, epsilon: float = 1e-3, dim: int = -1, keepdim: bool = False):
    if epsilon == 0:
        return torch.linalg.norm(x, dim=-1, keepdim=keepdim) / x.shape[-1]
    return torch.sqrt(torch.mean(torch.square(x), dim=dim, keepdim=keepdim) + epsilon ** 2) - epsilon
