# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act: str, inpace: bool = True):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act in ("silu", "swish"):
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unknown activation: {act}")
    if hasattr(m, "inplace"):
        m.inplace = inpace
    return m
