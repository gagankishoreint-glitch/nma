"""Multihead attention, optionally penalizing timestamp distance."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads, temporal=False):
        super().__init__()
        if dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.temporal = temporal
        self.raw_decay = nn.Parameter(torch.tensor(-5.), requires_grad=temporal)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, keys, values, timestamps=None, query_times=None,
                causal=False):
        batch, length, dim = query.shape
        if keys.shape[-2] == 0:
            return torch.zeros_like(query)
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(batch, -1, -1)
            values = values.unsqueeze(0).expand(batch, -1, -1)
        h, d = self.num_heads, dim // self.num_heads
        q = self.query_proj(query).view(batch, length, h, d).transpose(1, 2)
        k = keys.reshape(batch, -1, h, d).transpose(1, 2)
        v = values.reshape(batch, -1, h, d).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d)
        if self.temporal or causal:
            if timestamps is None or query_times is None:
                raise ValueError('Temporal retrieval requires query and key timestamps')
            # query_times: scalar or [batch, length]; timestamps: [entries].
            qt = torch.as_tensor(query_times, device=query.device).expand(batch, length)
            delta = qt[:, None, :, None] - timestamps[None, None, None, :]
            if self.temporal:
                scores = scores - F.softplus(self.raw_decay) * delta.abs()
            if causal:
                scores = scores.masked_fill(delta < 0, -torch.inf)
        # Avoid all--inf softmax even in backward (nan_to_num after softmax is
        # insufficient: its backward would still propagate NaNs).
        valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
        safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=-1) * valid.to(scores.dtype)
        context = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.out_proj(context)
