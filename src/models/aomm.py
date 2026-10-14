"""Append-only detached store. Capacity overflow raises; never evicts silently.

Gate supervision is provided explicitly by training episodes, not inferred from
unobservable 'memory use'. Projection parameters must be frozen before streaming
for stored and query representations to remain aligned.
"""
import torch
from torch import nn
from .retrieval import TemporalAttention


class AppendOnlyMemory(nn.Module):
    def __init__(self, dim, num_heads=4, max_entries=256, prefix_length=4,
                 gate_threshold=.5, temporal=False):
        super().__init__()
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.write_gate = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(),
                                        nn.Linear(dim // 2, 1), nn.Sigmoid())
        self.attention = TemporalAttention(dim, num_heads, temporal)
        self.prefix_proj = nn.Linear(dim, prefix_length * dim)
        self.prefix_length, self.dim = prefix_length, dim
        self.gate_threshold = gate_threshold
        self.register_buffer('keys', torch.zeros(max_entries, dim))
        self.register_buffer('values', torch.zeros(max_entries, dim))
        self.register_buffer('timestamps', torch.zeros(max_entries))
        self.register_buffer('write_mask', torch.zeros(max_entries, dtype=torch.bool))
        self.register_buffer('_size', torch.zeros((), dtype=torch.long))

    @property
    def size(self):
        return int(self._size.item())

    @torch.no_grad()
    def append(self, hidden_states, timestamps, use_gate=True):
        hidden = hidden_states.reshape(-1, self.dim)
        times = torch.as_tensor(timestamps, device=hidden.device).expand(len(hidden))
        accept = self.write_gate(hidden).flatten() >= self.gate_threshold
        if not use_gate:
            accept = torch.ones_like(accept)
        count = int(accept.sum())
        start = self.size
        if start + count > len(self.keys):
            raise OverflowError('Append-only memory is full; no entries were changed')
        stop = start + count
        self.keys[start:stop].copy_(self.key_proj(hidden[accept]))
        self.values[start:stop].copy_(self.value_proj(hidden[accept]))
        self.timestamps[start:stop].copy_(times[accept])
        self.write_mask[start:stop] = True
        self._size.fill_(stop)
        return accept

    def retrieve(self, query_hidden, query_times=None, causal=False):
        return self.attention(query_hidden, self.keys[:self.size],
                              self.values[:self.size], self.timestamps[:self.size],
                              query_times, causal=causal)

    def prefix_from_context(self, context):
        return self.prefix_proj(context[:, -1]).view(-1, self.prefix_length, self.dim)

    def soft_prefix(self, query_hidden, query_times=None):
        return self.prefix_from_context(self.retrieve(query_hidden, query_times))

    def episode_prefix(self, query_hidden, support_hidden, timestamps=None,
                       query_times=None):
        """Differentiable temporary support, never mutates the deployment store."""
        context = self.attention(query_hidden, self.key_proj(support_hidden),
                                 self.value_proj(support_hidden), timestamps, query_times)
        return self.prefix_from_context(context)

    @torch.no_grad()
    def reset(self):
        self.keys.zero_()
        self.values.zero_()
        self.timestamps.zero_()
        self.write_mask.zero_()
        self._size.zero_()
