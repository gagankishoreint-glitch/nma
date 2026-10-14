"""Honest controls. None of these is ROME or MEMIT.

TokenKNN implements a small exact-search token kNN-LM-style interpolation using
teacher-forced question/answer contexts, NOT an official reproduction benchmark.
LastLayerFineTune edits an independent untied output head; GPT-2 stays frozen.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LastLayerFineTune(nn.Module):
    def __init__(self, base, lr=1e-3):
        super().__init__()
        self.head = nn.Linear(base.dim, len(base.tokenizer), bias=False).to(base.device)
        self.head.weight.data.copy_(base.model.lm_head.weight.detach())
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.)

    def logits(self, base, ids):
        return self.head(base.get_hidden_states(ids))

    def update(self, base, prompt_ids, answer_ids, steps):
        ids = torch.cat([prompt_ids, answer_ids], dim=1)
        hidden = base.get_hidden_states(ids)[:, prompt_ids.shape[1]-1:-1]
        for _ in range(steps):
            loss = F.cross_entropy(self.head(hidden).reshape(-1, self.head.out_features),
                                   answer_ids.flatten())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return float(loss.detach())


class TokenKNN:
    def __init__(self, k=8, temperature=1., interpolation=.5):
        self.k, self.temperature, self.interpolation = k, temperature, interpolation
        self.keys, self.tokens = [], []

    def append(self, base, prompt_ids, answer_ids):
        ids = torch.cat([prompt_ids, answer_ids], dim=1)
        hidden = base.get_hidden_states(ids)[:, prompt_ids.shape[1]-1:-1]
        self.keys.extend(hidden[0].detach().unbind())
        self.tokens.extend(answer_ids[0].tolist())

    def logits(self, base, ids):
        hidden = base.get_hidden_states(ids)[:, -1]
        base_probs = base.model.lm_head(hidden).softmax(-1)
        if not self.keys:
            return base_probs.log()
        # Squared L2, canonical kNN-LM metric; dense exact search at this small scale.
        distances = torch.cdist(hidden, torch.stack(self.keys)).square()
        distances, indices = distances.topk(min(self.k, len(self.keys)), largest=False)
        tokens = torch.tensor(self.tokens, device=ids.device)[indices]
        weights = (-distances / self.temperature).softmax(-1)
        probs = torch.zeros_like(base_probs).scatter_add_(1, tokens, weights)
        return ((1-self.interpolation)*base_probs + self.interpolation*probs).clamp_min(1e-30).log()
