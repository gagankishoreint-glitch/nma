import os
import pytest
import torch
from src.models.base import FrozenBase
from src.models.aomm import AppendOnlyMemory


def test_frozen_base_forward_and_prefix_gradient():
    torch.set_num_threads(1)
    base = FrozenBase(backend='tiny', dim=32, layers=1, heads=4)
    ids = base.ids('A question?')
    assert base.get_hidden_states(ids).shape == (1, ids.shape[1], 32)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    mem = AppendOnlyMemory(32, 4, 4)
    prefix = mem.episode_prefix(torch.randn(1, 1, 32), torch.randn(2, 32))
    base(ids, prefix).square().mean().backward()
    assert mem.prefix_proj.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    mem.append(torch.randn(1, 32), 1, use_gate=False)
    assert base.forward_with_memory(ids, mem).shape[:2] == ids.shape
    base.train()
    assert not base.model.training


@pytest.mark.skipif(not os.environ.get('GPT2_LOCAL_PATH'), reason='Requires genuine local GPT-2 checkpoint')
def test_real_gpt2_forward():
    base = FrozenBase(model_name=os.environ['GPT2_LOCAL_PATH'], local_files_only=True)
    ids = base.ids('What is the capital of Zarnia?')
    assert base.get_hidden_states(ids).shape == (1, ids.shape[1], 768)
    assert base(ids).shape[-1] == 50257
    assert not any(p.requires_grad for p in base.parameters())
