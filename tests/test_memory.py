import io
import pytest
import torch
from src.models.aomm import AppendOnlyMemory
from src.models.retrieval import TemporalAttention


def test_append_is_immutable_and_detached():
    torch.manual_seed(0)
    mem = AppendOnlyMemory(8, 2, 4)
    source = torch.randn(1, 8, requires_grad=True)
    mem.append(source, 1, use_gate=False)
    old = (mem.keys[:1].clone(), mem.values[:1].clone(), mem.timestamps[:1].clone())
    mem.append(torch.randn(2, 8), [2, 3], use_gate=False)
    assert mem.size == 3
    assert all(torch.equal(a, b) for a, b in zip(old, (mem.keys[:1], mem.values[:1], mem.timestamps[:1])))
    assert not mem.keys.requires_grad and not mem.values.requires_grad
    assert source.grad is None
    assert mem.write_mask.sum() == 3


def test_overflow_atomic_and_reset():
    mem = AppendOnlyMemory(8, 2, 1)
    mem.append(torch.randn(1, 8), 1, use_gate=False)
    old = mem.keys.clone()
    with pytest.raises(OverflowError):
        mem.append(torch.randn(2, 8), [2, 3], use_gate=False)
    assert torch.equal(old, mem.keys) and mem.size == 1
    mem.reset()
    assert mem.size == 0 and not mem.write_mask.any() and not mem.keys.any()
    assert torch.equal(mem.retrieve(torch.ones(1, 1, 8)), torch.zeros(1, 1, 8))


def test_gate_rejection():
    mem = AppendOnlyMemory(8, 2, 4, gate_threshold=1.)
    accepted = mem.append(torch.zeros(2, 8), [0, 1])
    assert not accepted.any() and mem.size == 0


def test_roundtrip_store():
    mem = AppendOnlyMemory(8, 2, 4)
    mem.append(torch.randn(2, 8), [1, 2], use_gate=False)
    copy = AppendOnlyMemory(8, 2, 4)
    file = io.BytesIO()
    torch.save(mem.state_dict(), file)
    file.seek(0)
    copy.load_state_dict(torch.load(file, weights_only=True))
    assert copy.size == mem.size
    assert torch.equal(copy.keys, mem.keys)


def test_temporal_attention_prefers_nearby_and_masks_future():
    attn = TemporalAttention(4, 1, temporal=True)
    with torch.no_grad():
        attn.query_proj.weight.copy_(torch.eye(4))
        attn.out_proj.weight.copy_(torch.eye(4))
        attn.raw_decay.fill_(2.)
    q, k, v = torch.zeros(1, 1, 4), torch.zeros(2, 4), torch.eye(4)[:2]
    times = torch.tensor([1., 10.])
    result = attn(q, k, v, times, 10)
    assert result[0, 0, 1] > result[0, 0, 0]
    result = attn(q, k, v, times, 1, causal=True)
    assert torch.allclose(result[0, 0], v[0])
    result = attn(q, k, v, times, 0, causal=True)
    assert torch.isfinite(result).all() and not result.any()
    with pytest.raises(ValueError):
        attn(q, k, v)


def test_append_does_not_guarantee_read_stability():
    attn = TemporalAttention(4, 1)
    with torch.no_grad():
        attn.out_proj.weight.copy_(torch.eye(4))
    q = torch.zeros(1, 1, 4)
    first = attn(q, torch.zeros(1, 4), torch.ones(1, 4))
    second = attn(q, torch.zeros(2, 4), torch.cat([torch.ones(1, 4), -torch.ones(1, 4)]))
    assert not torch.equal(first, second)  # denominator competition even with frozen read weights


def test_episode_projection_gradients_without_store_mutation():
    mem = AppendOnlyMemory(8, 2, 4)
    loss = mem.episode_prefix(torch.randn(1, 1, 8), torch.randn(3, 8)).square().mean()
    loss.backward()
    assert mem.key_proj.weight.grad is not None
    assert mem.value_proj.weight.grad is not None
    assert mem.size == 0
