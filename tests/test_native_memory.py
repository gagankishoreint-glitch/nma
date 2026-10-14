import copy
import pytest
import torch
from memory.titans_memory import NeuralLongTermMemory, ByteCodec
from memory.encoder_bridge import HashEncoder
from memory.orchestrator import MemoryOrchestrator


def state(memory):
    return {k: v.clone() for k, v in memory.state_dict().items()}


def test_native_weights_change_reads_do_not_reset_exact():
    torch.set_num_threads(1)
    memory = NeuralLongTermMemory(input_dim=12, hidden_dim=8, mem_dim=8, max_bytes=32)
    x = torch.randn(12)
    initial = state(memory)
    loss = memory.test_time_update(x, '721 Example Lane')
    changed = state(memory)
    assert loss > 0 and memory.update_count == 1
    assert any(not torch.equal(initial[k], v) for k, v in changed.items())
    assert memory.read(x).shape == (1, 8)
    memory.compute_surprise(x, '721 Example Lane')
    memory.recall_text(x)
    assert all(torch.equal(changed[k], v) for k, v in memory.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in memory.parameters())
    memory.reset()
    assert all(torch.equal(initial[k], v) for k, v in memory.state_dict().items())
    assert memory.update_count == 0


def test_bad_write_atomic_flags_restored_and_no_graph():
    mem = NeuralLongTermMemory(input_dim=4, mem_dim=4, hidden_dim=4, max_bytes=4)
    before = state(mem)
    with pytest.raises(ValueError):
        mem.test_time_update(torch.ones(4), 'too much data')
    assert all(torch.equal(before[k], v) for k, v in mem.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in mem.parameters())
    with torch.no_grad():
        mem.test_time_update(torch.ones(4))
    with torch.inference_mode(), pytest.raises(RuntimeError):
        mem.test_time_update(torch.ones(4))
    with pytest.raises(ValueError):
        mem.test_time_update(torch.tensor([float('nan')]*4))


def test_byte_bridge_roundtrip_and_invalid_outputs_abstain():
    codec = ByteCodec(32)
    text = '茶馆, Beijing'
    logits = (codec.encode(text)*2-1)*20
    result = codec.decode(logits)
    assert result.text == text
    assert codec.decode(torch.zeros(codec.width)).text is None
    invalid = logits.clone()
    invalid[:16] = 20  # impossible length
    assert codec.decode(invalid).reason == 'invalid_frame'
    with pytest.raises(ValueError):
        codec.encode('x'*33)


def test_no_audit_store_needed_for_neural_read():
    encoder = HashEncoder(12)
    memory = NeuralLongTermMemory(input_dim=12, mem_dim=8, hidden_dim=8, max_bytes=32)
    audit = []
    orchestrator = MemoryOrchestrator(encoder, memory, audit.append)
    orchestrator.observe_slot('restaurant_address', '721 Example Lane')
    first = orchestrator.recall(['restaurant_address'])
    audit.clear()
    # Replace the audit sink with a function that would fail if read used it.
    orchestrator.audit_sink = lambda _: (_ for _ in ()).throw(AssertionError('audit access'))
    assert first == orchestrator.recall(['restaurant_address'])
    assert not hasattr(orchestrator, 'episodic_log')
    assert all(not isinstance(v, str) for v in memory.state_dict().values())
    with pytest.raises(ValueError):
        orchestrator.recall(['invented_slot'])


def test_hash_encoder_deterministic_and_rng_independent():
    a = HashEncoder().encode('restaurant address')
    torch.manual_seed(400)
    b = HashEncoder().encode('restaurant address')
    assert torch.equal(a, b) and torch.isclose(a.norm(), torch.tensor(1.))


def test_two_slots_are_one_gradient_step_per_observation():
    encoder = HashEncoder(12)
    mem = NeuralLongTermMemory(input_dim=12, mem_dim=8, hidden_dim=8, max_bytes=32)
    manager = MemoryOrchestrator(encoder, mem)
    manager.observe_slots({'restaurant_name': 'Lantern', 'restaurant_address': '721 Example Lane'})
    assert mem.update_count == 1
