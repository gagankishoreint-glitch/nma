import json
import pytest
import torch
from agent import NativeMemoryAgent, run_benchmark
from llm.scripted import ScriptedLLM
from memory.encoder_bridge import HashEncoder
from src.data.restaurant_task import RestaurantWorld, parse_action
from src.utils.agent_profile import ResourceProfiler


def test_tool_boundary_rejects_injection_and_fake_completion():
    for text in ['[DONE]', '{"tool":"shell","arguments":{"command":"rm -rf /"}}',
                 '{"tool":"send_address","arguments":{"address":"x","extra":"command"}}']:
        with pytest.raises(ValueError):
            parse_action(text)
    world = RestaurantWorld()
    assert not world.execute(2, {'tool': 'send_address', 'arguments': {'address': 'invented'}}).succeeded


def test_three_sessions_controls_and_weight_ablation(tmp_path):
    torch.set_num_threads(1)
    llm, encoder = ScriptedLLM(), HashEncoder()
    profiler = ResourceProfiler()
    # Portable unit test, not GPU telemetry validation.
    profiler.gpu_available = False
    summaries = run_benchmark(llm, encoder, profiler, tmp_path, seeds=[0])
    by_method = {row['method']: row for row in summaries}
    assert by_method['context_window']['session3_address_recall']
    assert not by_method['no_memory']['session3_address_recall']
    assert not by_method['full']['session3_address_recall']
    hashes = json.loads((tmp_path/'memory_hashes.json').read_text())
    for row in hashes:
        if row['method'] in ('full', 'neural_only'):
            assert row['updates'] > 0 and row['before'] != row['after']
        else:
            assert row['updates'] == 0 and row['before'] == row['after']
    assert 'PLUMBING FIXTURE' in (tmp_path/'PROCEDURES.md').read_text()
    # Check extraction call isolation: never includes the main agent's system.
    main = [c for c in llm.calls if c['user'].startswith('{')]
    assert len({c['system'] for c in main}) == 1
    assert all(isinstance(c['system'], str) and isinstance(c['user'], str) for c in llm.calls)


def test_empty_context_window_cannot_recall_prior_fact(tmp_path):
    llm = ScriptedLLM()
    agent = NativeMemoryAgent(llm, 'context_window', context_turns=0, print_scope=False)
    world, profiler = RestaurantWorld(7), ResourceProfiler()
    profiler.gpu_available = False
    assert agent.run_session(0, world, profiler, seed=7)[0]
    assert not agent.run_session(2, world, profiler, seed=7)[0]


def test_profiler_missing_gpu_is_null_not_fake_zero():
    profiler = ResourceProfiler()
    profiler.gpu_available = False
    snapshot = profiler.snapshot('unit_test')
    assert snapshot['peak_device_used_mib'] is None
    assert snapshot['gpu_device_used_mib'] is None


def test_real_benchmark_cannot_bypass_stability_gate(tmp_path):
    class Uncertified:
        backend = 'llama_cpp'
    with pytest.raises(RuntimeError, match='stability'):
        run_benchmark(Uncertified(), HashEncoder(), ResourceProfiler(), tmp_path, [0])


def test_sampled_vram_guard_rejects_at_35gib():
    from src.utils.agent_profile import BudgetExceeded
    profiler = ResourceProfiler()
    profiler.gpu_available = False
    profiler.peak = 3584
    with pytest.raises(BudgetExceeded):
        profiler.snapshot('mocked_budget_test')
