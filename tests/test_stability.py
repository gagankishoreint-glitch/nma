"""Opt-in genuine-model probes. Never pass via a scripted/mock LLM."""
import os
import pytest
from llm.stable_llm import StableLLM
from src.eval.agent_stability import run_stability_suite
from src.utils.agent_profile import ResourceProfiler


@pytest.fixture(scope='module')
def real_stability():
    path = os.environ.get('AGENT_GGUF_PATH')
    if not path:
        pytest.skip('No local GGUF: real LLM stability is NOT measured')
    profiler = ResourceProfiler().start()
    llm = StableLLM(path, n_gpu_layers=0)
    try:
        yield run_stability_suite(llm, profiler)
    finally:
        llm.close()
        profiler.close()


def check(report, name):
    test = next(t for t in report['tests'] if t['name'] == name)
    assert test['passed'], test


def test_tone_consistency(real_stability):
    check(real_stability, 'tone_consistency')


def test_memory_does_not_become_persona(real_stability):
    check(real_stability, 'memory_does_not_become_persona')


def test_instruction_injection_ignored(real_stability):
    check(real_stability, 'instruction_injection_ignored')


def test_goal_wins_over_memory(real_stability):
    check(real_stability, 'goal_wins_over_memory')
