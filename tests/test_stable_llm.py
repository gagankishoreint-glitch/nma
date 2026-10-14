import sys
import types
import pytest
from llm.stable_llm import StableLLM
from memory.encoder_bridge import EncoderBridge
import torch


def test_wrapper_text_only_isolates_calls_and_disables_cache(tmp_path, monkeypatch):
    class FakeLlama:
        def __init__(self, **kwargs):
            self.config, self.calls, self.resets = kwargs, [], 0
        def set_cache(self, value):
            self.cache = value
        def reset(self):
            self.resets += 1
        def create_chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            return {'choices': [{'message': {'content': 'action'}}]}
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, 'llama_cpp', types.SimpleNamespace(Llama=FakeLlama))
    path = tmp_path/'fake.gguf'
    path.write_text('fixture only, never loaded by llama.cpp')
    llm = StableLLM(path)
    for _ in range(2):
        assert llm.generate('immutable policy', 'data') == 'action'
    assert llm._llm.resets == 2 and llm._llm.cache is None
    assert llm._llm.config['n_gpu_layers'] == 0
    assert not llm._llm.config['offload_kqv']
    assert all(len(call['messages']) == 2 for call in llm._llm.calls)
    with pytest.raises(TypeError):
        llm.generate(torch.zeros(2), 'data')
    with pytest.raises(ValueError):
        llm.generate('policy', 'a'*6001)


def test_sentence_encoder_is_independent_cpu_and_local_only(monkeypatch):
    class FakeEncoder:
        def __init__(self, name, **kwargs):
            assert kwargs == {'device': 'cpu', 'local_files_only': True}
        def eval(self):
            return self
        def get_sentence_embedding_dimension(self):
            return 384
        def encode(self, text, **kwargs):
            assert kwargs['device'] == 'cpu'
            return torch.ones(384)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=FakeEncoder))
    bridge = EncoderBridge('local-fixture')
    assert bridge.encode('text').shape == (384,)
