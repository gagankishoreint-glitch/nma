"""Public llama.cpp chat API only. Text messages, no LLM weight/adapter edits.

Fixed prompts and delimiters are mitigations, not a behavioral/security proof.
"""
from pathlib import Path
import threading


class StableLLM:
    backend = 'llama_cpp'

    def __init__(self, model_path, n_ctx=1024, n_gpu_layers=0, seed=42,
                 n_threads=2, max_context_bytes=6000):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'GGUF checkpoint missing: {model_path}')
        from llama_cpp import Llama
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx,
                          n_gpu_layers=n_gpu_layers, offload_kqv=n_gpu_layers > 0,
                          n_threads=n_threads, seed=seed, verbose=False)
        self._llm.set_cache(None)
        self._lock = threading.Lock()
        self.max_context_bytes = max_context_bytes

    def generate(self, system, user, max_tokens=256, temperature=0.):
        if not isinstance(system, str) or not isinstance(user, str):
            raise TypeError('LLM accepts strings only')
        if len(user.encode()) > self.max_context_bytes:
            raise ValueError('Task context exceeds explicit byte budget')
        # Calls are independent. No extracted context or prior conversation is
        # implicitly reused. llama.cpp also enforces its actual token limit.
        with self._lock:
            self._llm.reset()
            result = self._llm.create_chat_completion(
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': user}],
                max_tokens=max_tokens, temperature=temperature)
        content = result['choices'][0]['message']['content']
        if not isinstance(content, str):
            raise ValueError('Chat backend returned no text')
        return content

    def close(self):
        self._llm.close()
