"""Frozen causal LM; differentiable soft-prefix path, never no_grad around it."""
import string
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer


class CharacterTokenizer:
    """Offline smoke-test tokenizer; NOT the GPT-2 tokenizer."""
    def __init__(self):
        self.chars = ['<pad>', '<eos>'] + list(string.printable)
        self.vocab = {c: i for i, c in enumerate(self.chars)}
        self.pad_token_id, self.eos_token_id = 0, 1

    def encode(self, text, **kwargs):
        return [self.vocab[c] for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(self.chars[int(i)] for i in ids if int(i) > 1)

    def __len__(self):
        return len(self.chars)


class FrozenBase(nn.Module):
    def __init__(self, backend='gpt2', model_name='gpt2', local_files_only=False,
                 dim=48, layers=2, heads=4):
        super().__init__()
        self.backend = backend
        if backend == 'tiny':
            self.tokenizer = CharacterTokenizer()
            config = GPT2Config(vocab_size=len(self.tokenizer), n_embd=dim,
                                n_layer=layers, n_head=heads, n_positions=256,
                                resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
                                pad_token_id=0, eos_token_id=1)
            self.model = GPT2LMHeadModel(config)
        elif backend == 'gpt2':
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=local_files_only)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name, local_files_only=local_files_only,
                output_hidden_states=True, attn_implementation='eager')
        else:
            raise ValueError(f'Unknown backend: {backend}')
        self.model.requires_grad_(False)
        self.model.eval()
        self.dim = self.model.config.n_embd

    @property
    def device(self):
        return next(self.model.parameters()).device

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def ids(self, text):
        return torch.tensor([self.tokenizer.encode(text)], device=self.device)

    def get_hidden_states(self, input_ids, attention_mask=None):
        with torch.no_grad():
            return self.model.transformer(input_ids=input_ids,
                                          attention_mask=attention_mask).last_hidden_state

    def forward(self, input_ids, prefix=None):
        if prefix is None:
            return self.model(input_ids=input_ids).logits
        embeddings = self.model.transformer.wte(input_ids)
        inputs = torch.cat([prefix, embeddings], dim=1)
        # Frozen parameters still transmit gradients to the prefix.
        return self.model(inputs_embeds=inputs).logits[:, prefix.shape[1]:]

    def forward_with_memory(self, input_ids, memory_module, query_times=None):
        hidden = self.get_hidden_states(input_ids)[:, -1:, :]
        if memory_module.size == 0:
            return self(input_ids)
        prefix = memory_module.soft_prefix(hidden, query_times=query_times)
        return self(input_ids, prefix)
