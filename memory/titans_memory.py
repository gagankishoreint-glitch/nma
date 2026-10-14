"""Small online associative MLP, NOT the published Titans algorithm.

Values live in learned parameters, not a hidden text/codebook lookup. A fixed
UTF-8 bit codec exposes the otherwise missing representation-to-text bridge.
Each observation performs exactly ONE SGD step. This is deliberately a hard
one-shot baseline; an untrained decoder is allowed to fail/abstain.
"""
from dataclasses import dataclass
import copy
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecodedMemory:
    text: str | None
    confidence: float
    reason: str


class ByteCodec:
    def __init__(self, max_bytes=96):
        if not 1 <= max_bytes <= 65535:
            raise ValueError('max_bytes must be in [1, 65535]')
        self.max_bytes = max_bytes
        self.width = (max_bytes + 2) * 8

    def encode(self, text):
        raw = text.encode('utf-8')
        if len(raw) > self.max_bytes:
            raise ValueError('Value exceeds byte capacity; no truncation permitted')
        frame = len(raw).to_bytes(2, 'big') + raw + bytes(self.max_bytes-len(raw))
        return torch.tensor([(byte >> bit) & 1 for byte in frame
                             for bit in range(7, -1, -1)], dtype=torch.float32)

    def decode(self, logits, min_confidence=.8):
        probs = logits.detach().float().sigmoid().flatten().cpu()
        if probs.numel() != self.width:
            raise ValueError('Incorrect codec shape')
        confidence = float(torch.maximum(probs, 1-probs).min())
        if not torch.isfinite(probs).all() or confidence < min_confidence:
            return DecodedMemory(None, confidence, 'low_confidence')
        bits = (probs >= .5).reshape(-1, 8)
        frame = bytes(sum(int(b) << (7-i) for i, b in enumerate(row)) for row in bits)
        length = int.from_bytes(frame[:2], 'big')
        if length == 0 or length > self.max_bytes or any(frame[2+length:]):
            return DecodedMemory(None, confidence, 'invalid_frame')
        try:
            text = frame[2:2+length].decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            return DecodedMemory(None, confidence, 'invalid_utf8')
        if any(ord(c) < 32 for c in text):
            return DecodedMemory(None, confidence, 'control_character')
        return DecodedMemory(text, confidence, 'decoded')


class NeuralLongTermMemory(nn.Module):
    def __init__(self, input_dim=384, mem_dim=64, hidden_dim=32, num_layers=2,
                 lr=.1, weight_decay=0., max_bytes=96, seed=42,
                 min_confidence=.8):
        super().__init__()
        if num_layers < 1 or min(input_dim, mem_dim, hidden_dim) < 1:
            raise ValueError('Invalid dimensions')
        if lr <= 0 or weight_decay < 0 or not .5 <= min_confidence <= 1:
            raise ValueError('Invalid learning or confidence settings')
        self.input_dim, self.mem_dim = input_dim, mem_dim
        self.lr, self.weight_decay = lr, weight_decay
        self.min_confidence = min_confidence
        self.codec = ByteCodec(max_bytes)
        # Isolate initialization from unrelated model/random seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            dims = [input_dim] + [hidden_dim]*(num_layers-1) + [mem_dim]
            layers = []
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2:
                    layers.append(nn.GELU())
            self.memory_mlp = nn.Sequential(*layers)
            self.predictor = nn.Linear(mem_dim, input_dim)
            self.text_decoder = nn.Linear(mem_dim, self.codec.width)
        self.requires_grad_(False)
        self.eval()
        self._initial_state = copy.deepcopy(self.state_dict())
        self.update_count = 0

    def _input(self, x):
        p = next(self.parameters())
        x = torch.as_tensor(x, dtype=p.dtype, device=p.device).detach()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != self.input_dim or not torch.isfinite(x).all():
            raise ValueError('Expected finite [batch, input_dim] input')
        return x

    @torch.no_grad()
    def read(self, x):
        return self.memory_mlp(self._input(x))

    def _loss(self, x, value_text=None):
        state = self.memory_mlp(x)
        if value_text is None:
            return F.mse_loss(self.predictor(state), x)
        texts = [value_text] if isinstance(value_text, str) else value_text
        if not isinstance(texts, list) or len(texts) != x.shape[0] or not all(isinstance(t, str) for t in texts):
            raise ValueError('Provide one UTF-8 target string per slot query')
        bits = torch.stack([self.codec.encode(t) for t in texts]).to(x.device)
        return F.binary_cross_entropy_with_logits(self.text_decoder(state), bits)

    @torch.no_grad()
    def compute_surprise(self, x, value_text=None):
        return float(self._loss(self._input(x), value_text))

    def test_time_update(self, x, value_text=None):
        """Pre-update prediction loss; single SGD step, no retained graph/state.

        Explicit slot/value supervision is required for textual recall. Plain
        embeddings only train reconstruction; they cannot magically decode facts.
        Must not be invoked inside inference_mode (no_grad is supported).
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Memory writes require autograd, not inference_mode')
        x = self._input(x)
        parameters = list(self.parameters())
        original_flags = [p.requires_grad for p in parameters]
        try:
            with torch.enable_grad():
                for p in parameters:
                    p.requires_grad_(True)
                loss = self._loss(x, value_text)
                grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            if not torch.isfinite(loss) or any(g is not None and not torch.isfinite(g).all() for g in grads):
                raise ValueError('Non-finite update; parameters unchanged')
            with torch.no_grad():
                for p, g in zip(parameters, grads):
                    if g is not None:
                        p.add_(g + self.weight_decay*p, alpha=-self.lr)
            self.update_count += 1
            return float(loss.detach())
        finally:
            for p, flag in zip(parameters, original_flags):
                p.requires_grad_(flag)
                p.grad = None

    @torch.no_grad()
    def recall_text(self, x):
        x = self._input(x)
        if x.shape[0] != 1:
            raise ValueError('One query per text read')
        return self.codec.decode(self.text_decoder(self.memory_mlp(x))[0],
                                 self.min_confidence)

    @torch.no_grad()
    def reset(self):
        # Exact original state, including both decoders; not new random weights.
        self.load_state_dict(self._initial_state)
        self.update_count = 0
