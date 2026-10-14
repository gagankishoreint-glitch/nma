"""Independent CPU encoders; never request LLM hidden states."""
import hashlib
import re
import torch
from torch.nn import functional as F


class HashEncoder:
    """Deterministic offline plumbing encoder, NOT MiniLM/semantic evidence."""
    backend = 'hash_fixture'

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, text):
        vector = torch.zeros(self.dim)
        for token in re.findall(r'\w+', text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], 'big') % self.dim] += 1 if digest[4] % 2 else -1
        return F.normalize(vector, dim=0)


class EncoderBridge:
    backend = 'sentence_transformer'

    def __init__(self, model_name, local_files_only=True):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name, device='cpu',
                                           local_files_only=local_files_only)
        self.encoder.eval()
        self.dim = self.encoder.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, text):
        return self.encoder.encode(text, convert_to_tensor=True,
                                   normalize_embeddings=True, device='cpu').detach()
