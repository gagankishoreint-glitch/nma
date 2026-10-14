import torch
from src.models.base import FrozenBase
from src.train.baselines import LastLayerFineTune, TokenKNN
from src.data.synthetic_facts import generate_facts
from src.train.train_aomm import qa_ids, digest


def test_baselines_leave_base_frozen():
    torch.set_num_threads(1)
    base = FrozenBase(backend='tiny', dim=16, layers=1, heads=4)
    start = digest(base)
    head = LastLayerFineTune(base)
    before = head.head.weight.clone()
    prompt, answer = qa_ids(base, generate_facts(1)[0])
    head.update(base, prompt, answer, 2)
    assert not torch.equal(before, head.head.weight)
    knn = TokenKNN()
    knn.append(base, prompt, answer)
    assert len(knn.tokens) == answer.numel()
    scores = knn.logits(base, prompt)
    assert scores.shape == (1, len(base.tokenizer))
    assert torch.allclose(scores.exp().sum(-1), torch.ones(1), atol=1e-5)
    assert digest(base) == start
