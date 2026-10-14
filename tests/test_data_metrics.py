import torch
import pytest
from src.data.synthetic_facts import generate_facts
from src.eval.metrics import exact_match, forgetting_rate, recall_at_k, mean_recall


def test_fictional_unique_reproducible_disjoint():
    a = generate_facts(100, 7)
    assert a == generate_facts(100, 7)
    assert a != generate_facts(100, 8)
    assert len({f.country for f in a}) == len({f.city for f in a}) == 100
    b = generate_facts(100, 7, 'calibration')
    assert not {f.country for f in a} & {f.country for f in b}
    assert not {f.city for f in a} & {f.city for f in b}
    assert all(f.city not in f.question for f in a)


def test_metrics_do_not_confuse_new_items_with_forgetting():
    assert forgetting_rate([[1], [1, 0], [1, 0, 0]]) == 0.
    assert forgetting_rate([[1], [0, 1], [0, 0, 1]]) == 1.
    assert exact_match(' Zorpel001.', 'Zorpel001') == 1.
    assert exact_match('Zorpel001 extra', 'Zorpel001') == 0.
    assert recall_at_k(torch.tensor([[1., 3., 2.]]), torch.tensor([2]), 2) == 1.
    assert mean_recall([0, 1]) == .5
    with pytest.raises(ValueError):
        generate_facts(0)
