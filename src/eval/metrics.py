"""Keep overall learning separate from forgetting of the SAME facts."""
import numpy as np
import torch


def recall_at_k(logits, target, k=1):
    return (logits.topk(k, dim=-1).indices == target[..., None]).any(-1).float().mean().item()


def mean_recall(values):
    return float(np.mean(values)) if len(values) else float('nan')


def forgetting_rate(history):
    """Mean per-item peak past accuracy minus final accuracy (triangular history)."""
    if len(history) < 2:
        return 0.
    final = history[-1]
    # Last-injected item has no subsequent retention measurement.
    drops = [max(row[i] for row in history[i:-1]) - final[i]
             for i in range(len(final) - 1)]
    return float(np.mean(drops))


def normalize_answer(text):
    return text.strip().rstrip('.').strip().casefold()


def exact_match(prediction, answer):
    return float(normalize_answer(prediction) == normalize_answer(answer))
