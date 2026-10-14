"""Experiment 1: calibration before deployment, then append-only fact stream.

No online training of AOMM in the main condition: neither stored vectors nor the
read path drift during deployment. Retrieval competition can still hurt recall.
This is a narrower hypothesis than general elimination of forgetting.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from src.data.synthetic_facts import generate_facts
from src.eval.metrics import exact_match, forgetting_rate
from src.models.aomm import AppendOnlyMemory
from src.models.base import FrozenBase
from src.train.baselines import LastLayerFineTune, TokenKNN
from src.utils.logging import CSVLogger
from src.utils.seed import seed_everything


def digest(model):
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def qa_ids(base, fact):
    return base.ids(fact.question), base.ids(fact.answer)


def qa_loss(base, fact, prefix):
    prompt, answer = qa_ids(base, fact)
    logits = base(torch.cat([prompt, answer], 1), prefix)
    return F.cross_entropy(logits[:, prompt.shape[1]-1:-1].reshape(-1, logits.shape[-1]),
                           answer.flatten())


def calibrate(base, memory, cfg, seed, logger):
    """Disjoint support episodes. Auxiliary gate labels are explicit relevance labels.

    Support facts are positive; weather sentences are negative. This is privileged
    task supervision, not proof of a general learned utility policy. Store writes
    are detached; temporary calibration supports retain projection gradients.
    """
    facts = generate_facts(cfg['calibration_facts'], seed + 10000, 'calibration')
    support = torch.cat([base.get_hidden_states(base.ids(f.statement))[:, -1] for f in facts])
    distractors = torch.cat([base.get_hidden_states(base.ids(f.distractor))[:, -1] for f in facts])
    queries = [base.get_hidden_states(base.ids(f.question))[:, -1:] for f in facts]
    gate_inputs = torch.cat([support, distractors])
    labels = torch.cat([torch.ones(len(facts)), torch.zeros(len(facts))]).to(base.device)
    optimizer = torch.optim.AdamW([p for p in memory.parameters() if p.requires_grad],
                                  lr=cfg['lr'], weight_decay=0.)
    rng = random.Random(seed)
    start = time.perf_counter()
    memory.train()
    for step in range(cfg['calibration_steps']):
        index = rng.randrange(len(facts))
        # All calibration facts compete for retrieval, and questions carry no answer.
        times = torch.arange(len(facts), device=base.device, dtype=torch.float32)
        prefix = memory.episode_prefix(queries[index], support, times, len(facts)-1)
        answer_loss = qa_loss(base, facts[index], prefix)
        gate_probs = memory.write_gate(gate_inputs).flatten()
        gate_loss = F.binary_cross_entropy(gate_probs, labels)
        loss = answer_loss + cfg['gate_loss_weight'] * gate_loss + cfg['sparsity'] * gate_probs.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.)
        optimizer.step()
        logger.log(seed=seed, step=step+1, qa_loss=float(answer_loss.detach()),
                   gate_bce=float(gate_loss.detach()), elapsed_seconds=time.perf_counter()-start)
    memory.eval()
    memory.requires_grad_(False)
    with torch.no_grad():
        pred = memory.write_gate(gate_inputs).flatten() >= memory.gate_threshold
        return {'gate_calibration_accuracy': float((pred == labels.bool()).float().mean()),
                'calibration_seconds': time.perf_counter()-start}


@torch.no_grad()
def predict(base, fact, method, memory=None, head=None, knn=None, timestamp=None,
            max_new_tokens=24):
    ids = base.ids(fact.question)
    start = ids.shape[1]
    prefix = None
    if method.startswith('aomm') and memory.size:
        query = base.get_hidden_states(ids)[:, -1:]
        prefix = memory.soft_prefix(query, query_times=timestamp)
    output = []
    for _ in range(max_new_tokens):
        if method == 'last_layer_ft':
            logits = head.logits(base, ids)[:, -1]
        elif method == 'knn_lm':
            logits = knn.logits(base, ids)
        else:
            logits = base(ids, prefix)[:, -1]
        token = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, token], 1)
        output.append(int(token))
        text = base.tokenizer.decode(output, skip_special_tokens=True)
        if '.' in text or int(token) == base.tokenizer.eos_token_id:
            # Period is our declared answer delimiter; do not accept arbitrary suffixes.
            return text.split('.', 1)[0]
    return base.tokenizer.decode(ids[0, start:].tolist(), skip_special_tokens=True)


@torch.no_grad()
def diagnose(base, memory, facts, seed, timestamp, max_new_tokens):
    """Oracle single-entry retrieval isolates prefix decoding from competition.

    Attention ranking averages head probabilities, not decoder attribution.
    These are diagnostics, not extra accuracy baselines for the main figure.
    """
    rows = []
    for fact in facts:
        query = base.get_hidden_states(base.ids(fact.question))[:, -1:]
        hidden = base.get_hidden_states(base.ids(fact.statement))[:, -1]
        learned_prefix = memory.soft_prefix(query, query_times=timestamp) if memory.size else None
        oracle_prefix = memory.episode_prefix(query, hidden)
        prompt, _ = qa_ids(base, fact)
        ids, generated = prompt.clone(), []
        for _ in range(max_new_tokens):
            token = base(ids, oracle_prefix)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, token], 1)
            generated.append(int(token))
            text = base.tokenizer.decode(generated, skip_special_tokens=True)
            if '.' in text or int(token) == base.tokenizer.eos_token_id:
                break
        prediction = base.tokenizer.decode(generated, skip_special_tokens=True).split('.', 1)[0]
        q = memory.attention.query_proj(query).view(1, memory.attention.num_heads, -1)
        k = memory.keys[:memory.size].view(memory.size, memory.attention.num_heads,
                                                    base.dim // memory.attention.num_heads)
        scores = torch.einsum('bhd,nhd->bhn', q, k) / (k.shape[-1] ** .5) if memory.size else None
        if scores is not None:
            ranking = scores.softmax(-1).mean(1).argmax(-1).item()
            expected = memory.key_proj(hidden)[0]
            matches = torch.isclose(memory.keys[:memory.size], expected[None], atol=1e-5, rtol=1e-5).all(-1)
            retrieval_correct = int(matches[ranking])
        else:
            retrieval_correct = 0
        rows.append(dict(seed=seed, country=fact.country,
                         no_memory_nll=float(qa_loss(base, fact, None)),
                         learned_prefix_nll=float(qa_loss(base, fact, learned_prefix)),
                         oracle_prefix_nll=float(qa_loss(base, fact, oracle_prefix)),
                         oracle_exact_match=exact_match(prediction, fact.answer),
                         attention_top1_correct=retrieval_correct,
                         oracle_prediction=prediction, target=fact.answer))
    return rows


def run(config, output):
    cfg = config['experiment']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    torch.set_num_threads(cfg.get('cpu_threads', 1))
    methods = cfg['methods']
    supported = {'aomm', 'aomm_no_gate', 'aomm_temporal', 'no_memory', 'last_layer_ft', 'knn_lm'}
    if set(methods)-supported:
        raise ValueError(f'Unsupported methods: {set(methods)-supported}')
    fields = ['backend', 'seed', 'method', 'injection', 'fact_index', 'correct',
              'mean_recall', 'first_fact_recall', 'memory_entries', 'accepted_writes',
              'elapsed_seconds', 'prediction', 'target']
    logs = CSVLogger(output/'exp1_log.csv', fields)
    calibration_log = CSVLogger(output/'calibration.csv',
                                ['seed', 'step', 'qa_loss', 'gate_bce', 'elapsed_seconds'])
    write_log = CSVLogger(output/'writes.csv', ['seed', 'method', 'injection', 'kind',
                                               'gate_probability', 'accepted', 'memory_entries'])
    summaries, metadata, diagnostics = [], [], []
    total_start = time.perf_counter()
    try:
        for seed in cfg['seeds']:
            seed_everything(seed)
            base = FrozenBase(**config['base']).to(cfg.get('device', 'cpu'))
            initial_hash = digest(base)
            facts = generate_facts(cfg['num_facts'], seed)
            # Zero-shot evaluation occurs BEFORE any stream fact is exposed to any method.
            zero_start = time.perf_counter()
            zero_predictions = [predict(base, f, 'no_memory', max_new_tokens=cfg['max_new_tokens']) for f in facts]
            zero = [exact_match(p, f.answer) for p, f in zip(zero_predictions, facts)]
            zero_elapsed = time.perf_counter()-zero_start
            with open(output/f'zero_shot_seed{seed}.json', 'w') as file:
                json.dump([{'question': f.question, 'target': f.answer, 'prediction': p,
                            'correct': c} for f, p, c in zip(facts, zero_predictions, zero)], file, indent=2)
            memory = AppendOnlyMemory(base.dim, max_entries=2*len(facts), **config['memory']).to(base.device)
            cal_info = calibrate(base, memory, cfg, seed, calibration_log)
            # Gate/no-gate/time ablations share EXACTLY the same calibration weights.
            for method in methods:
                method_start = time.perf_counter()
                memory.reset()
                memory.attention.temporal = method == 'aomm_temporal'
                head = LastLayerFineTune(base, cfg['ft_lr']) if method == 'last_layer_ft' else None
                knn = TokenKNN(**config.get('knn', {})) if method == 'knn_lm' else None
                history, writes = [], 0
                for step, fact in enumerate(facts, 1):
                    if method.startswith('aomm'):
                        entries = [fact.statement]
                        if cfg.get('distractors', True):
                            entries.append(fact.distractor)
                        for entry_index, entry in enumerate(entries):
                            hidden = base.get_hidden_states(base.ids(entry))[:, -1]
                            accepted = int(memory.append(hidden, step, use_gate=method != 'aomm_no_gate').sum())
                            writes += accepted
                            write_log.log(seed=seed, method=method, injection=step,
                                          kind='fact' if entry_index == 0 else 'distractor',
                                          gate_probability=float(memory.write_gate(hidden).item()),
                                          accepted=accepted, memory_entries=memory.size)
                    elif head is not None:
                        head.update(base, *qa_ids(base, fact), cfg['ft_steps'])
                    elif knn is not None:
                        knn.append(base, *qa_ids(base, fact))
                    predictions = [predict(base, f, method, memory, head, knn, step,
                                           cfg['max_new_tokens']) for f in facts[:step]] if method != 'no_memory' else zero_predictions[:step]
                    correct = [exact_match(p, f.answer) for p, f in zip(predictions, facts[:step])]
                    history.append(correct)
                    entries = memory.size if method.startswith('aomm') else len(knn.tokens) if knn else 0
                    elapsed = time.perf_counter()-method_start + (zero_elapsed if method == 'no_memory' else 0.)
                    for index, (p, c) in enumerate(zip(predictions, correct)):
                        logs.log(backend=config['base']['backend'], seed=seed, method=method,
                                 injection=step, fact_index=index, correct=c,
                                 mean_recall=float(np.mean(correct)), first_fact_recall=correct[0],
                                 memory_entries=entries, accepted_writes=writes,
                                 elapsed_seconds=elapsed, prediction=p, target=facts[index].answer)
                summary = dict(seed=seed, method=method, final_recall=float(np.mean(history[-1])),
                               mean_curve_recall=float(np.mean([np.mean(h) for h in history])),
                               forgetting_rate=forgetting_rate(history),
                               acquisition_recall=float(np.mean([h[i] for i, h in enumerate(history)])),
                               first_fact_final_recall=history[-1][0], memory_entries=entries,
                               zero_shot_recall=float(np.mean(zero)), elapsed_seconds=elapsed)
                if method == 'aomm':
                    diagnostics.extend(diagnose(base, memory, facts, seed, len(facts),
                                                cfg['max_new_tokens']))
                if method.startswith('aomm'):
                    # used payload vs preallocated capacity; excludes read-module parameters
                    summary['store_bytes_used'] = memory.size * (2*base.dim*4 + 4 + 1)
                    summary['store_bytes_allocated'] = sum(b.numel()*b.element_size() for b in memory.buffers())
                elif knn is not None:
                    summary['store_bytes_used'] = len(knn.tokens)*(base.dim*4 + 8)
                    summary['store_bytes_allocated'] = summary['store_bytes_used']
                else:
                    summary['store_bytes_used'] = summary['store_bytes_allocated'] = 0
                summary['adaptation_parameters'] = sum(p.numel() for p in memory.parameters()) if method.startswith('aomm') else sum(p.numel() for p in head.parameters()) if head is not None else 0
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            final_hash = digest(base)
            if initial_hash != final_hash:
                raise AssertionError('Frozen base parameters CHANGED')
            metadata.append(dict(seed=seed, base_hash_before=initial_hash,
                                 base_hash_after=final_hash, **cal_info))
    finally:
        logs.close()
        calibration_log.close()
        write_log.close()
    with open(output/'summary.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if diagnostics:
        with open(output/'diagnostics.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    manifest = dict(backend=config['base']['backend'], total_seconds=time.perf_counter()-total_start,
                    python=platform.python_version(), torch=torch.__version__,
                    device=cfg.get('device', 'cpu'), git_revision=revision,
                    seeds=metadata, config=config,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder in ['src', 'configs', 'scripts']
                                   for p in sorted(Path(folder).rglob('*'))
                                   if p.is_file() and p.suffix in {'.py', '.yaml', '.sh'}},
                    note='tiny = RANDOM frozen transformer, not GPT-2 evidence; ROME not run')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Total experiment wall clock: {manifest["total_seconds"]:.2f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/exp1_smoke.yaml')
    parser.add_argument('--output', default='results/smoke')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run(config, args.output)


if __name__ == '__main__':
    main()
