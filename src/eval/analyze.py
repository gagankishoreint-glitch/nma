"""Seed-level statistics; per-fact rows are NOT independent replicates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(root):
    root = Path(root)
    log = pd.read_csv(root/'exp1_log.csv')
    summary = pd.read_csv(root/'summary.csv')
    manifest = json.loads((root/'manifest.json').read_text())
    curves = log.drop_duplicates(['seed', 'method', 'injection'])
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, data in curves.groupby('method'):
        stats = data.groupby('injection').mean_recall.agg(['mean', 'std'])
        x = stats.index.to_numpy()
        mean, std = stats['mean'].to_numpy(), stats['std'].fillna(0).to_numpy()
        ax.plot(x, mean, marker='.', label=method)
        ax.fill_between(x, np.maximum(0, mean-std), np.minimum(1, mean+std), alpha=.12)
    tiny = manifest['backend'] == 'tiny'
    ax.set(title=('OFFLINE SMOKE — random tiny base; NOT GPT-2 evidence' if tiny else
                  'Frozen GPT-2 fictional fact recall (ROME not evaluated)'),
           xlabel='Facts injected', ylabel='Exact answer recall (mean ± 1 SD)', ylim=(-.03, 1.03))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    (root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures/exp1_recall.png', dpi=160)
    plt.close(fig)
    # Predeclared primary endpoint: final recall. Secondary: mean curve recall.
    tests = []
    if 'aomm' in set(summary.method):
        for comparator in sorted(set(summary.method)-{'aomm'}):
            left = summary[summary.method == 'aomm'].set_index('seed').final_recall
            right = summary[summary.method == comparator].set_index('seed').final_recall
            paired = pd.concat([left.rename('aomm'), right.rename('control')], axis=1).dropna()
            delta = (paired.aomm - paired.control).to_numpy()
            status = 'ok'
            p = None
            if len(delta) < 2 or np.std(delta, ddof=1) == 0:
                status = 'undefined: too few seeds or zero paired-difference variance'
            else:
                p = float(ttest_rel(paired.aomm, paired.control).pvalue)
            tests.append(dict(comparator=comparator, n_seeds=len(delta),
                              mean_difference=float(delta.mean()), paired_t_p=p,
                              status=status, exploratory=tiny))
    # Holm family-wise correction, with undefined tests counted in the family.
    valid = sorted([t for t in tests if t['paired_t_p'] is not None], key=lambda t:t['paired_t_p'])
    running = 0.
    for rank, test in enumerate(valid):
        running = max(running, min(1., (len(tests)-rank)*test['paired_t_p']))
        test['holm_p'] = running
    (root/'statistics.json').write_text(json.dumps(tests, indent=2, allow_nan=False))
    grouped = summary.groupby('method')[['final_recall', 'acquisition_recall', 'forgetting_rate',
                                          'mean_curve_recall', 'memory_entries']].agg(['mean', 'std'])
    grouped.to_csv(root/'aggregate.csv')
    print(grouped.to_string())
    print('Paired tests:', json.dumps(tests, indent=2))
    print('Flat zero recall is failure to acquire, NOT successful retention.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/smoke')
    analyze(parser.parse_args().results)
