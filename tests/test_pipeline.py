import json
from pathlib import Path
import pandas as pd
import yaml
from src.train.train_aomm import run
from src.eval.analyze import analyze


def test_end_to_end_logs_and_determinism(tmp_path):
    config = yaml.safe_load(Path('configs/experiments/exp1_smoke.yaml').read_text())
    config['experiment'].update(seeds=[0], num_facts=2, calibration_facts=3,
                                calibration_steps=2, ft_steps=1, max_new_tokens=3)
    output = tmp_path/'first'
    run(config, output)
    analyze(output)
    rows = pd.read_csv(output/'exp1_log.csv')
    assert len(rows) == 6 * 3
    assert (rows.fact_index < rows.injection).all()
    assert rows.groupby(['method', 'injection']).mean_recall.nunique().eq(1).all()
    assert rows.elapsed_seconds.ge(0).all()
    summary = pd.read_csv(output/'summary.csv')
    assert len(summary) == 6
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['seeds'][0]['base_hash_before'] == manifest['seeds'][0]['base_hash_after']
    assert (output/'figures/exp1_recall.png').stat().st_size > 100
    assert len(json.loads((output/'statistics.json').read_text())) == 5
    run(config, tmp_path/'second')
    again = pd.read_csv(tmp_path/'second/exp1_log.csv')
    pd.testing.assert_frame_equal(rows.drop(columns=['elapsed_seconds']),
                                  again.drop(columns=['elapsed_seconds']))
