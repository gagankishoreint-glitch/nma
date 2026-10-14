# Progress — 2026-10-14

## Summary

Built and executed the narrow Experiment 1 prototype. **The smoke results do not
support AOMM superiority.** This is a working research starting point, not a
publishable result or a demonstration that append-only memory eliminates
interference.

## What is implemented

- Genuine GPT-2 loading path with frozen parameters and differentiable soft-prefix
  injection; explicit failure if weights are missing.
- Offline 73,824-parameter random transformer for pipeline validation only.
- Learned key/value/query projections, four-vector prefix codec, supervised
  sigmoid gate, preallocated detached append-only buffers, overflow errors.
- Disjoint calibration/stream fictional facts, shuffled bindings, weather
  distractors, multi-token answer generation and pre-exposure zero-shot checks.
- No-memory, independent last-output-head FT, and exact token kNN-style controls;
  gate-off and fixed-decay ablations. **None is labeled ROME.**
- Five-seed CSVs, full configs/source hashes, before/after base checksums, wall
  time, seed-level paired tests/Holm correction, six-curve figure, and notebook.
- Oracle single-entry retrieval diagnostics and per-write gate decisions.

## Executed run

```bash
PYTHON=.venv/bin/python bash scripts/run_exp1.sh smoke
```

CPU, seeds 0–4, **10 facts per seed**, 24 calibration facts, 120 calibration steps,
12 current-fact FT steps, and one weather distractor per fact. The base is random,
not a pretrained GPT-2. Total experiment wall clock: **31.18 seconds**, including
calibration and oracle diagnostics; package installation and analysis/plotting
are excluded. This timing must not be extrapolated to GPT-2.

Final exact-answer recall (mean ± sample SD over five seeds):

| Method | Final recall | Acquisition recall | Mean peak-to-final forgetting |
|---|---:|---:|---:|
| AOMM | 0.0% ± 0.0% | 0.0% | 0.0 points* |
| AOMM, no gate | 0.0% ± 0.0% | 0.0% | 0.0 points* |
| AOMM, fixed time decay | 0.0% ± 0.0% | 0.0% | 0.0 points* |
| Token kNN-style | 88.0% ± 13.0% | 94.0% | 8.9 points |
| Last-output-head FT | 10.0% ± 0.0% | 14.0% | 24.4 points |
| No memory | 0.0% ± 0.0% | 0.0% | 0.0 points* |

*Zero forgetting without acquisition is **not** retained knowledge. FT's mean
recall curve falls from 80% after the first injection to 10% at the end, and its
same-item forgetting is positive. kNN retains much better but also has positive
same-item forgetting: adding records can change nearest-neighbor decisions.

The exploratory paired final-recall test favors kNN over AOMM (AOMM−kNN = −0.88,
raw p ≈ 0.000112; Holm p ≈ 0.000562). This is a tiny/random-model diagnostic, **not
research evidence about GPT-2**, and cannot justify expanding the scope. Constant
paired differences for other comparisons have undefined t-tests and are logged
as null rather than invented significance.

### Diagnostic findings

- The gate accepted all 50 stream facts and rejected all 50 weather distractors.
  This only validates an easy supervised relevance classifier, not learned
  counterfactual memory utility.
- AOMM stores 10 entries vs 20 for always-write; used tensor payload is 3,890 vs
  7,780 bytes. Both preallocate 7,788 bytes, including the size buffer. **There is
  no allocated-memory saving** in this implementation. kNN has 110 contextual
  token entries / 22,000 payload bytes; Python object overhead is not counted.
- Oracle single-entry prefixes still score **0% exact recall**. Mean answer NLL:
  no memory 4.5567, learned retrieval 4.5747, oracle retrieval 4.5747. The attention
  top-1 relevant-entry rate is 10% (chance among ten accepted facts). Both the
  content/injection path and retrieval need work; gate rejection is not causing
  this run's failure. Do not conclude that every prefix architecture must fail.
- Frozen-base checksums match before/after all methods for every seed.

## Validation / bugs

`python -m pytest -q`: **12 passed, 1 skipped**. Tests cover append immutability,
overflow atomicity, reset, detached buffers, checkpoint round-trip, gate rejection,
temporal masking, prefix/projection gradients, frozen weights, independent FT,
no-future-fact logs, and deterministic end-to-end output.

The skipped integration test requires genuine GPT-2 weights. A local-only GPT-2
load was attempted and failed because no checkpoint is cached. This sandbox has
no CUDA device and cannot access Hugging Face. Details: `results/environment.json`.
There was no silent model substitution.

An all-future temporal attention row could have produced NaNs. The implementation
now masks all-ineligible rows before softmax, returning zero context. The exact
installed dependency set is recorded in `requirements-lock.txt`; `pip check`
reported no broken requirements.

## Still missing / fairness limitations

- **GPT-2 experiment not run.** The full 100-fact/five-seed config is a plan, not
  a measured result. No 20-GPU-hour guarantee.
- **Real ROME not run.** The official repo inspected has no GPT-2 124M config and
  describes GPU-only operation. A validated 124M port is needed under this cap.
  No fake ROME, MEMIT, full-model FT, EWC or TTT results are included.
- AOMM receives extra calibration; kNN uses teacher-forced QA contexts at write
  time while AOMM encodes statement vectors. FT also receives current QA
  supervision. Warm-up, write-time encoding and bytes are **not matched**. No
  causal superiority claim is justified by these controls.
- Static, template-generated QA can reward string memorization. No paraphrase,
  long-answer, adversarial-similarity, or realistic utility generalization test yet.
- Time toggle is fixed/untrained; it does not test historical QA or G4.
- No verified novelty review, formal paper, or Experiments 2–4. Those are deferred.

## Best next move

**Fix acquisition before scaling retention experiments.** On a machine with local
GPT-2 weights, first verify an oracle single-fact prefix can reconstruct unseen
answers. Compare raw-text retrieved prefixes and longer/token-valued memory to
isolate information bottlenecks. Only after that works, train retrieval with
held-out binding episodes and match write-time information, calibration and byte
budgets against kNN. Then preregister/freeze the protocol for five 100-fact seeds.
Use an explicit validated editor baseline before making model-editing claims.

The sharper research question is whether learned writes improve the
**retention–acquisition–storage trade-off**, not whether immutable storage proves
forgetting has been eliminated. Detailed alternatives and milestones are in
[`docs/RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

## Artifacts

- [`results/figures/exp1_recall.png`](../results/figures/exp1_recall.png)
- [`results/exp1_log.csv`](../results/exp1_log.csv)
- [`results/manifest.json`](../results/manifest.json)
- [`results/statistics.json`](../results/statistics.json)
- `results/smoke/{summary,calibration,diagnostics,writes,aggregate}.csv`
- [`notebooks/analysis.ipynb`](../notebooks/analysis.ipynb)
