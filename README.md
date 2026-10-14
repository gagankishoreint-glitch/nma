# Text-Only Native-Memory Agent — active iteration

A separate experiment now lives in `agent.py`, `memory/`, `llm/`, and `prompts/`.
It keeps the LLM text-only and compares weight-only neural slot memory against
no memory and bounded raw context. It **does not** search audit logs or use a
vector index for the native condition. See [MISSION.md](MISSION.md),
[PROGRESS.md](PROGRESS.md), and [technical decisions](docs/NATIVE_AGENT.md).

```bash
# Offline plumbing ONLY: scripted transport + deterministic hash encoder
PYTHON=.venv/bin/python bash scripts/run_agent.sh smoke

# Optional genuine-model dependencies (not installed in this sandbox)
.venv/bin/pip install -r requirements-agent.txt

# Real mode: requires local files, passes stability gate BEFORE memory tasks
PYTHON=.venv/bin/python bash scripts/run_agent.sh real \
  --model /local/path/chat-model.gguf --encoder /local/path/all-MiniLM-L6-v2 \
  --n-ctx 1024 --n-gpu-layers 0

# Opt-in real behavioral tests; otherwise explicitly skipped
AGENT_GGUF_PATH=/local/path/chat-model.gguf .venv/bin/python -m pytest -q tests/test_stability.py
```

This sandbox has no GPU or local model weights. **Real LLM stability is not run**;
`results/stability_tests.json` honestly records four `not_run` tests, not four
passes. `results/agent_log.csv` and `PROCEDURES.md` are explicitly scripted fixture
artifacts. The fixture context baseline recalls the address; the cold neural
codec abstains. These are engineering checks, not evidence about a 4B LLM/Titans.

- `results/agent_smoke/`: scripted run, summaries, parameter hashes, environment.
- `results/agent_real/`: blocked genuine-model preflight; no fabricated task logs.
- `results/agent_manifest.json`: last completed agent run's provenance.
- `results/agent_attempt_manifest.json`: latest attempt, including failures.
- `results/stability_tests.json`: latest attempt's behavioral-test status.
- `scripts/run_agent.sh` prints the mission on every invocation. Re-running a mode
  replaces that mode's generated artifacts; use `--output` to preserve other runs.

Default real mode is CPU. GPU offload needs working `nvidia-smi`, a CUDA-enabled
llama.cpp build and target-hardware validation. The sampled 3.5-GiB stop guard is
**not** a hard allocation cap. No LoRA, hidden-state extraction or production tool
execution is implemented. The supplied Titans API, persona guarantees, and fixed
VRAM estimates are not assumed true; see the decision record for corrections.

---

# Append-Only Learned Memory

A research prototype testing whether a supervised write gate and append-only
retrieval can improve memory retention on a **frozen GPT-2 (124M)**. Stored records
are never overwritten, but prediction stability is **not guaranteed**: retrieval
competition, rejected writes, and decoding errors remain. This repository makes
those failure modes measurable; it does not claim to solve continual learning.

## Status

**Five-seed CPU smoke test executed, not a GPT-2 research result.** The offline
base is a randomly initialized 73,824-parameter transformer. AOMM failed to acquire
answers (0% final exact recall); the kNN-style comparator achieved 88% ± 13%.
This does not support the proposed superiority claim. See [archived progress](docs/APPEND_ONLY_PROGRESS.md)
and [research decisions / next steps](docs/RESEARCH_PLAN.md).

No GPT-2 weights are available in this sandbox, no GPU is present, and outbound
access to Hugging Face is unavailable. The code never silently substitutes tiny
weights when a GPT-2 run is requested. Real ROME is **not run** or impersonated.

## Install (Python 3.11)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

`requirements-lock.txt` captures the exact installed smoke-run environment. Its
PyPI PyTorch wheel pulls CUDA libraries even though this run used CPU; it needs
several GB of disk. On another machine a compatible official CPU-only PyTorch
wheel may be used, but record the changed dependency environment. No wandb login,
dataset download, or remote model is required for smoke mode.

## Run Experiment 1

```bash
# Default = explicitly labeled offline pipeline validation (about half a minute here)
bash scripts/run_exp1.sh
# Equivalent without activating the venv:
PYTHON=.venv/bin/python bash scripts/run_exp1.sh smoke

# Research configuration: genuine locally cached GPT-2 required; NOT executed here
bash scripts/run_exp1.sh gpt2
```

For GPT-2, set `base.model_name` in
`configs/experiments/exp1_sequential.yaml` to a local Hugging Face-format GPT-2
checkpoint directory (with tokenizer files), or cache `gpt2` on an unrestricted
machine beforehand. `local_files_only: true` makes missing weights fail clearly.
Set `experiment.device: cuda` only on a CUDA machine. Enable the integration test:

```bash
GPT2_LOCAL_PATH=/path/to/gpt2 python -m pytest -q tests/test_base.py
```

Custom configs/output directories:

```bash
python -m src.train.train_aomm --config configs/experiments/exp1_smoke.yaml --output results/my_run
python -m src.eval.analyze --results results/my_run
```

Modes use separate `results/smoke/` and `results/gpt2/` directories. The shell
script copies the **last successful run** into the requested top-level artifacts:

- `results/exp1_log.csv`: triangular per-fact/per-step recall, predictions, seed,
  method, elapsed time, accepted writes, entries.
- `results/figures/exp1_recall.png`: six curves, mean ± sample SD over seeds.
- `results/manifest.json`: backend, exact config, source hashes, base checksums,
  versions and wall-clock time. Read this before interpreting the figure.
- `results/statistics.json`: paired seed-level tests, Holm correction; undefined
  tests are explicitly null. These smoke statistics are exploratory only.
- Per-mode extras: `calibration.csv`, `summary.csv`, `aggregate.csv`,
  `diagnostics.csv`, zero-shot predictions for each seed, and `config.json`.
- `notebooks/analysis.ipynb`: reruns analysis and displays the saved figure.

Running a mode again replaces its artifacts; choose a custom directory to retain
runs. CSV is always used, not a cloud logging dependency. Experiment elapsed time
includes calibration and diagnostics, but not installation or plotting.

## Architecture and training

1. Frozen GPT-2 yields contextual hidden states. Offline debug mode uses a small
   GPT-2-shaped random model and character tokenizer, **not** pretrained GPT-2.
2. AOMM contains learned key/value projections, multihead query attention, a
   four-vector soft-prefix decoder, and a sigmoid MLP write gate.
3. Disjoint calibration facts train QA via a differentiable temporary support
   store. Gate BCE labels distinguish QA facts from weather distractors, with a
   sparsity penalty. This is **supervised task relevance**, not general utility
   learning or differentiation through hard store writes.
4. All AOMM parameters are then frozen. Each stream fact (plus a distractor) is
   gate-filtered into detached preallocated buffers. No overwrite, eviction, or
   base update is allowed; overflow is an explicit error.
5. Retrieval becomes soft embeddings prepended to the query's token embeddings.
   Gradients pass through the frozen model into soft prefixes during calibration.
6. Evaluate generated complete city names after every injection, not just the
   first answer token. Names are synthetic and assignments seed-shuffled; zero-
   shot recall is measured before exposure.

Controls: no memory; independent untied last-output-head fine-tuning; exact
squared-L2 token kNN interpolation; gate-off ablation; fixed-decay time toggle.
The primary static-fact AOMM uses no temporal decay. `TemporalAttention` supports
a learned nonnegative decay and optional future mask, but the static experiment's
time toggle uses the **same weights and an untrained fixed decay**, not learned
temporal reasoning. **Do not call any control ROME/MEMIT.**

Baselines have unequal warm-up and store-byte budgets; counters are reported but
not matched yet. Official ROME has no provided GPT-2 124M configuration in the
inspected repo. These are prototype controls, not publication-ready comparisons.

## Layout / scope

```text
configs/             model examples and smoke/GPT-2 experiment settings
src/models/          frozen LM, append-only store, temporal attention
src/data/            fictional facts and QA
src/train/           calibration, sequential evaluation and honest controls
src/eval/            recall/forgetting metrics, plots, paired statistics
src/utils/           deterministic seeds, streaming CSV logging
scripts/             one-command Experiment 1 pipeline
notebooks/           analysis notebook
results/             executed smoke artifacts + provenance
tests/              invariants, gradient checks, deterministic pipeline test
docs/               research choices, limitations, evidence and follow-ups
```

Experiments 2–4, consolidation, hardware co-design, capacity solutions, and models
larger than GPT-2 124M are intentionally deferred. There is no preprint-ready
paper or asserted novelty yet. Do not cite the supplied unverified reading list.
