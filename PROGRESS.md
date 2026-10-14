# Progress — text-only native-memory agent — 2026-10-14

## Bottom line

Implemented the new agent as a separate experiment. **No genuine LLM stability or
4GB-VRAM claim has been validated.** The available offline plumbing test succeeds
with bounded raw context but the cold native neural codec abstains. That is not
an empirical result about Titans or a 4B LLM.

The previous GPT-2 append-only experiment and its original artifacts are preserved;
its report is now [docs/APPEND_ONLY_PROGRESS.md](docs/APPEND_ONLY_PROGRESS.md).

## Built

- `MISSION.md`, printed first by `scripts/run_agent.sh` in both modes.
- `memory/titans_memory.py`: single-step online associative MLP, frozen reads,
  exact initial-state reset, loss logging, no retained gradient graphs. **Custom
  Titans-inspired prototype, not the published Titans algorithm/library.**
- A bounded learned UTF-8 decoder, so private values really must come from
  neural parameters; no episode-vector search or hidden answer-codebook lookup.
- `memory/encoder_bridge.py`: independent CPU/local sentence encoder; explicit
  deterministic hash fixture for offline testing. No LLM hidden states.
- Goal/slot routing, secondary generic procedures, isolated JSON extraction with
  success/schema/domain checks and markdown-fence handling.
- Text-only llama.cpp wrapper: fresh two-message calls, no adapters, hidden-state
  hooks, soft prompts or LLM weight updates. Main policy is unchanged across
  calls. Dynamic memory/procedure text is JSON **user data**, not interpolated
  into the system role. The supplied template is retained in the prompt file.
- Strict simulated tool executor: generated actions do not count as execution;
  only verified outcomes may write facts or authorize procedure extraction.
- Four real-model stability probes, including twenty repeated contract calls;
  failed/unrun real stability prevents the real memory benchmark.
- No-memory, N-turn raw-context, neural-only, full frozen-memory and full-system
  conditions; seeded names/addresses, exact address-delivery scoring, per-step
  surprise, recalled data, procedure, action, outcome, resource logs, and hashes.

## What actually ran

```bash
PYTHON=.venv/bin/python bash scripts/run_agent.sh smoke
.venv/bin/python -m pytest -q
```

**Scripted transport + hash encoder**, five worlds/seeds, three logical sessions
within one process per condition, N=4 raw turns, at most two actions/session.
No actual restaurant booking, message delivery, real-model generation, MiniLM
encoding, or cross-process recovery was tested.

| Condition | Session 1 search | Session 2 booking | Session 3 exact address |
|---|---:|---:|---:|
| No memory | 5/5 | 0/5 | 0/5 |
| Context window | 5/5 | 5/5 | 5/5 |
| Neural only | 5/5 | 0/5 | 0/5 |
| Full, frozen memory | 5/5 | 0/5 | 0/5 |
| Full, online memory | 5/5 | 0/5 | 0/5 |

These outcomes validate harness/data-flow behavior, **not LLM performance**. The
scripted fixture is a transparent finite-state context consumer. It has no access
to hidden world truth. The neural condition returns low-confidence/unknown values;
it cannot yet acquire name/address strings with this unpretrained, one-step codec.
Its failure must not be used to reject Titans or claim context memory superiority
in a real agent. One-step padding-dominated BCE and the confidence threshold are
major confounds that need calibration/codec work before a scientific comparison.

- 75 step rows written; all no-memory/context conditions have zero neural updates.
- Full and neural-only each perform **three updates**: one per outcome, with the
  two Session-1 slots batched into a single step. Their checksums change.
- Frozen-memory checksums remain unchanged. Read/reset invariants pass unit tests.
- The generic booking procedure from a separate successful synthetic warm-up is
  present in every full-system Session-2 prompt. It contains no private address;
  it cannot solve missing factual recall on its own.
- `PROCEDURES.md` was generated, prominently labeled **scripted fixture**, and
  exports the last full-condition seed's procedures. It is not a real LLM
  extraction-quality result.

## Resources and environment

- Actual host RAM: **3,939.9 MiB (~3.85 GiB)**, not the requested 16GB target.
- GPU / measured VRAM: **unavailable / null**. There is no GPU in this sandbox;
  do not interpret null as a measured zero or claim a 4GB-model fit.
- Scripted-run process peak RSS: **372.3 MiB** (includes runtime/import overhead).
- Post-import runner time: **0.120 seconds**. This excludes interpreter startup,
  imports, installation and any real LLM/encoder loading. No GPU-time estimate.
- Neural module: **90,352 parameters**, 361,408 FP32 parameter bytes, plus the
  reset snapshot/temporary gradients/runtime. Even its 384→32→64 MLP alone has
  14,432 parameters; the supplied “under 10K” estimate was incorrect.
- PyTorch 2.2.2 is installed. `llama-cpp-python`, `sentence-transformers` and the
  optional Titans libraries are absent; no GGUF/encoder weights are available.
  Hugging Face is outside this sandbox's outbound allowlist.

Real-mode preflight was also executed and correctly exited **2**, requiring local
GGUF/encoder paths. Its environment, blocked status and source hashes are recorded
in `results/agent_real/`. No real-model task CSV was fabricated.

Real mode defaults to CPU, n_ctx=1024. Optional GPU offload requires `nvidia-smi`
telemetry. Device-total sampling guards at 3,584 MiB, but cannot cap an allocation
or guarantee observing every transient peak. It accounts for native llama.cpp
memory unlike torch-only VRAM stats, and conservatively includes other workloads.

## Tests and stability status

**29 passed, 5 skipped.** Skips: four genuine-GGUF behavioral tests and the earlier
GPT-2 integration test. Passing structural/mocked tests is not behavioral evidence.

`results/stability_tests.json` correctly says:

- tone consistency: **not_run**
- memory/persona injection: **not_run**
- instruction injection: **not_run**
- goal precedence: **not_run**
- `all_passed: null`

A scripted backend is forbidden from certifying these tests. Providing an
“all four passed” file without a model would fabricate evidence. Also, passing
these narrow tests later would not prove general persona stability, privacy,
robust prompt-injection resistance, or safety. “Tone” is an explicitly limited
JSON-action consistency proxy, not a comprehensive linguistic classifier.

Tests cover actual parameter updates, read immutability, exact reset, UTF-8
round-trip/abstention, atomic oversized-write errors, one batch/one gradient step,
no audit lookup, text-only transport/reset isolation, extraction validation and
success gating, constrained tools, context baselines, frozen-memory ablation,
real-benchmark stability gating, and the sampled VRAM guard.

## Best next steps

1. **Obtain local GGUF + encoder weights on suitable hardware.** Install optional
   dependencies, run real stability first, and retain failures/raw outputs. Start
   with a small instruct GGUF rather than assume Qwen3-4B will fit or follow JSON.
2. **Fix isolated neural acquisition before scaling.** Compare calibrated/pretrained
   content decoding and a normalized-delta fast-weight control on held-out slot
   values. Never make the native condition pass by covertly looking up audit text.
3. Only after acquisition works: run the same real-model conditions, including
   frozen-memory/neural-only attribution ablations and byte/compute accounting.
4. Measure the actual target GPU. Reduce context/offload or model size when the
   guard fails; a sampled guard is not a hard capacity guarantee.
5. Expand paraphrases, multiple entities and sessions only after the above. No
   MemGuide reproduction, general procedure/privacy solution, published-Titans
   comparison, state-of-the-art result or continual-learning claim is established.

Technical decisions, source/API verification and explicit departures from the
supplied sketches: [docs/NATIVE_AGENT.md](docs/NATIVE_AGENT.md).

## Artifacts

- `results/agent_smoke/{agent_log.csv,summary.csv,memory_hashes.json,manifest.json}`
- `results/agent_real/{environment.json,manifest.json,stability_tests.json}`
- `results/agent_log.csv` + `results/agent_manifest.json`: last **completed** run
- `results/agent_attempt_manifest.json`: latest attempt, including failed runs
- `results/stability_tests.json`: latest attempt's behavioral-test status
- `PROCEDURES.md`: last completed run's explicitly labeled procedure export

Re-running a mode replaces its generated outputs. Use `--output` for separate runs.
