# Research decision record — 2026-10-14

## Claim to test, not assume

**Does a learned write policy improve the retention–acquisition–storage trade-off
relative to always-write retrieval, at matched resource budgets?**

Append-only storage prevents deletion/overwriting of accepted records. It does
**not** imply stable predictions. New keys change the attention normalization;
read-module updates can invalidate old embeddings; gating can reject needed
facts; capacity is finite. A unit test demonstrates changed retrieval after an
append with the old record and read parameters untouched. Do not use the title
“Eliminating Catastrophic Interference” as a finding.

## Design choices and alternatives

| Choice | Implemented decision | Alternative and trade-off |
|---|---|---|
| Injection | Four learned soft-prefix vectors, decoded by frozen GPT-2 | Cross-attention inserted in each block has more capacity, but a larger implementation/intervention surface. Use only if oracle-prefix tests reveal a decoding bottleneck. |
| Deployment | Freeze **both** base and read/write parameters after disjoint calibration | Train the retriever on each stream fact; more plastic, but old memory vectors become stale and gradients can interfere. Not implemented as a claimed append-only guarantee. |
| Memory granularity | One final contextual hidden vector per statement | Token-level entries preserve detail at higher memory cost; the kNN comparator deliberately exposes this difference. |
| Gate supervision | Known QA-bearing facts = 1, weather distractors = 0; BCE + sparsity | Counterfactual utility from loss with/without an entry is more general but expensive/noisy. “Was this memory used?” is not an observable binary label. |
| Time | No decay in primary static-fact run; exploratory fixed-decay toggle | Trainable decay exists in `TemporalAttention`, but static facts cannot establish temporal reasoning. A past-time query also needs a causal eligibility mask, not just absolute decay. |
| Decoder metric | Autoregressive, normalized full-city exact match | Teacher-forced next-token accuracy inflates performance when fictional names share tokens. NLL is diagnostic only. |
| Capacity | Explicit error on overflow; no silent eviction/merge | Consolidation violates strict append-only semantics and is out of scope. |

The calibration loss is `QA cross_entropy + gate_loss_weight * BCE + sparsity *
mean(gate_probability)`. QA gradients train key/value/query/prefix projections via
**temporary differentiable support**. Buffer appends are detached. BCE trains the
gate separately: this is a supervised fact/distractor policy, **not** end-to-end
credit assignment through a hard append decision. That limitation must be stated.

Calibration/stream names are disjoint (`Vek...` vs `Zor...`); country-to-city
assignments are independently shuffled. A question contains no answer. Training
episodes see all calibration supports and only calibration answers. Stream
methods see just the current/past facts. Zero-shot recall is checked first, not
assumed to be zero. It does not prove a model has never seen a name.

## Implemented comparison protocol

- Seeds 0–4. Full configured stream: 100 facts; offline smoke: 10.
- Evaluate **every previously injected fact** after each append/update.
- AOMM, no-gate AOMM, and fixed-decay AOMM share calibration weights.
- Token kNN-LM-style: exact squared-L2 neighbors of contextual hidden states;
  interpolate next-token distributions with the frozen LM. Datastore is built
  from teacher-forced QA contexts; each target token has its own record. This is
  a useful, strong control but not the official large-scale implementation.
- Last-layer FT: independent output projection initialized from GPT-2's head,
  optimized on current QA; no replay. Untied so the frozen input embedding is
  never changed. Not full-model fine-tuning and not ROME.
- No-memory: same frozen base. Predictions can be reused because this model
  never changes; original evaluation time is included in its reported time.
- Disclose **unequal warm-up and byte budgets**: AOMM has calibration training,
  FT starts directly at the stream, and kNN stores more contextual token vectors.
  No matched-compute/storage superiority claim is valid yet. CSV summaries report
  tensor payload bytes and adaptive parameter counts (excluding Python overhead).
- Train/test steps and elapsed times live in CSV. Manifest records full config,
  base checksums, versions, source hashes, and total experiment time. Dependency
  installation and analysis/plotting are outside that experiment timer.

### Statistical plan

Primary endpoint for future confirmatory runs: final exact-answer recall.
Secondary: mean recall-curve value, acquisition (correct immediately after each
injection), same-item peak-to-final forgetting, first-item retention, used bytes.
All comparisons pair **seeds**, never facts or overlapping injection steps.
Mean ± sample SD and two-sided paired t-tests are exploratory with five seeds;
Holm correction is across five AOMM-vs-control comparisons. Constant differences
produce an undefined t-statistic; record null, not a manufactured p-value. Plot
bands are ±1 SD, not confidence intervals. The tiny run is not confirmatory.

Peak-to-final forgetting averages previously injected items; the last item has
no later retention measurement. It can look good when nothing was learned: always
read it together with acquisition and absolute accuracy.

## Evidence and official baseline constraints

Verified repository metadata/content through GitHub API on 2026-10-14:

- ROME official code: https://github.com/kmeng01/rome
  (inspected commit `0874014cd9837e4365f3e6f3c71400ef11509e04`). README calls it
  GPU-only and advertises GPT-2 XL / GPT-J. `hparams/ROME/` lists GPT-2 medium,
  large, XL, and GPT-J; **no GPT-2 124M configuration** was listed. Medium/large/XL
  and GPT-J violate this project's 124M cap. A 124M port needs causal-tracing
  validation, moment statistics, edit localization checks, and its own compute
  budget. Do not fabricate a “ROME-style” substitute. ROME necessarily edits its
  own independent base copy; the no-base-update rule applies to AOMM.
- Official kNN-LM code: https://github.com/urvashik/knnlm . The implementation here
  is a small exact-search adaptation, not a reproduction of its reported results.

The supplied Titans/MIRAS identifiers and the other proposed paper names have
**not** been independently verified in this project. Do not cite them as verified
related work; there is no “~10 edits” fact, accepted-conference claim, or novelty
assertion in this repository. Formal bibliography/novelty review remains work.

## Decision gates / best follow-ups

1. **Acquisition before retention:** oracle single-entry prefix decoding must work
   on unseen fictional bindings. `diagnostics.csv` compares no-memory, learned
   retrieval, and oracle retrieval answer NLL, oracle exact match, and attention
   top-1 relevance. Poor oracle decoding means debugging the codec comes first.
2. **Separate retrieval from decoding:** add raw-text retrieved-prefix and exact
   lexical-lookup controls, single-entry reconstruction episodes, and longer
   token-valued memories. Compare to the existing kNN control. Do not select
   hyperparameters by final evaluation seed performance.
3. **Run genuine frozen GPT-2:** obtain local weights in an unrestricted machine,
   validate the integration test, pilot 10–20 facts to measure acquisition and
   cost, then freeze settings before five independent 100-fact seeds. No credible
   <=20 GPU-hour promise exists yet.
4. **Make the learned gate meaningful:** vary relevance and distractor ratio at
   fixed byte budgets. Compare random-write, always-write, and supervised-write;
   use new paraphrases and name generators. Task-type labels alone are too easy
   to establish learned utility. Measure false rejection/acceptance separately.
5. **Strengthen novelty and controls:** matched warm-up/budget comparisons,
   paraphrased questions, longer random answers, similar keys, repeated subjects,
   an online-read-adaptation ablation, and a validated real editor baseline.
   Distinguish learning a binding from memorizing a QA string.
6. **Only then consider temporal QA:** repeated subjects, updates and historical
   queries, exclusion of future facts, timestamp-vs-latest-vs-no-time controls.
   Static-fact time decay is not evidence for G4. Expand to continual learning/TTT
   only after meaningful GPT-2 acquisition and credible comparisons, not merely
   a p-value. Experiments 2–4 are intentionally not built in this iteration.

These checks are falsifiable milestones, not an acceptance/significance target.
A well-supported negative result is preferable to tuning until p < .05.
