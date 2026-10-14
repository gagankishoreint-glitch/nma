# Text-only native-memory agent: implementation decisions

## Two experiments, not a retroactive rewrite

`agent.py`, `memory/`, `llm/`, and `prompts/` implement the new text-only experiment.
The previous `src/models/aomm.py` / GPT-2 soft-prefix experiment is retained for
reproducibility, but **is not imported or invoked by the agent**. Its results have
not been altered. Archived report: [APPEND_ONLY_PROGRESS.md](APPEND_ONLY_PROGRESS.md).

## What was corrected in the proposed design

1. **A cosine search over stored episode embeddings is external retrieval.** An
   MLP's surprise multiplied by cosine similarity does not make that retrieval
   weight-only memory, and does not decode the MLP's representation into an address.
   This implementation does not search its audit logs, retain private slot values
   in a Python lookup table, or maintain a nearest-neighbor codebook.
2. **Add a real content decoder.** The neural condition maps a public slot-query
   embedding to a bounded UTF-8 value using an MLP and parameterized byte-bit
   decoder. The fixed codec is not a list of previously seen answers. It uses a
   two-byte length header, at most 96 content bytes, and zero padding. Unknown,
   invalid, low-confidence or oversized values cause abstention/error, not a guess
   or silent truncation. This is a deliberately simple, unpretrained codec, not
   a reproduction of Titans. Its cold-start acquisition weakness is a confound,
   not a result about the published architecture.
3. **Supervision is explicit.** The simulator yields structured name/address
   fields. All fields in one tool outcome are batched for **one SGD step** on
   bitwise prediction BCE. Other outcomes receive one embedding-reconstruction
   step. This is supervised associative memory; reconstruction of a sentence
   embedding alone is not advertised as factual text recall. The two loss types
   have different units and must not be directly compared as calibrated surprise.
4. **Read does not update.** Inner-loop parameters are temporarily differentiable;
   `autograd.grad` produces one update, then gradients/flags are cleared. No graph,
   optimizer-moment buffer, or replay batch is retained. Reset restores the exact
   construction-time parameters of the MLP **and both decoders**, not a newly
   sampled network. No explicit adaptive learning-rate or momentum mechanism is
   claimed. Optional weight decay is ordinary SGD weight decay, default zero.
5. **Parameter claims are measured.** With 384→32→64 the MLP itself has 14,432
   parameters, already more than the claimed 10K. Including reconstruction and
   UTF-8 decoders the prototype has **90,352 parameters / 361,408 FP32 parameter
   bytes**. There is also a reset snapshot, gradients during updates, and a large
   PyTorch runtime. Parameter bytes are not a measured VRAM footprint.
6. **Preserve the trust boundary.** `prompts/agent_system.txt` preserves the supplied
   template. Runtime sends its fixed policy portion plus a fixed action contract
   as system text. Goal, procedure, observations, memories, and optional history
   are serialized in a **separate JSON user message**, not interpolated into the
   system role. This is an intentional departure from the unsafe placement of
   retrieved material at system privilege. All inputs to the LLM are plain text;
   no soft prompts, hidden-state hooks, weights, embeddings, or LoRA adapters.
7. **No persona theorem.** Text retrieval also changes model activations, and
   prompt instructions/delimiters cannot guarantee instruction isolation. Likewise,
   soft-prefix injection is not known to inevitably cause persona drift. The
   implementation avoids it as a scope decision, not as a universal empirical law.
   Temperature zero is not a cross-hardware determinism guarantee.
8. **Only actual outcomes decide completion.** `[DONE]` has no power. JSON actions
   are schema/allowlist validated and run in a synthetic executor. Model actions
   are never themselves recorded as true facts. The executor must confirm success
   before extraction. There are no real bookings, messages, payments or shell tools.

## Native read/write path

- Encoder: independent frozen MiniLM (CPU/local files only) for a real run; stable
  token hashing for the explicitly scripted plumbing run. Neither uses the LLM's
  hidden states. Sentence-encoder size is model-specific, not a fixed VRAM claim.
- Public schema: restaurant name, restaurant address, booking status. The schema
  contains questions, **not values**. Only requested/missing slots are queried.
- Neural primary memory: MLP parameters + predictor/text decoder parameters.
  No index or record lookup; no retained external facts at read time.
- Secondary audit: CSV containing actual observations/actions/surprises. The
  orchestrator cannot read it back. Unit tests clear/replace the audit sink and
  verify unchanged read results.
- Secondary procedures: strict, domain-whitelisted generic plans. Known episode
  values and simple digit/email patterns are rejected. No private address may be
  used as a procedural-memory shortcut. This validation is adequate only for this
  synthetic fixture, **not** a general privacy/injection detector. Plans remain
  untrusted advisory data and never execute tools directly.

This is a **separate online neural memory**, not a claim that llama.cpp's frozen
LLM becomes a native-memory architecture. It is a bounded single-selected-
restaurant prototype, not arbitrary multi-entity semantic recall.

## Comparison and scoring

Five independently initialized memory/world seeds, same goals and synthetic world
per condition. Each seed runs the following in independent agent instances:

| Condition | Cross-session information |
|---|---|
| `no_memory` | Current goal/observation only; no neural module, history or procedure |
| `context_window` | Last N raw action/tool-observation turns; default N=4; no neural module/procedure |
| `neural_only` | Online neural weights; no raw history or procedures |
| `full_frozen_memory` | Generic procedures + identically initialized neural model; no weight updates |
| `full` | Online neural weights + generic procedures |

A separate successful simulator search/booking supplies a warm-up log for isolated
procedure extraction. It uses a different restaurant from evaluation. Generic
procedures are not shared across evaluation conditions/seeds after initialization.
The same LLM is loaded once, but **each call resets the public llama.cpp inference
state, disables its cache, and supplies two fresh messages**. Extraction gets a
separate system/user call, not the main-loop conversation. Cached model weights
are naturally shared; no claim of a mathematically pure/deterministic function.

The synthetic sequence:

1. Find a restaurant in Beijing: `find_restaurant(city=Beijing)` reveals a seeded
   fictional name and address. Exact strings are not put into later goals.
2. Book the restaurant found: only `book_table(name=<exact selected name>)`
   succeeds. Missing information should lead to `ask_user`, which ends this
   synthetic session unsuccessfully without leaking the correct value.
3. Send its address: only `send_address(address=<exact selected address>)` succeeds.
   Session-3 recall is this exact address delivery, not substring/token overlap.

An intent/slot table is the transparent task-specific router. This is **not** a
trained MemGuide reproduction or a chain-of-thought/LoRA slot reasoner. Native
weights persist across logical sessions within one process; restart recovery is
not evaluated. Histories have an explicit N-turn bound. Context too large for the
text/token budget produces a logged transport failure, not silent truncation.
No acquisition→retention conclusion is valid when the neural condition abstains.
The scripted transport reads only delivered messages; it is an evidence-consuming
finite-state test double, **not an LLM**. Its results cannot support model claims.

## Stability gate

Real mode performs 23 independent calls: 20 repeated action-contract probes, plus
persona injection, instruction override, and stale-goal contradiction probes.
All require the same narrow Beijing-search action under the actual agent policy.
“Tone” is an explicitly limited action-format consistency proxy, not an undefined
classifier or a proof of general style stability. The probes use the action task,
not the arithmetic prompt from the sample, to exercise the deployed contract.
Outputs and failures are recorded. Failed/blocked probes stop the real memory
experiment. The public benchmark entry point also checks for a passed real-model
report. A scripted transport is prohibited from certifying these probes.

No local GGUF here → all four behavioral tests are **not_run/null**, and the four
opt-in pytest integration tests are skipped. Structural/mocked tests prove code
properties, not behavior of an absent model. Supplying “all four passed” here
would be fabricated.

## Resource measurement

- Actual sandbox: ~3.85 GiB host RAM, **not** the requested 16GB machine; no GPU,
  no GGUF, no MiniLM checkpoint, no `llama-cpp-python` / `sentence-transformers`.
- Default real transport: CPU (`n_gpu_layers=0`, KV offload disabled), n_ctx=1024.
  Start with a small instruct GGUF and compatible embedded chat template. 4B Q4
  feasibility depends on weights, KV/cache/backend/workspace; it is not guaranteed.
- CUDA offload requires working `nvidia-smi` telemetry. A sampling thread records
  device-total used MiB (including native llama.cpp allocations and other processes)
  and stops work at the 3,584 MiB guard. It is conservative, not process-attributed,
  and can miss transient peaks. It is **not a hard GPU allocation cap** or a proven
  <=4GB result. On failure, reduce offload/context or use CPU before retrying.
- Process peak RSS uses Linux `getrusage`. It includes import/runtime overhead and
  is a shared-process lifetime maximum, not a fair per-method footprint comparison.
- Every stage and step is profiled. Unavailable GPU telemetry is **null**, not a
  fabricated zero. PyTorch's CUDA allocator statistics would miss llama.cpp usage,
  so they are not used as total VRAM. Real model/encoder hashes are recorded.

## Library/API verification

The inspected **unofficial** repository https://github.com/lucidrains/titans-pytorch
at commit `1d40c445fa1794fb28582c721786482b5b683e47` (2026-10-14) documents:

```python
# documented upstream example, not an installed dependency here
from titans_pytorch import NeuralMemory
mem = NeuralMemory(dim=384, chunk_size=64)
retrieved, mem_state = mem(sequence)
```

Its install name is `titans-pytorch`, not the supplied `titans-torch` API. The
proposed `NeuralLongTermMemory(...).test_time_update()` is **our custom API**, not
an asserted upstream function. No upstream Titans package or published numbers
were used in this experiment. The repo links the Titans paper identifier, but
this project has not reproduced that paper's algorithm/results.

## Next decisions

1. Run genuine GGUF stability first, then a real local encoder integration check.
   Do not bypass the gate or label fixture results empirical model behavior.
2. Validate one-slot neural **acquisition** in isolation before any long sequence.
   The cold byte decoder, padding-dominated BCE, fixed SGD scaling and confidence
   threshold are major confounds. Investigate a properly pretrained content codec
   or a normalized-delta fast-weight associative control, with settings selected
   on separate calibration values. Do not add hidden record/codebook lookup just
   to make the “native” condition succeed.
3. Only then test multiple slots, interference, paraphrases and cold vs calibrated
   neural memory with matched information/compute budgets. Keep frozen-memory and
   neural-only ablations to attribute any improvement.
4. Measure real GPU peaks on the target hardware; tune context/offload conservatively.
   Start smaller than 4B if needed. Do not extrapolate the scripted CPU timing.
5. Treat extraction quality/privacy and general prompt-injection resistance as
   separate unsolved evaluations, not consequences of JSON or a mission file.
