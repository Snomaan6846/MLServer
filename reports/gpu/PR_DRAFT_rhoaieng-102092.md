# PR DRAFT — RHOAIENG-102092: REST binary tensor extension + gRPC converter fix

> STATUS: **READY FOR PR** — final campaign complete (Entry 013: 159/159
> trials valid, all parity-gated). Everything below uses final-campaign
> numbers. Only PR creation remains, pending user approval.
> Target: `opendatahub-io/MLServer` master ← fork `rhoaieng-102092`

## Commits

- `0dd67502` — KServe v2 binary tensor data extension for REST
- `8424bf32` — eliminate redundant deep copy in gRPC raw converters

Rebased onto current master `c39ea879`; the changes above are bit-identical
to the benchmarked commits, and the full suite was re-verified green on the
combined tree.

## Title

RHOAIENG-102092: KServe v2 binary tensor data extension for REST inference

## Summary

Two performance changes for the RF-DETR serving study on Tesla T4
(RHOAIENG-102092), each benchmark-gated with bit-exact/byte-exact parity in
every trial:

1. **KServe v2 Binary Tensor Data Extension for REST** (`0dd67502`) —
   zero-copy `np.frombuffer` views over binary request bodies, binary
   response encoding, JSON fallback with zero behavior change.
   **4.0× REST throughput** (7.8 → 31.4 RPS stock→binary at C2–C16;
   6.2× at C1), request payload 15.26 MB → 3.15 MB, response 542 KB →
   114 KB.

2. **gRPC raw converter deep-copy elimination** (`8424bf32`) — the raw
   converters deep-copied every input/output (`model_copy(deep=True)`)
   before clearing a field that `include_contents=False` never reads.
   Now packs straight from the originals: zero copies, zero mutations,
   byte-identical wire output. **+19.5% gRPC single-request throughput**
   (C1 19.08 → 22.81 RPS [CI95 1.180–1.209]; p50 52.3 → 43.7 ms), parity
   at load (GPU-capped).

## Evidence

### Final campaign (Entry 013 — 159 trials, 4 sessions, 3 trials/cell, paired bootstrap CI, bit-exact/byte-exact gates on every trial)

Automated campaign at `8424bf32`, ship image digest verified in every
session's receipts, hermetic test gate (`tox -e mlserver-venv` from
`poetry.lock`: **1427 passed / 37 skipped / 0 failed**, excluding
docker-daemon-dependent `tests/cli/test_build.py`).

### REST (binary tensor, recommended config)

| | Stock | Binary tensor | |
|---|---|---|---|
| RPS @ C8 | 7.84 | 31.47 | **4.02×** [4.013–4.067] |
| RPS @ C16 | 7.84 | 31.44 | **4.00×** [3.996–4.024] |
| C1 p50 | 251 ms | 41 ms | 6.2× |
| C1 RPS | 3.97 | 24.44 | **6.24×** [6.143–6.237] |
| GPU util @ C8 | 24% | 80% | device-bound |
| CPU-ms/req @ C8 | 178 | 42 | −76% |
| Memory | parity | parity | no regression |

JSON fallback (feature ON, JSON requests): 1.05× — parity/no-regression
at every concurrency. One marginal cell: C4 p99 +5.4% (592 vs 562 ms,
just over the 5% tolerance); p50 better, all other concurrencies equal or
improved.

### gRPC (opt-in config: raw buffers + zmq + converter fix)

| | Stock | Opt-in | |
|---|---|---|---|
| C1 RPS | 5.02 | 22.86 | **4.51×** [4.486–4.614] |
| C8 RPS | 7.92 | 31.12 | **3.93×** [3.925–3.976] |
| C8 p50 | 1,009 ms | 257 ms | −75% |
| CPU-ms/req @ C8 | 194 | 16 | −92% |

No p99 or memory regressions at any concurrency.

### Batched model (max_batch_size=4, dynamic batch)

Binary over stock: 1.08–1.23× at C1–C4. Mixed binary/JSON co-batching at
no cost vs pure JSON. Note: all batched variants — stock included —
degrade at C8+ (adaptive-batcher queue behavior, pre-existing and
independent of these changes; documented as Phase-3 Track C1).

### Stability

60-minute soak on the ship candidate: 113,483/113,483 requests (zero
failures), 31.52 RPS sustained, p50→p99 spread 1.7%, memory flat
(+2.4 MB/h = noise), PSI pressure 0.00. Final campaign round-to-round
spreads 0.1–2.8% on all unbatched cells.

## Config recommendation (no code): `cuda_graph=true`

Validated on the binary REST path: +0.5% (parity; CI-significant at 4/5
concurrencies), zero regressions, existing PR #356 functionality —
recommend enabling for fixed-shape FP32 single-input models.

## Test plan

- Full suite green in a hermetic venv built from `poetry.lock`
  (`tox -e mlserver-venv`): **1427 passed / 37 skipped / 0 failed**,
  excluding `tests/cli/test_build.py` (requires a real docker daemon; the
  local docker socket is podman's compat shim — environmental, no
  code-path overlap with these commits; CI runs them on real docker).
- Binary extension: 65 unit + 22 batching + 32 endpoint tests, including
  the adaptive-batching coexistence contract and the fallback compliance
  sweep.
- gRPC: 125/125 converter/servicer tests (existing double-conversion and
  caller-mutation guards pass unchanged) + large-output regression test.
- Live-server validation: bit-exact JSON↔binary parity (REST), bit-exact
  gRPC↔REST cross-protocol parity, byte-identical responses across
  repeats.
- Every benchmark trial (159/159) gated on bit-exact REST / byte-exact
  gRPC responses vs per-device references (CUDA graph replay and the
  batched model included); zero failed requests.

## Reviewer notes

- Scope is steady-state prediction performance (startup/loading excluded
  per ticket).
- Not included (evaluated and closed with evidence, see the study ledger):
  msgpack IPC serialization (reached ~parity; IPC is ~3% of CPU budget),
  httptools (no measurable gain), smaller FP32 JSON payloads (already
  minimal).
- `rest_worker_json` regresses on slow-CPU GPU nodes (0.67×); hardware-
  class guidance is documented in the study (keep OFF on this class).
- Full evidence trail: final-campaign session data (159 trials with
  receipts carrying env, pod UID, image digest and parity hashes),
  profiling flamegraphs (py-spy/memray), and the study ledger
  (`BENCHMARKING_HISTORY.md` entries 001–013).

## Independent review response

An independent Codex review of both commits found 3 P1 + 5 P2 issues in the
binary extension (the gRPC fix was confirmed sound: 26 wire comparisons,
byte-identical across 13 datatypes). All findings fixed and verified:

- **Per-output binary negotiation** (KServe): request-wide default + named
  overrides; JSON requests can receive binary; mixed responses
- **BYTES framing**: length-prefixed elements, byte-identical to
  `mlserver.raw`'s Triton-derived implementation (differential-tested),
  bounds-checked; all 18 truncation points rejected
- **Adaptive batching**: array inputs restored before list-based batching,
  transport-independently
- **Error mapping**: malformed metadata → 400 (never 500); gzip → 400;
  text/plain → 422 — `test_worker_body_errors.py` passes **unmodified**
- **Metrics**: single accounting per request (double-count removed)
- **Headers**: model/CloudEvents preserved on binary responses
- **Route composition**: `rest_worker_json` worker path preserved for JSON
  bodies when both settings are on
- **Zero-copy**: input arrays are true memoryview views of the body
  (`np.shares_memory` verified)

Verification beyond unit tests: 25 endpoint-level tests through the HTTP app
with the setting enabled; real-model E2E — 7 negotiation paths bit-exact
through actual RF-DETR inference; the reviewer's reproduction cases re-run
against the fixed code. Full suite: 1350 passed.

## Follow-ups (not in this PR)

- ndarray-native response pipeline (codec keeps arrays end-to-end) —
  the identified lever for the remaining ~10% gap to Triton (31.5 vs
  35.1 RPS on identical hardware/model).
- Adaptive batcher high-concurrency collapse (measured in the final
  campaign, Entry 013 / Phase-3 Track C1): all batched variants — stock
  included — drop from 8.3 to 1.7 RPS at C8+ with the GPU idle while
  requests queue. Pre-existing behavior, independent of these changes;
  harness and references persisted for the fix→verify loop.
