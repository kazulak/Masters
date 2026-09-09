# Cost-guided path study checkpoint

The controlling contract is section 11 of
`thesis/upmem-system-and-path-optimization-plan-v2.md`. This study changes path
search and its objective, not the accepted executor. It supersedes the protocol
in `upmem_family_aligned_path_study_v1.json`; historical observations and weights
are not imported into the new fit.

## Binding (2026-09-09)

- Preparation parent: `29ebaa2d1b20a1f6469435efe7309f89e161f263`.
- Execution tag: `thesis-upmem-kernel-schedule-system-v1`, verified to peel to
  `459935f586fdd16c82013838e6d27a12604c3093`.
- Frozen runtime source SHA-256:
  `b7168cd09f007978622346fd9954bdda54beb9ce48d870e4df8d158ed8681c02`.
- Unchanged workload SHA-256:
  `f2c85f27508d9b9fb5f55d13b89334939aa424330e27555dfafe3216452382d7`.
- Six training and six test instances; both qualified topologies per instance.
  Test interpretation is instance/size transfer, not family holdout.
- `configs/upmem_cost_guided_path_study_v1.json` binds the execution policies,
  binary identities, study settings, and finite budget. Recorded binary hashes
  still require fresh verification at hardware admission.

`scripts/upmem_cost_guided_path.py inspect` verifies local workload/QASM bytes
and calculates 192 initial, 144 first-feedback, 144 second-feedback, and 288
evaluation attempts: at most 768. The remaining 24 of the previous 792 cap are
not a retry or refill reserve. Inspection does not allocate or admit hardware.

## Research environment

Python 3.10.12, cotengra 0.7.5, and Optuna 4.5.0 are bound for this study.
`requirements-upmem-path-search.txt` pins the resolved isolated environment,
including transitive dependencies. It does not replace the frozen executor's
environment. The package extra `path-search` and hosted CI install the two
research libraries; actual traces must record and match the full research lock.

An isolated environment was installed under ignored
`runs/p6-preparation/cost-guided-research/.venv`; `pip check` passed. A pinned
Optuna smoke test accepted a positive-infinity rejected proposal and then a
finite proposal. The installed cotengra signature supports `max_repeats`, scalar
`costmod`/`temperature`, explicit seed, `accel=False`, and `parallel=False`.
It does not accept `max_time`.

## Implementation status

The pure `upmem_launch_cost_v1` primitives implement serial H/P/N contributions
and a joint movement-plus-work maximum per physical launch, development-only
global median scales, and all 1,001 integer coefficient tuples. Historical
six-term and log-ratio functions retain their previous meaning.

The host-pass observer and serial adaptive-search engine are implemented.
Fifty focused tests pass, including a full 128-proposal production-plan score
callback, two fresh-process trace comparisons, objective-responsive proposals,
timeout cleanup, fact-snapshot ownership, and all 24 workload/topology greedy
metadata extractions. The independent read-only score/host-pass and search
reviews found no remaining blockers. The incorrectly suspected output-reshape
copy was withdrawn after a real assembly memory-sharing regression.

This checkpoint is not a calibrated optimization result. End-to-end acceptance
of physical observations, feedback-round control and final physical evaluation
remain pending. The follow-on software work below adds the pure grid fitter.
In particular, scoring a fixed candidate pool does not satisfy cost-guided
search. The final G/F/R/U comparison must demonstrate and evaluate feedback
during proposal generation.

The private CLI exposes `inspect`, `initialize`, `initial-search` and
`freeze-initial`. These are offline preparation commands, not a hardware
controller. No accepted-round resume, physical collection, or fitting command
is enabled by this checkpoint. Mandatory search tests do
not skip when Optuna or a private environment path is unavailable: they use
the active test interpreter and require the pinned library versions.

An initial full-suite run found two existing M7C artifact checks failed because
of a redundant Optuna addition to the hashed historical CI constraints file.
That addition was removed, both affected tests passed, and the historical
constraints file is unchanged. No legacy artifact or test expectation was
rewritten; the research extra and research lock carry the new dependency.
The completed rerun passed all 2,023 tests in 225.17 seconds with no skips.
Ruff passed across `src`, `tests`, and `scripts`; `git diff --check` passed.

No new SDK or physical campaign has been run at this checkpoint. No coefficients
or normalization scales have been fitted or frozen from physical observations.

## Follow-on software checkpoint

The paired-observation fitter enumerates all 1,001 integer tuples and uses
session-inclusive same-cell/round/block log ratios with equal family weighting
and the prescribed deterministic tie breaks. It rejects evaluation data,
incomplete or duplicate observations, policy/source mismatches, fallback,
numerical failures and failed execution/startup admission. Its private result
is not evidence acceptance; the existing canonical archive verification and
two-copy acceptance must precede any call using physical observations.

The `frozen_path` planning input replays the exact selected complete path through
the existing runner. Strict integer-pair validation and network/DAG hashes
precede native allocation. It does not rerun a library search, select a new
schedule, or change runtime, transport or numerical behavior. Tests include a
CPU runner roundtrip through canonical evidence verification. The initial
candidate packet adapter reuses the existing collection, resource-route and
QASM helpers, retaining all path roles while deduplicating execution rows.

Offline preparation requires the exact clean source and pinned environment.
Initialization freezes only development-greedy scales. Each initial cell search
reserves a new directory and retains every proposal in flushed JSONL; a failed
or interrupted search cannot be refilled. Freezing the initial round checks all
declared cells, invocation/seed/row identities, objective tells and duplicate
facts against the complete traces. It does not grant physical admission.

Final G/F/R/U selection binds the supplied profile and normalization by hash
and verifies the scored U trace against that profile. U cannot inherit F's
candidate pool. Search records distinguish adaptive ask/tell, candidate
generation and complete-plan evaluation durations; the latter currently combines
lowering, feature extraction and score evaluation. Timing fields are excluded
from deterministic trace equality, never from the retained raw trace.

The integration still to finish is bounded round execution/acceptance using the
existing physical runner, offline refits from newly accepted raw evidence,
feedback search commands, final freeze/evaluation and durable reporting. Do not
run a campaign merely because preparation or pure fitting tests pass.

Follow-on qualification: all 2,192 tests passed in 227.87 seconds in the pinned
research environment, with no skips. Ruff passed across src/tests/scripts and
diff checks passed. The independent replay review found no blockers; the
preparation review's objective/profile and trace-provenance findings were
repaired and passed its focused re-review. Runtime and historical CI constraints
retain the hashes recorded above. No new SDK or physical execution was performed.

## Host-pass inventory

`execution_features.extract_launch_cost_features` observes the production
control expansion and adds a separately identified float32 host-pass proxy.
Historical execution-feature fields and semantics are unchanged. For each
contract, source storage dtype is the initial tensor dtype or complex64 for a
produced intermediate, as used by the frozen runtime.

| Pass | Modeled reads plus writes |
|---|---|
| Forced operand copy (`runtime._prepare_complex_operation`) | Twice each source operand's bytes |
| Component casts before real/imag lowering | Two passes of `elements * (source_component_bytes + 4)` when dtype changes |
| Operand-only label reductions (`tiling._sum_labels`) | `8 * (input_elements + reduced_elements)` across both components |
| Canonical contiguous copies (`tiling._as_batched_matrix`) | `16 * reduced_elements` when required; singleton-axis permutations and contiguous views cost zero |
| Complex canonical packing and encoded-plane copies | `32 * (B*M*K + B*K*N)` |
| Tile contiguity (`wave_work._encoded_slice`) | `2*L` for each noncontiguous input-plane slice |
| Tile bytes and padding | `L+A`, plus `2*A` when padding requires concatenation, for logical length L and aligned length A |
| Envelope payload join (`packed_wave.pack_wave_envelope`) | `2*A` per nonempty input plane |
| Four-lane assembly (`tiling.assemble_output_tiles`) | `64*D + 12*sum(product_count*m*n)` over the node's active slots |
| Complex decode and final operation copy | `56*D`, with D output elements |
| Host graph reduction | `16*D + 24*D*(input_count-1)`, plus any required source dtype conversion |
| Final result copy | `(source_dtype_bytes + 8)*output_elements` |

Array views, result `memoryview` slices, and `np.frombuffer` do not copy payloads.
Assembly restores the exact declared output shape, so its transpose/reshape is
a view, not a compulsory contiguous copy. All counts come from shapes, labels,
dtypes and emitted controls; no dummy intermediate arrays are allocated.

P deliberately excludes finiteness-validation scans, digest hashing, filesystem
reads/writes, native SDK buffer copies, Python records, control-record byte
construction, allocator internals, and hidden library temporaries. It is the
volume of these named host preparation/reconstruction passes, not measured
memory-bus traffic or all host work. H includes planned aligned host/DPU traffic;
M estimates local MRAM/WRAM traffic. Distinct legs of an actual copy chain are
counted separately, not asserted to be independent timing mechanisms.
