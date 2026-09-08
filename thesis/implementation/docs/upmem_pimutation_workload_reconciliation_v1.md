# PIMutation Workload Reconciliation v1

Status: `workload_definitions_frozen`. This note freezes the 12 workload
definitions and their training/test roles only. It does not freeze the physical
admission, candidate pool, fitted pretest profile, or physical execution, and
it does not authorize SDK or physical execution.

**Fidelity hold (after definition commit `1e14dd7`):** the recorded definitions
are not yet accepted as the final PIMutation workload. Do not generate their
candidate pools, calibrate, or fit weights. Source comparison below establishes
a mismatch in HS and a changed QRNG computation in the repeat-two proposal.
Keep the definition commit and prior evidence intact; correct workload semantics
under a new explicit source/definition identity before lifting this hold.

## Purpose

The current six-instance P6 path record is a development study. Its 92-attempt
development packet and its existing split labels are not the final benchmark
workload. The final benchmark must first reconcile the six families represented
by Table 2 of Lee et al., *PIMutation: Exploring the Potential of PIM
Architecture for Quantum Circuit Simulation*:

<https://arxiv.org/html/2503.00668v1>

The paper is a primary workload reference, not evidence that this repository's
local generators or execution paths are the official PIMutation implementation.

## Family and generator audit

The following counts are one-repeat gate-count facts from the lead's local
metadata audit at the 16-allocated-qubit case (HS uses 8 logical and 16
allocated qubits). Counts are shown as `1Q / 2Q`. The `quest_compatible` column
matches the paper Table 2 counts for every family; matching counts do not
establish gate-order or output equivalence or make this an exact paper instance.

| Paper family | Paper Table 2 | Current `builtin` | Current `quest_compatible` | Repository interpretation |
| --- | ---: | ---: | ---: | --- |
| BB84 / `BB_n` | 32 / 0 | 22 / 0 | 32 / 0 | workload-shape reproduction |
| BV / `BV_n` | 32 / 15 | 32 / 8 | 32 / 15 | shape reproduction; not textbook phase-kickback by default |
| EDC / `EDC_n` | 32 / 30 | 2 / 15 | 32 / 30 | workload-shape reproduction |
| HiddenSubgroup / `HS_2n` | 48 / 16 | 24 / 8 | 48 / 16 | identity-preserving shape reproduction |
| QRNG / `QRNG_n` | 16 / 0 | 16 / 0 | 16 / 0 | textbook QRNG in the native manifest |
| XOR / `XOR_n` | 0 / 15 | 1 / 15 | 0 / 15 | workload-shape reproduction |

The current Python source has separate alias tables for the two generator
kinds. Both map `hidden_shift` to the internal family `hs`, but that alias is
not a paper identity: [`circuits.py`](../src/quantum_bench/circuits.py#L14)
and [`circuits.py`](../src/quantum_bench/circuits.py#L49)
define the aliases, while `quest_compatible_circuit` and `builtin_circuit`
construct different operation sequences at
[`circuits.py`](../src/quantum_bench/circuits.py#L192)
and [`circuits.py`](../src/quantum_bench/circuits.py#L288).

The primary recommendation for a future paper-aligned benchmark is to use the
existing `quest_compatible` generator after source and semantic qualification,
because its audited repeat-one gate counts match Table 2. `quest_compatible`
names a circuit generator/source kind; it does not select CPU placement or a
runtime backend. The local reconstructed generator and the native manifest do
not authenticate the official paper artifact. Gate-count agreement is therefore
necessary evidence for reconciliation, not proof of gate order, state-vector
output equivalence, or algorithmic identity.

The native manifest records the six names and their shape-level interpretation
at [`circuit_manifest.c`](../native/quest_cpu/src/circuits/circuit_manifest.c#L7)
and [`README.md`](../native/quest_cpu/README.md#L41).
The general Python manifest currently labels several raw source names as
`workload-shape reproduction` at
[`circuits.py`](../src/quantum_bench/circuits.py#L486);
that label must not be read as an exact-paper claim.

## Historical versus final scope

The archived benchmark matrix lists the six families and sizes 8, 10, 12, 14,
16, 18 and 20, yielding 42 historical family/size cases at
[`THESIS_BENCHMARK_MATRIX.md`](archive/architecture-reset-2026/THESIS_BENCHMARK_MATRIX.md#L7).
That grid is historical continuity evidence, not a campaign freeze and not a
declaration that those are the paper's exact instances. The archived matrix
also explicitly says the comparison is relative rather than identical in
kernels, paths, software versions or hardware
([same document](archive/architecture-reset-2026/THESIS_BENCHMARK_MATRIX.md#L161)).
The old CPU/GPU roadmap and its route/repeat choices are not imported into the
P6 path-study budget.

The current P6 development record contains six instances, including the
Stress/HS/EDC development cells and the GHZ/XOR development candidates. It is
not the final six-family PIMutation benchmark. The existing configuration and
archive remain byte-for-byte historical records; they are not silently
retargeted.

## Required pre-timing packet

Before any calibration timing, create and hash a new packet with these reserved
identifiers:

| Artifact | Reserved ID | Required contents |
| --- | --- | --- |
| Workload manifest | `upmem_pimutation_full6_workload_v1` | all six families, exact generator kind/name/parameters, operation identities, exposure disposition |
| Candidate set | `upmem_pimutation_full6_candidates_v1` | fixed-pool source, generator seed/strategy, candidate and physical-plan hashes, no post-timing generation |
| Study config | `upmem_pimutation_full6_path_study_v1` | execution profile/contract, numeric policy, topology, source, model and split fields |
| Budget record | `upmem_pimutation_full6_budget_v1` | exact rows, warmups, measurements, adaptive rounds, elapsed cap, failed-attempt accounting and no-refill rule |

The workload manifest and study protocol now define the final workload roles
and a bounded attempt protocol: 792 new attempts plus the separate 92-attempt
historical development record, or 884 in the aggregate ledger. This is not a
physical-admission or candidate-pool freeze. The closed 92-attempt development
allocation and the former 540-attempt template cannot be reused by relabeling.

The gating order is: define and review the six-family workload; generate the
fixed candidate pools with the declared strategy; freeze the physical selections;
qualify source, plan, memory and execution identities; calibrate; fit; freeze
the pretest profile; then test. Candidate generation is permitted after workload
review and before the fixed-pool freeze. After that freeze, timing cannot create
new candidates or refill deduplicated slots.

The final packet must satisfy all of the following before timing:

1. Every one of the six families remains in the benchmark ledger. Low headroom,
   collection underutilization, or hard infeasibility is recorded with its raw
   admission facts and is not omitted or replaced after timing.
2. Training and test use the same six families but have different canonical
   instance identities: `(generator kind, generator name, canonical parameters,
   operation identity)`. This is instance-held-out evidence, not a
   disjoint-family split; changing size alone does not justify a family-held-out
   claim.
3. Before calibration timing, freeze the workload, candidate/config/source/
   split/budget, fixed candidate pools, physical selections and execution
   contract. Calibration then supplies training evidence; fit the profile after
   training and model comparison and freeze its hash before final test timing.
   The profile hash is not a circular pre-calibration requirement. A bounded
   exposure audit covers local and available remote metadata. Unknown or
   unavailable evidence remains explicitly unknown; correctness-observed is not
   optimization-unseen.
4. The frozen executor memory policy remains a declared 512 MiB host admission
   budget with zero reserve, not an RSS bound. Plan identity, native allocation,
   active-DPU/rank coverage, startup, execution, replay and accuracy gates remain
   mandatory. A full complex128 statevector at 32 qubits is about 64 GiB and
   therefore cannot be treated as admitted under this 512 MiB policy; no
   automatic 32-qubit claim follows.

Additional families or instances may be proposed later only under a separately
identified workload/config/budget amendment. They cannot displace or silently
reduce the required six-family benchmark.

The workload split is now defined and frozen for the final benchmark role, but
no model form, candidate set, physical admission, timing result, or physical
claim is established by this reconciliation note. The current P6 freeze record
is superseded for the final benchmark role, while its original configuration,
source identity and development evidence remain preserved for audit.

Generation may start only after lead review and commit of the new pre-timing
packet. Later additions require a separate identified workload/config/budget
amendment and cannot displace the required six families.

## Finalized Full-Six Workload Definition

The workload manifest is `upmem_pimutation_full6_workload_v1` with status
`workload_definitions_frozen`. The decision is:

| Role | Families | Generator | Parameters |
| --- | --- | --- | --- |
| Training | all six families | `quest_compatible` | `n_qubits=16`, `repeat_layers=1` |
| Test | BB84, BV, HS, QRNG, XOR | `quest_compatible` | `n_qubits=18`, `repeat_layers=2` |
| Test | EDC | `quest_compatible` | `n_qubits=15`, `repeat_layers=1` |

For HS, `n_qubits` is represented as `allocated_qubits`; the allocation is
even and `logical_qubits=allocated_qubits/2`, with `depth=1`. Training and test
therefore retain all six family labels in both splits while using distinct
canonical definition identities. This is mixed parameter-heldout evidence, not
a family-heldout split and not a uniform size/depth transfer claim: five test
instances use n=18/repeat=2, while EDC uses the admissible n=15/repeat=1
exception. The exact paper instances remain unverified.

The manifest records gate counts and `quantum_bench.problem_id.v1` operation
identities generated by the existing core circuit API only. It does not create
TN paths, candidates, timing rows, or a physical budget. The bounded local and
remote parameter audit located no prior repeat-2 record, but this remains a
bounded non-match rather than global untouched certification. The prior 92-row
P6 record remains development-only, is excluded from the final fit, and its raw
evidence is not rewritten.

The EDC n=15/repeat=1 planning record reports both topologies feasible, work
72, semantic expansion 73,870, and memory admission pass; its supplied planning
record is
`runs/p6-preparation/pimutation-full6-reconciliation-v1/greedy-test-edc15-r1-feasibility.json`
with SHA-256
`552f4ee6058307bf1f4a8d8c35ac57a33bf8a890c5919d2c884391ce8f103505`.
It contains no timing or physical-execution claim.

Two uniform EDC probes were rejected before timing by the planning
identity-expansion gate. The n=18/repeat=2 probe reached the early-cutoff lower
bound 1,048,852 against the 1,000,000 preregistered cap. The n=16/repeat=2
probe reached the early-cutoff lower bound 1,048,820 against the same cap.
Neither number is an exact full-DAG total, and neither finding implies that a
larger cap would admit the workload. This is a preparation-safety identity
expansion gate, distinct from DPU, WRAM, and MRAM admission; neither probe made
an invalid physical attempt.

The execution context remains the existing rank-1 `1dpu_t8` and `4dpu_t8`
contract, float32 wave policy, accepted source `459935f...`, and declared
512 MiB host budget with zero reserve. The study protocol defines 792 new
attempts: 288 initial training, 144 for one adaptive training round, 144
training confirmation, and 216 test, plus the separate 92-attempt historical
development ledger. There is no separate physical validation stage. Candidate
generation, greedy/reference selection, and the 65-member fixed-pool strategy
remain pending lead review/commit and later qualification. The old 92-attempt
record is development-only, excluded from the final fit, and its raw records
remain unchanged. Optional families remain deferred; all six required families
stay in both splits.

## Source-Fidelity Review After Definition Freeze

Software qualification of `1e14dd7bb63b9f530c2542dbe5daeb763de7df2a` passed:
1,922 local tests in 211.61 seconds, Ruff, and hosted CI run
`34284570809`. This proves software consistency, not paper fidelity. The local
JUnit artifact `runs/p6-preparation/pytest-pimutation-full6-definition.xml` has
SHA-256 `66b2e3865e14c47e86c297d0588578c8c75608d72a67e1c0349ed530044f7503`.

PIMutation Table 2 cites QASMBench for HS and QRNG. The inspected primary
reference is [QASMBench HS4](https://github.com/pnnl/QASMBench/blob/357b942396d5c2b7cbc1c229c585a6ef5ccaebac/small/hs4_n4/hs4_n4.qasm),
pinned to `357b942396d5c2b7cbc1c229c585a6ef5ccaebac`. Its four-qubit circuit
has 24 single-qubit gates and four CNOTs before terminal measurements; this is
not the 12 single-qubit gates and four CNOTs implied by PIMutation's HS row.
It therefore supplies a concrete source comparison, not authentication of the
authors' exact transformed benchmark.

For one disjoint control/target pair, the published chronological gate sequence
is `Hc,Ht,Xc,Ht,CX,Ht,Xc,Hc,Ht,Ht,CX,Ht,Hc,Ht`. Direct multiplication using
the conventional real H/X/CNOT matrices maps `|00>` to `|10>` (control first).
The repository sequence is `Hc,Xc,Hc,CX,CX,Hc,Xc,Hc`, which is identity:
the CNOTs cancel and each `H X H` is Z. Thus the sequences differ in both
unitary and zero-input output; matching gate counts cannot establish fidelity.

The proposed QRNG test repeats every H twice. Since `H H = I`, it returns
the zero-input state rather than the uniform state of one QRNG layer. XOR's
CNOT-only circuit also preserves zero input, although that alone does not
prove a defect in a benchmark that intentionally includes such operations.
Identity or simple-output circuits are not intrinsically invalid execution
tests; they must not silently substitute for the requested algorithm workload.

Required next action is a bounded source-backed reconciliation of the six
families and their parameterizations, including terminal-measurement removal
for the established pre-measurement full-statevector query. Preserve all six
families. Do not solve this mismatch by relabeling the current shapes, selecting
only convenient families, or changing kernels. No new pool, physical timing,
or weight fit was produced under these definitions.
