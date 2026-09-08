# PIMutation Workload Reconciliation v1

Status: `reconciliation_pending`; documentation only. This note does not freeze
the final split, fit a model, generate candidates, or authorize SDK or physical
execution.

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

These are reserved identifiers, not a fabricated final split. The numeric final
budget remains unresolved until the six-family instances and exposure audit are
complete. The closed 92-attempt development allocation and the former 540-attempt
template cannot be reused by relabeling.

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

No final split, model form, candidate set or physical claim is established by
this reconciliation note. The current P6 freeze record is superseded for the
final benchmark role, while its original configuration, source identity and
development evidence remain preserved for audit.

The current P6 pause remains in force until this workload reconciliation and its
pre-timing packet are reviewed. Later additions require a separate identified
workload/config/budget amendment and cannot displace the required six families.
