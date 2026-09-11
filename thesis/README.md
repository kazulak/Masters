# UPMEM Tensor-Network Quantum-Circuit Simulation

This directory contains the research implementation and reproducibility artifacts for a
Master's thesis on exact tensor-network quantum-circuit simulation using UPMEM
Processing-in-Memory hardware.

**Research status: complete and frozen.** The final physical UPMEM execution system
(P1-P5) and the cost-guided contraction-path study (P6) have been completed. The final
P6 campaign executed 678 physical attempts within a preregistered ceiling of 768, with
zero retries and zero replacement observations.

## Research question

The implementation studies whether exact, untruncated tensor-network contraction can be
mapped efficiently to digital PIM and whether contraction-path selection should account
for the execution characteristics of the UPMEM system rather than conventional FLOP cost
alone.

The final retained UPMEM execution profile uses:

- one UPMEM rank;
- `packed_wave_v1` transport and one persistent native controller;
- `static_dag_waves_v1` scheduling of dependency-ready contractions onto disjoint DPU
  groups;
- intra-contraction tiling and tasklet parallelism;
- `fused_when_admitted_v1` four-product complex execution;
- the retained WRAM-panel kernel (`panel_only_v1`);
- host-roundtrip intermediates;
- split-complex float32 as the primary performance policy;
- shared-scale complex int8 as a separately characterized numerical policy.

The project does **not** claim multi-rank scaling, energy efficiency, universal int8
accuracy, arbitrary graph residency, or general superiority over CPU/GPU simulation.

## Main findings

The final composed executor showed substantial hierarchical scaling on the recorded
Stress16 diagnostic: 1-DPU T1 to 1-DPU T16 achieved 4.501040x session-inclusive
speedup, while 1-DPU T8 to 4-DPU T8 achieved 1.530679x.

Four-product complex launch fusion passed fresh development confirmation at 1.9034x
session-inclusive speedup for the confirmed cell. Static DAG-wave scheduling achieved a
1.214213x equal-cell geometric session-inclusive speedup across its six-cell development
A/B, with a fresh Stress16 D4/T8 confirmation of 1.431726x. The outer-K1 specialization
and the bounded resident-pair probe were valid negative results and were not retained.

The final P6 path study compared:

- **G** — deterministic greedy contraction path;
- **F** — FLOP-selected path from a bounded cotengra search;
- **R** — UPMEM-aware reranking of the same conventional search trace;
- **U** — a separate cotengra search in which the UPMEM cost is fed back during adaptive
  candidate generation.

Across 12 held-out circuit/topology cells, the primary R/U session-inclusive ratio was
1.001343x with a descriptive paired-block 95% interval [0.994836, 1.008509]; R and U
selected the same physical path in 8/12 cells. Under the tested search budget, adaptive
UPMEM-guided generation therefore provided no resolved benefit beyond UPMEM-aware
reranking.

UPMEM-aware selection itself was useful: F/U was 1.040425x session-inclusive overall
and 1.081841x on the 4-DPU topology. G/U was 1.266508x session-inclusive overall.
These ratios describe physical execution after path selection; offline path-search time is
reported separately.

## Scientific source identities

The physical executor is frozen at:

```text
459935f586fdd16c82013838e6d27a12604c3093
tag: thesis-upmem-kernel-schedule-system-v1
```

The P6 software used for every new physical attempt is frozen at:

```text
2beea27411c16e90ed76988613ddb00bcc09f942
tag: thesis-upmem-cost-guided-software-v1
```

The accepted P6 result/audit package was recorded at:

```text
8df2ebac61bacd08309ea490309be5a8dcb943b2
tag: thesis-upmem-cost-guided-results-v1
```

Later documentation/publication commits do not replace those scientific source identities.

## Start here

The active code is in:

```text
implementation/
```

Technical usage and tests:

```text
implementation/README.md
```

Final implementation status:

```text
implementation/STATUS.md
```

Architecture:

```text
implementation/ARCHITECTURE.md
```

Documentation index:

```text
implementation/docs/README.md
```

Final cost-guided audit package:

```text
implementation/thesis_results/upmem_cost_guided_path_v1/
```

The P6 research protocol is preserved in:

```text
upmem-system-and-path-optimization-plan-v2.md
upmem-path-search-operational-runbook-v1.md
```

They are historical protocol records and must not be retroactively edited to make the
observed results look cleaner.

## Reproducing the software checks

```bash
git submodule update --init --recursive

cd implementation
python3.10 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
../.venv/bin/python -m pip install -c ci/constraints.txt -e '.[dev,path-search]'

make PYTHON=../.venv/bin/python test
../.venv/bin/python -m ruff check src tests scripts

cd thesis_results/upmem_cost_guided_path_v1
sha256sum -c SHA256SUMS
python3 tools/test_p6_readout.py
```

These commands reproduce software and audit-package checks. They do not reproduce the
historical physical campaign merely by running on an arbitrary machine.

## Thesis-ready final numbers

The standalone publication repository contains a post-experiment reporting layer under
`publication/`. It verifies the immutable P6 audit package and produces thesis-ready
numbers without executing hardware or refitting the model:

```bash
python3 publication/build_thesis_numbers.py \
  --repo . \
  --output publication/generated
```

See `REPRODUCIBILITY.md` and `PROVENANCE.md` in the standalone repository for the exact
scope of reproducibility and the mapping back to the original research repository.
