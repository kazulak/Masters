# Implementation

This directory contains the active implementation used for the Master's thesis research
on exact tensor-network quantum-circuit simulation on UPMEM PIM hardware.

**Final status:** research implementation frozen. The executor and P6 path-search campaign
are complete; no further optimization is required for the thesis result.

## Final execution flow

```text
SimulationJob
  -> target-neutral TensorNetwork
  -> complete contraction path
  -> ContractionDAG
  -> target-specific execution

CPU/TN routes:
  NumPy same-DAG replay / Quimb / cotengra / QuEST

Final UPMEM route:
  ContractionDAG
  -> UpmemPlan
  -> static_dag_waves_v1
  -> packed_wave_v1
  -> persistent native ABI-v5 prepared-wave execution
  -> WRAM-panel DPU kernel
  -> deterministic host reconstruction/reduction
  -> canonical evidence
```

`TensorNetwork` contains semantic tensor-network structure only. `ContractionDAG` is the
logical execution IR. `UpmemPlan` and the static DAG-wave scheduler contain target
placement, tiling, topology, and execution-policy decisions.

## Frozen UPMEM profile

The final retained executor is commit
`459935f586fdd16c82013838e6d27a12604c3093`, tagged
`thesis-upmem-kernel-schedule-system-v1`.

The retained study policy is:

```text
transport:           packed_wave_v1
schedule:            static_dag_waves_v1
complex execution:   fused_when_admitted_v1
geometry:            panel_only_v1
intermediates:       host_roundtrip_v1
primary numeric:     split_complex_float32_v1
rank count:          1
```

Tasklet parallelism, multi-DPU contraction, and independent-DAG execution on disjoint DPU
groups are part of the frozen executor. Outer-K1 specialization and production residency
were evaluated but not retained. Exact slicing remains an explicitly declared
transformation rather than an automatic production policy.

## Final P6 path optimization

The P6 software source is
`2beea27411c16e90ed76988613ddb00bcc09f942`, tagged
`thesis-upmem-cost-guided-software-v1`.

The five-term launch-aware ranking surrogate is:

```text
C =
    theta_H * H / s_H
  + theta_P * P / s_P
  + theta_N * N / s_N
  + sum_launch max_dpu(
        theta_M * M[launch,dpu] / s_M
      + theta_W * W[launch,dpu] / s_W
    )
```

The final frozen integer weights are:

```text
[1, 2, 1, 1, 5]
```

These are **ranking parameters** over normalized model terms. They are not measured
runtime percentages and are not unique architectural constants.

The final P6 audit package is:

```text
thesis_results/upmem_cost_guided_path_v1/
```

Its accepted physical campaign used 678 attempts within the 768-attempt ceiling, with
zero retries and zero replacements.

## Main P6 result

The primary held-out comparison is R/U:

```text
R = UPMEM-aware reranking of the conventional F search trace
U = separate search with UPMEM cost fed back during adaptive generation
```

Session-inclusive R/U:

```text
1.0013425605931643x
descriptive paired-block 95% interval:
[0.9948355448727421, 1.0085087497299605]
same selected path: 8 / 12 cells
```

Therefore the bounded experiment does not resolve an additional physical benefit from
UPMEM-guided candidate generation beyond UPMEM-aware reranking.

UPMEM-aware selection does improve on the FLOP-selected path:

```text
F/U overall session-inclusive: 1.040424568249768x
F/U 4-DPU session-inclusive:   1.0818412974163845x
F/U 4-DPU steady wall:         1.1153755638648002x
```

Search time is an offline planning cost and is reported separately from physical
session-inclusive execution.

## Install and test

From this directory:

```bash
python3.10 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
../.venv/bin/python -m pip install -c ci/constraints.txt -e '.[dev,path-search]'

make PYTHON=../.venv/bin/python test
../.venv/bin/python -m ruff check src tests scripts
```

The publication CI also builds the QuEST CPU runner and verifies the final P6 audit
package.

## Normal software commands

```bash
make PYTHON=../.venv/bin/python plan \
  CONFIG=configs/tn_benchmark_reset.yml \
  OUTPUT=runs/example-plan

make PYTHON=../.venv/bin/python run \
  CONFIG=configs/tn_benchmark_reset.yml \
  OUTPUT=runs/example-run

make PYTHON=../.venv/bin/python verify \
  INPUT=runs/example-run

make PYTHON=../.venv/bin/python report \
  INPUT=runs/example-run \
  REPORT_OUTPUT=runs/example-report
```

## Physical hardware warning

Historical physical UPMEM experiments are already complete and frozen. Do not rerun them
as part of normal repository verification.

The physical controller intentionally requires exact source, binaries, SDK, resource
ownership, CPU/governor facts, evidence storage, and a once-only invocation identity.
A new physical campaign would be a new experiment and must not be presented as a
reproduction of the accepted P6 observations without a separately frozen protocol.

## Evidence rules

Manifests use `evidence_manifest_v2`, samples use `evidence_sample_v4`, sessions use `evidence_session_v1`, and reports use `evidence_report_v5`.
Simulator timing is never physical-performance evidence. Numerical-policy correctness is
separate from approximation error. Missing component timings are unavailable, not zero.
Speedup claims use compatible timing boundaries and matched controls.

For P6, evaluation data never enter fitting. Method aliases that select the same path
share one physical observation rather than being counted as independent samples.

## Layout

```text
src/quantum_bench/        Python implementation
native/                   QuEST and UPMEM native code
configs/                  experiment/workload definitions
scripts/                  qualification, analysis, and bounded study controllers
tests/                    software and protocol tests
docs/                     architecture/evidence research records
thesis_results/           tracked compact result/evidence packages
```

See `STATUS.md` for the final capability matrix and `docs/README.md` for the document
index.
