# M5.5 Whole-Circuit UPMEM Runbook

Status: additive implementation baseline; local implementation and
hardware-free validation are in progress. Physical ETH acceptance is pending.

M5.5 exercises the complete research path:

```text
circuit -> tensor network -> planner -> hashed TaskGraph
        -> NumPy same-plan CPU reference or physical UPMEM v4 engine
        -> normalized records and report
```

The physical route is intentionally bounded. It uses output/K-tiled requests,
one persistent v4 session per selected rank, one bulk synchronous set launch
per request, unique per-DPU descriptors, deterministic sequential TaskGraph
task order, and concurrent submission of separate rank sessions within one
task. Float32 requests use float32 operands. Host-packed int8 requests quantize
each task operand once on the host, transfer packed int8 payloads, and perform
int8 x int8 -> int32 MACs on the DPU, int64 reduction of K-partials on the
host, and one dequantization per task output.

Whole-graph intermediate tensors are currently owned by the Python host tensor
store and re-uploaded for downstream contractions. This runbook therefore does
not claim graph-wide DPU residency, PID-Comm, SimplePIM compute, ATiM,
SparseP, energy efficiency, or calibrated planning.

## 1. Prepare A Clean Checkout

From `thesis/implementation` on the local machine or ETH server:

```bash
make setup
make doctor
make test
```

The project uses the repository-managed parent environment (`../.venv`) through
the Makefile. Do not substitute a system Python for this workflow.

Build and plan the smoke profile without allocating or launching a DPU:

```bash
make m5-circuit-plan
```

This invokes the current CLI as:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python \
  -m quantum_bench.bench m5-circuit-study \
  --suite configs/suites/m5_circuit_smoke.yml --prepare-only --build
```

The plan artifact is written below `build/m5_circuit_study_plan/`.

## 2. Select Physical Ranks

Physical execution fails closed unless both conditions hold:

```bash
export UPMEM_ALLOW_PHYSICAL_HARDWARE=1
export UPMEM_HW_RANK_PATH=/dev/dpu_rank1
```

Use `UPMEM_HW_RANK_PATH` for one-rank profiles. Use
`UPMEM_HW_RANK_PATHS` for a profile that includes multiple ranks:

```bash
export UPMEM_HW_RANK_PATHS=/dev/dpu_rank1,/dev/dpu_rank20
unset UPMEM_HW_RANK_PATH
```

Paths must be explicit unique `/dev/dpu_rankN` paths. The software does not
discover a rank, substitute a simulator, or fall back to CPU execution.
Choose healthy ranks on the ETH host before running the study.

## 3. Staged Physical Acceptance

Run the stages in order and retain each generated directory under ignored
`runs/evidence/`.

### Smoke

The smoke suite has one BV 4q case, one planner, both numeric policies, one
CPU row, and one physical one-DPU topology. It is a functionality and schema
gate, not a performance result.

```bash
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
make m5-circuit-smoke
```

### Canonical whole-circuit grid

The canonical suite contains QRNG, BV, XOR, BB84, EDC, and HS at 8, 10, 12,
14, 16, 18, and 20 qubits: 42 circuit cases. It compares opt_einsum greedy and
deterministic cotengra FLOP-greedy planner variants, float32 and host-packed
int8 policies, the same-plan CPU reference, and the physical one-rank 8-DPU
route. The older custom UPMEM v2 planner remains a separate modeled diagnostic:
its single-DPU rank cap does not describe the v4 tiled execution policy.

First prepare the exact suite without allocating hardware:

```bash
M5_CIRCUIT_PLAN_SUITE=configs/suites/m5_circuit_canonical.yml \
make m5-circuit-plan
```

Then execute it on one selected rank:

```bash
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
M5_CIRCUIT_SUITE=configs/suites/m5_circuit_canonical.yml \
make m5-circuit-study
```

The canonical profile has one warmup and three measured repetitions. Its
physical rows are eligible for same-plan timing comparison only when the
recorded admission fields say so.

Each native response covers one bulk request and therefore records
`request_level_speedup_applicable=false`. Performance admission is decided only
for the repeated, validated whole-circuit study row; request timing alone is
never reported as circuit speedup.

### Scaling profile

The scaling suite uses EDC 20q, one planner, both numeric policies, and
one-rank 1/2/4/8/16/32/64-DPU variants plus a two-rank 128-DPU variant. Supply
two explicit healthy ranks because the suite contains a two-rank variant:

```bash
M5_CIRCUIT_PLAN_SUITE=configs/suites/m5_circuit_scaling.yml \
make m5-circuit-plan

UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
UPMEM_HW_RANK_PATHS=/dev/dpu_rank1,/dev/dpu_rank20 \
M5_CIRCUIT_SUITE=configs/suites/m5_circuit_scaling.yml \
make m5-circuit-study
```

The suite has one warmup and five measured repetitions. The report must keep
one-rank and two-rank topology series distinct.

### Large boundary profile

The large profile contains BV, EDC, and HS at 22, 24, 26, 28, and 30 qubits,
with both planner and numeric variants, and a one-rank 64-DPU physical route.
It is primarily a support-boundary and runtime study. Cases exceeding the
configured live/output limits must be retained as explicit unsupported rows.

```bash
M5_CIRCUIT_PLAN_SUITE=configs/suites/m5_circuit_large.yml \
make m5-circuit-plan

UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
M5_CIRCUIT_SUITE=configs/suites/m5_circuit_large.yml \
make m5-circuit-study
```

Do not turn a large-profile unsupported row into a CPU fallback or omit it
from the report.

## 4. Report A Completed Run

Each execution prints its run directory and writes
`m5_circuit_study_summary.json`, `normalized_records.jsonl`, the resolved
configuration, and plan metadata below `runs/evidence/`. Generate the report
without rerunning hardware:

```bash
M5_CIRCUIT_RUN=runs/evidence/m5_circuit_study/m5_circuit_study/<timestamp> \
M5_CIRCUIT_REPORT=runs/comparisons/m5_circuit_study/<timestamp> \
make m5-circuit-report
```

The direct equivalent is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python \
  -m quantum_bench.bench m5-circuit-report \
  --input runs/evidence/m5_circuit_study/m5_circuit_study/<timestamp> \
  --output runs/comparisons/m5_circuit_study/<timestamp>
```

Inspect `tables/`, `plots/`, and `plot_manifest.json`. The report retains
cross-algorithm rows for context but only admits same-plan CPU/UPMEM ratios
when physical execution, validation, release, timing scope, repetitions, and
all identity hashes satisfy the report contract. Energy remains an explicit
TODO because no sensor source is currently recorded.

## 5. Required Success Fields

For a physical completed row, check at least:

```text
status = completed
target_requested = upmem
target_observed = physical_hardware
hardware_allocation_verified = true
native_kernel_executed = true
hardware_kernel_executed = true
hardware_release_verified = true
exact_once = true
scientific_validation_status = passed
no_fallback_used = true
simulator_kernel_executed = false
cpu_fallback_used = false
```

For a timing comparison, also require `hardware_speedup_applicable=true`,
`timing_is_bringup_only=false`, the configured repeated measurements, and
matching circuit, tensor-network, contraction-plan, path, numeric-policy, and
timing-scope identities with the CPU row. A physically successful row that
does not meet these conditions remains functionality evidence.

## 6. Failure Collection

On failure, keep the entire run directory, including normalized records,
manifest files, resolved suite, stdout/stderr, and native/build artifacts that
the current retention policy writes. Record the printed `run_dir` before
starting another attempt. For transfer from ETH, archive the directory without
editing its contents:

```bash
tar -czf /tmp/m5_circuit_<timestamp>.tar.gz \
  runs/evidence/m5_circuit_study/m5_circuit_study/<timestamp>
```

Copy that archive into the local ignored inbox under `runs/inbox/eth/m5_5/` and
report it with `M5_CIRCUIT_RUN=<copied-run-directory> make m5-circuit-report`.
Do not retry a failed physical row through the simulator or CPU.

## 7. Claim Boundary

M5.5 can establish that a whole circuit was lowered, planned, executed through
the selected engine, validated, and recorded with stable identities. It can
compare numeric policies on the same plan and compare planner/path variants for
the same circuit and tensor network when the rows are admitted. It does not yet
establish UPMEM speedup over CPU or GPU, energy efficiency, graph-wide
DPU-resident execution, parallel execution of independent TaskGraph tasks,
SimplePIM compute, PID-Comm communication, ATiM/SparseP kernels, multi-DIMM
scalability, or hardware-calibrated planner superiority. Those require later
architecture components and a separate final benchmark freeze.
