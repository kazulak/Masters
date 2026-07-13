# Physical UPMEM Dense MVP

This is the Phase 1A hardware bring-up path. It proves only that the
thesis-owned L1 dense kernel can execute one deterministic int8 x int8 ->
int32 contraction on exactly one physical DPU and match the retained CPU
int32 reference. It does not measure speedup, energy, scaling, scheduling, or
general tensor-network performance.

The suite is [upmem_hardware_mvp.yml](../configs/suites/upmem_hardware_mvp.yml):
`dense_l1_2x2` runs first, followed only after success by `dense_l1_4x4`. Each
case has five sequential repetitions, one DPU, one tasklet, and a 30-second
timeout. The profile rejects any larger dimensions, complex input, dtype
change, tiling, simulator selection, or alternate DPU count.

## ETH Procedure

From the checked-out `thesis/implementation` directory, after creating the
user-local uv environment:

```bash
git rev-parse HEAD
git status --short thesis/implementation
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_upmem_hardware_mvp.py tests/test_dense_bridge.py -q
make PYTHON=.venv/bin/python upmem-hw-mvp-plan
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make PYTHON=.venv/bin/python upmem-hw-mvp
```

`upmem-hw-mvp-plan` resolves the suite, writes deterministic input manifests,
and builds an isolated native copy. It never allocates or launches a DPU.
`upmem-hw-mvp` requires the explicit environment opt-in and creates a full
evidence run beneath:

```text
runs/evidence/upmem_hardware_mvp/upmem_hw_dense/<timestamp>/
```

The run keeps the resolved suite, environment, input/output manifests, native
source snapshot, build output, bounded stdout/stderr, exact CPU reference, and
normalized rows. It is never promoted automatically to `thesis_results/current`.

## Required Success Fields

Every successful normalized row must have:

```text
target_requested=hardware
target_observed=hardware
backend_id=upmem_sdk_hardware_dense
hardware_profile_version=hardware_mvp_l1_v1
requested_dpu_count=1
allocated_dpu_count=1
tasklets_per_dpu=1
hardware_allocation_verified=true
hardware_kernel_executed=true
simulator_kernel_executed=false
cpu_fallback_used=false
exact_integer_match=true
validation_status=passed
hardware_speedup_applicable=false
```

The Make target runs `check-upmem-hardware` after execution. A failed row stays
failed and records its `failure_stage`; the command never retries through the
simulator, NumPy, mocks, or another backend.

## Failure Collection And Reruns

Inspect the retained per-repeat bridge directory first:

```bash
RUN=runs/evidence/upmem_hardware_mvp/upmem_hw_dense/latest
find "$RUN" -maxdepth 6 -type f | sort
sed -n '1,220p' "$RUN"/upmem_hardware_mvp_summary.json
sed -n '1,260p' "$RUN"/cases/dense_l1_2x2/repeat_00/bridge/output_manifest.json
```

Use the exact failing `failure_stage`, output manifest, native status sidecar,
and bounded stderr for diagnosis. Do not change profile limits, add retries, or
switch to the generic loop while debugging the dense MVP. Rerun the same
explicit command after fixing the stated issue.

Only after all 2x2 and 4x4 repetitions pass may the next wave enable one
separate tiny generic-loop physical case.
