# Physical UPMEM Generic TaskGraph MVP

This Phase 1B command proves one thesis-owned generic-loop contraction on one
physical UPMEM DPU. It is functionality evidence only. It does not establish
general quantum-circuit execution, speedup, energy efficiency, scaling,
multi-DPU scheduling, SimplePIM integration, or path-planner quality.

The fixed suite is
[upmem_hardware_generic_mvp.yml](../configs/suites/upmem_hardware_generic_mvp.yml).
It executes the synthetic real-valued TaskGraph node:

```text
A[a,b,c] x B[c,d,e] -> C[a,b,d,e]
```

Both operands are deterministic `2x2x2` int8 tensors. The output has 16 int32
elements and is written in two eight-element output tiles. The profile fixes
one DPU, one tasklet, synchronous launch, five sequential repeats, a
30-second timeout, and the native SDK allocation literal `backend=hw`.

## ETH Procedure

From `thesis/implementation`, after cloning or pulling the repository:

```bash
make setup

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest \
  tests/test_upmem_hardware_generic_mvp.py \
  tests/test_upmem_sdk_generic_loop_runner.py \
  tests/test_generic_bridge.py -q

make upmem-hw-generic-plan
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-generic-mvp
```

`make setup` requires `uv` and creates or reuses the sibling `../.venv` using
the repository's pinned Python version. It never runs GPU or UPMEM hardware.
The generic plan command writes deterministic manifests and can compile an
isolated source copy, but does not allocate or launch a DPU.

The execution command creates evidence under:

```text
runs/evidence/upmem_hardware_generic_mvp/upmem_hw_generic/<timestamp>/
```

## Required Success Fields

Every row must record:

```text
route_id=upmem_tn_hardware_generic_loop_mvp
backend_id=upmem_sdk_hardware_generic_loop
target_requested=hardware
target_observed=hardware
requested_dpu_count=1
allocated_dpu_count=1
tasklets_per_dpu=1
hardware_kernel_executed=true
simulator_kernel_executed=false
cpu_fallback_used=false
synthetic_real_taskgraph_mvp=true
not_real_quantum_circuit=true
generic_output_tile_elements=8
generic_output_tile_count=2
exact_integer_match=true
validation_status=passed
hardware_speedup_applicable=false
```

The retained `host_status.json` must show `allocation_profile=backend=hw`, one
requested and allocated DPU, one tasklet, `success=true`, and no failure stage.
The host-side allocation/load/H2D/launch/D2H timings are bring-up diagnostics,
not a kernel benchmark.

## Failure And Rerun

Inspect the first failed repeat before changing anything:

```bash
RUN=runs/evidence/upmem_hardware_generic_mvp/upmem_hw_generic/latest
sed -n '1,260p' "$RUN"/upmem_hardware_generic_mvp_summary.json
find "$RUN"/cases -name output_manifest.json -print
sed -n '1,320p' "$RUN"/cases/generic_real_abc_cde_2/repeat_00/bridge/output_manifest.json
```

Use its `failure_stage`, native status sidecar, and bounded stderr. Do not set
`DPU_BACKEND`, `UPMEM_PROFILE`, or `UPMEM_PROFILE_BASE` to force a backend.
Do not add retries or fallback. Rerun the same command after correcting the
specific failure.

Keep successful generic MVP evidence separate from `thesis_results/current`
until it has been independently audited and promoted as a dedicated compact
physical-hardware capsule.
