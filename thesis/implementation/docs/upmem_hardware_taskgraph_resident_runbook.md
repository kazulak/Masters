# UPMEM MRAM-Resident TaskGraph Runbook

This runbook covers the additive Phase 1B route:

- route: `upmem_tn_hardware_taskgraph_resident`
- backend: `upmem_sdk_hardware_taskgraph_resident`
- profile: `hardware_taskgraph_single_dpu_mram_resident_v1`
- protocol: `generic_loop_resident_graph_session_v1`
- timing scope: `one_dpu_mram_resident_full_taskgraph_v1`

Legacy dense, generic/persistent, and taskgraph-study routes are historical
artifacts. This resident route is the only physical TaskGraph command exposed
by the current CLI and Makefile.

Acceptance is manual on the ETH hardware host and is intentionally outside
pytest and CI. Historical evidence remains reportable but is never re-run by
the active test gate.

## Prepare

From the implementation root:

```text
PYTHONPATH=src python3 -m quantum_bench.bench \
  upmem-hardware-taskgraph-resident \
  --suite configs/suites/upmem_hardware_taskgraph_resident_path_quantization.yml \
  --prepare-only --build
```

Preparation plans both path variants for all 13 cases, creates a deterministic
topological lifetime map, validates the binary graph package, and builds the
resident-only host/DPU pair. Preparation never allocates a DPU.

## Execute

Physical execution is deliberately opt-in and does not accept simulator
configuration:

```text
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 DPU_BACKEND= \
  PYTHONPATH=src python3 -m quantum_bench.bench \
  upmem-hardware-taskgraph-resident \
  --suite configs/suites/upmem_hardware_taskgraph_resident_path_quantization.yml \
  --execute
```

The native host parses and validates one complete graph package before DPU
allocation, transfers initial slots once, launches ordered descriptors
synchronously, and transfers only final output components. It releases the DPU
on every post-allocation path; a timeout reports release as unconfirmed.

## Profile And Evidence

The bounded profile is one DPU, one tasklet, rank at most 16, tensors at most
256 elements, 32 logical tasks, 128 component operations, 128 slot descriptors,
and a 512 KiB MRAM pool. A capacity miss is a structured
`hardware_profile_violation`; there is no host spill or CPU fallback.

Resident slots are canonical float32. `per_task_resident_requantize` computes
each operand scale as `max_abs / 127`, or `1` for an all-zero operand, then uses
explicit nearest-even rounding, clips to `[-127, 127]`, computes int8 by int8
to int32, and dequantizes to the float32 output slot. Complex work uses real
and imaginary slots, four contractions, and a DPU combine:
`ar_br - ai_bi` and `ar_bi + ai_br`.

Evidence records package-parse timing separately from allocation/upload,
actual aligned initial/descriptor/control/final transfers, zero intermediate
H2D/D2H, modeled legacy host-rehydrated-equivalent bytes, slot/lifetime data,
launch/task counts, numeric scales/saturation, output hashes, and separate
resident-policy and full-precision accuracy statuses. The route makes no
speedup, energy, scheduler, or multi-DPU claim.
