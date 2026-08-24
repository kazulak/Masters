# M4.1 physical differential run

M4.1 is a small control-plane integration test. It executes the unchanged
M3.1 three-task/two-wave graph through the frozen raw SDK route and through a
SimplePIM-management-assisted allocation route. The contraction kernel and
custom package transfers remain thesis-owned. The route is functionality and
bring-up evidence only: it makes no speedup, scaling, concurrency, or energy
claim.

## Prepare

From a clean checkout with the project environment installed:

```bash
make doctor
make upmem-hw-m4-1-plan
```

Preparation/build copies the native source into `build/`, checks the pinned
SimplePIM commit, stages the management extension, and builds both provider
binaries. It never allocates a DPU.

## Execute on ETH hardware

```bash
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m4-1
```

The command must report ten measured rows: five raw SDK rows and five
SimplePIM-management rows, plus one warm-up row for each provider. The run is
accepted only when both providers allocate two physical DPUs per request,
execute all three tasks in two waves, validate against the CPU same-plan
reference, agree with each other, and release the allocation.

## Inspect and retain

```bash
UPMEM_HW_TASKGRAPH_M4_1_RUN=runs/evidence/upmem_hardware_taskgraph_m4_1/upmem_hw_taskgraph_m4_1/latest \
  make upmem-hw-m4-1-report
```

Keep the raw evidence directory outside Git or under the ignored ETH inbox.
The important files are `normalized_records.jsonl`, the summary, resolved
suite/profile, environment, native session copy, stage marker, and response
manifests. Do not promote this run to a thesis performance snapshot.

## Interpretation

The result demonstrates that SimplePIM management can participate in physical
allocation for this fixed route. It does not demonstrate SimplePIM operator
execution, SimplePIM kernels, persistent allocation, multi-DPU scaling, or
performance improvement. Those remain the next M4 work items.
