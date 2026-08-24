# KISS M3.1 Frontier Runbook

M3.1 is a strict physical-hardware functionality fixture for the committed
`one_qubit_ry_h_ry_a.qasm` circuit. It contains one `opt_einsum` greedy workload,
`numeric_mode: none`, one warmup, five measured repeats, two DPUs, and one
tasklet per DPU.

The suite is fail-fast. A failed warmup prevents all measured requests; a
failed measured request stops the measured sequence immediately. Artifacts
contain only requests that were actually attempted; no missing request rows
are synthesized.

## Prepare

Preparation loads the committed suite, parses the actual QASM, builds the real
tensor network and task graph, computes the CPU frontier reference, creates the
frontier plan and resident package, and optionally builds the native binaries.
It never allocates or launches a DPU. When `--build` is supplied, the native
host runs its validate-only package check; that check must report no allocation,
launch, or release attempt.

```sh
cd thesis/implementation
make upmem-hw-frontier-m3-1-plan
```

The plan artifact is written below
`build/upmem_hardware_frontier_m3_1_plan/`.

## Execute

Physical execution requires an explicit opt-in and an unset `DPU_BACKEND`:

```sh
cd thesis/implementation
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-frontier-m3-1
```

The command builds once, then issues one distinct warmup request and five
distinct measured requests on a successful run. Native response evidence, output-file bindings,
the native transfer invariant, and CPU-reference accuracy at absolute
tolerance `1e-6` are required for every completed request. Failures preserve
the exact native failure stage, response context, and response artifact; there
are no retries, CPU fallbacks, simulator fallbacks, or alternate routes.

## Evidence boundary

Successful rows derive three source tasks, three physical task instances, two
waves and two barriers, and DPU task counts `[2, 1]` from validated native
response fields. Failed rows retain those values as expected counts only;
observed, executed, and completed counts are null and parallelism evidence is
`not_observed` when execution was not validated.
Overlap is explicitly unmeasured. Transfer accounting is native SDK observed
and invariant-checked. Timing is host-observed with kernel timing unavailable.
The evidence supports hardware functionality and host-observed timing only. It
does not support speedup, scaling, concurrency, or energy claims.

Measured rows are in `normalized_records.jsonl`; the separate warmup row is in
`warmups.jsonl`. No reporter, plot, or thesis snapshot is generated.
