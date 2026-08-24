# UPMEM Provider Qualification Runbook

This is the M1 physical-qualification workflow. It qualifies one provider
probe at a time; it does not claim provider integration into the thesis
executor or a performance improvement.

## Local preparation

From `thesis/implementation`, run:

```bash
make upmem-provider-plan
```

This prepare-only command creates a unique plan under
`build/provider_qualification/<unique-id>/plan.json`, fingerprints the catalog,
source, patch, and toolchain, and does not build, allocate, or launch a DPU.

## ETH physical qualification

After committing the implementation and checking out that commit cleanly on
the ETH host, run:

```bash
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-provider-qualify PROVIDER=simplepim
```

The explicit opt-in is mandatory. Simulator selectors are rejected and there
is no simulator, CPU, or other fallback. A failed or blocked run is not
substitute evidence.

Canonical artifacts are under
`runs/evidence/provider_qualification/simplepim/<run>/`:

- `provider_qualification.json`: summary and admission status;
- `raw_runner_preflight.json`: preflight response;
- `raw_runner_result.json`: native runner response when emitted;
- `normalized_records.jsonl`: normalized qualification record; and
- `run_manifest.json`: source, tool, command, and artifact fingerprints.

Keep the complete run directory together and do not promote it as a general
benchmark result.

## SimplePIM probe and passed fields

The probe is one DPU, 12 configured tasklets, and a 256-element `uint32`
virtual-array map plus zip operation. It checks native execution and exact
output functionality only. It records logical bytes, but makes no simulator,
fallback, performance, scaling, or energy claim.

A passed canonical result requires, at minimum:

- `status: qualified`, `provider_id: simplepim`, and
  `probe_id: simplepim_va_map_zip_v1`;
- `hardware_preflight_verified: true`, physical target, and observed DPU
  count `1`;
- `configured_tasklets_per_dpu: 12`;
- `native_execution: true`, `validation_performed: true`, and
  `exact_validation: true`;
- `fallback_used: false`, `simulator_kernel_executed: false`,
  `release_status: released`, and canonical
  `resource_release_status: confirmed`;
- passed build/execution records, input/output hashes and sizes; and
- matching pinned source, runner, catalog, compiler, command, and tracked
  patch fingerprints. The patch must be staged and applied, with before,
  after, and staged hashes recorded.

The upstream SimplePIM map implementation has a `DPU_ASSERT` failure path
whose cleanup/release behavior is not fully controllable by this probe. If an
upstream assertion terminates the process, release is unknown/unconfirmed and
the result cannot pass qualification. Do not turn that limitation into a
successful result by rerunning through a fallback.

## M1 scope and status

PID-Comm is a separate qualification lane requiring UPMEM SDK `2021.3.0`,
AVX512, and 1024 DPUs. The official ATiM artifact and SparseP source remain
unpinned and blocked. All four providers remain central to later architecture
work, but only the SimplePIM harness and probe are implemented locally now.

Current M1 progress: harness and SimplePIM probe implemented locally;
physical qualification pending. M1 is not complete.
