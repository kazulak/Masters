# ETH UPMEM TaskGraph Hardware Runbook

This route is correctness-only evidence for a bounded circuit-derived
TaskGraph on one physical DPU. It does not establish speedup, energy
efficiency, scaling, multi-DPU scheduling, or general hardware performance.

From `thesis/implementation` on the ETH hardware host:

```bash
make setup
make doctor
make upmem-hw-taskgraph-plan
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-taskgraph
make upmem-hw-taskgraph-report
```

The plan resolves the fixed suite, writes manifests, and optionally builds the
isolated native source. It must not allocate or launch a DPU. Execution
requires the explicit opt-in above and has no simulator or CPU fallback.
The report is derived from the saved normalized records. It produces the
physical TaskGraph validation plot plus float32/int8 transfer and error plots;
the runtime comparison remains a visible TODO because the first route creates
one allocation/load/release session per logical TaskGraph contraction.

If a run reports `failure_stage=kernel_timeout`, do not rerun it blindly: the
host process was terminated before release could be confirmed. Retain the
artifacts, inspect the site-specific UPMEM allocation state or contact the
hardware administrator, then start a new run only after the resource is known
to be free. The route records this condition as release-unverified rather than
claiming cleanup succeeded.

Artifacts are retained below:

```text
build/upmem_hardware_taskgraph_plan/<timestamp>/
runs/evidence/upmem_hardware_taskgraph/<route>/<timestamp>/
```

Keep the resolved suite, plan/run summaries, environment manifest, native
status, bounded stdout/stderr, per-task manifests, CPU references, and
normalized rows together. Treat all timings as bring-up diagnostics. The only
claim supported by this route is physical TaskGraph correctness under the
fixed single-DPU profile, with float32/int8 transfer and numerical-error
attribution; it is not a speedup result.
