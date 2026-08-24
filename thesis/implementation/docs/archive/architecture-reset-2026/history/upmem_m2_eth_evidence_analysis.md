# M2 ETH Physical Evidence Analysis

Audit date: 2026-08-02

Evidence source: developer-local `thesis/implementation/eth-evidence/`

Source revision: `53f0f9bc97b623a53a967eacc4c3605b0ac4814d`

Run ID: `2026-08-02_19-06-21`

## Verdict

The run is valid physical functionality evidence for the bounded M2 control
path. It proves that the repository allocated exactly two physical DPUs,
assigned one restricted contraction-index slice to each DPU, launched the DPU
set asynchronously, synchronized once, read both partial outputs, reconstructed
the final tensor on the host, and matched the CPU/reference outputs without a
simulator or CPU execution fallback.

It does not yet close the useful two-DPU work-distribution gate. Every second
slice produced an all-zero partial because each one-gate circuit starts in
`|0>` and the selected sliced index is the input-state index. DPU 1 executed a
valid package, but its arithmetic contribution was zero in all cases. The run
therefore demonstrates physical dispatch and reconstruction, not balanced
parallel contraction work, scaling, or speedup.

## Audited Evidence

| Check | Observed result |
| --- | --- |
| Route | `upmem_tn_hardware_sliced_resident_two_dpu` |
| Backend | `upmem_sdk_hardware_sliced_resident_two_dpu` |
| Source state | clean worktree at the recorded source revision |
| Workloads | one-qubit X, H, and Z |
| Warmups / measured repeats | one warmup and three measured repeats per case |
| Rows | 3 warmup rows and 9 measured rows |
| Allocation | 2 requested, 2 allocated, verified in all rows |
| Assignment | slice 0 -> DPU 0; slice 1 -> DPU 1 in all rows |
| Launch | one asynchronous DPU-set launch and one synchronization per row |
| Tasklets | one per DPU |
| Completion / release | all slices completed; release confirmed in all rows |
| Fallback | `cpu_fallback_used=false`; simulator execution false |
| Validation | all 9 measured rows passed CPU and expected-output checks |
| Maximum output error | `1.2101617041793133e-08` for H; zero for X and Z |
| Transfer invariant | 1,744 B H2D + 16 B D2H = 1,760 B in every row |
| Binary integrity | copied host and DPU binaries match their recorded SHA-256 hashes |
| Output determinism | output and scientific-plan hashes are stable across repeats |

The H2D volume is mostly fixed package/control traffic: 32 B of initial operand
payload, 1,664 B of descriptors, and 48 B of control data. Descriptor and
control bytes are 98.17% of the recorded H2D bytes for this tiny fixture. These
are application-visible SDK bytes, not physical DIMM traffic counters.

## Timing Observations

All values below are diagnostic wall-clock measurements in seconds. They are
not a speedup comparison and do not isolate DPU kernel time.

| Case | Median native elapsed | Median subprocess | Median reconstruction/validation | Median total route |
| --- | ---: | ---: | ---: | ---: |
| X | 0.055425 | 0.062249 | 0.020074 | 0.102944 |
| H | 0.056012 | 0.062589 | 0.020780 | 0.105179 |
| Z | 0.059182 | 0.065801 | 0.020656 | 0.106527 |

Across all nine rows, median native elapsed time is 0.056012 s, median
subprocess time is 0.062589 s, and median total route time is 0.105179 s. The
recorded reconstruction interval includes package and response validation,
file fingerprinting, partial-file loading, and the Python sum. It must not be
interpreted as pure arithmetic reconstruction time. Allocation, binary load,
H2D, kernel, synchronization, and D2H are not separately timed in this run.

## Partial-Output Finding

The first measured partials are representative of every repeat:

| Case | Slice 0 / DPU 0 | Slice 1 / DPU 1 | Reconstructed output |
| --- | --- | --- | --- |
| X | `[0, 1]` | `[0, 0]` | `[0, 1]` |
| H | `[0.70710677, 0.70710677]` | `[0, 0]` | `[0.70710677, 0.70710677]` |
| Z | `[1, 0]` | `[0, 0]` | `[1, 0]` |

This is mathematically expected for contraction-index slicing of a single gate
applied to the fixed initial state `[1, 0]`. It is not a native-code failure,
but it makes the current fixture unsuitable for evaluating load balance or the
benefit of two-DPU execution.

## Evidence-Semantics Defects

The native responses are internally consistent, but several normalized fields
fall back to generic defaults and contradict the route-specific evidence:

- `parallelism_mode=not_applicable` despite executed two-DPU slice dispatch;
- `slice_parallel_execution=false` despite an asynchronous DPU-set launch;
- `slicing_enabled=false` even though two contraction-index slices execute;
- `hardware_functionality_evidence=false` despite verified hardware execution;
- `timing_is_bringup_only=false` despite the MVP claim boundary;
- per-slice manifests retain the reused one-DPU backend/profile and timing-scope
  labels; and
- the profile says `synchronous_execution=true`, while the native operation is
  asynchronous launch followed by blocking synchronization.

These defects do not invalidate the native allocation, execution, output, or
hash evidence. They do make the normalized rows unsafe for generic slicing or
parallelism reports until corrected. Future records should use route-specific
fields and distinguish asynchronous device launch from a blocking host API.

The environment manifest also lacks the hostname, SDK version, rank/DPU
identity, and observed hardware topology. `run_manifest.json` reports
`upmem_sdk_available=unknown` despite successful physical execution. The local
bundle has no top-level checksum manifest, although the recorded binary and
per-package hashes that were checked are valid. These are provenance gaps for
future benchmark evidence, not blockers for this development result.

## Phase Evaluation

| Milestone | Status after this run |
| --- | --- |
| M0 execution contracts | Complete foundation |
| M1 external-provider qualification | Incomplete and remains a parallel lane |
| M2 physical two-DPU control path | Passed on ETH |
| M2 useful two-slice contraction | Not yet passed because slice 1 has zero useful contribution |
| M2 general TaskGraph scheduling/scaling | Not implemented |
| M3 operation-aware provider/kernel system | Next major architecture milestone after M2.1 |
| M4-M9 | Not started or only supported by earlier bounded primitives |

The important progress is physical rather than performance-related: the
repository now has an executed two-DPU ownership, launch, synchronization,
transfer, reconstruction, and cleanup path. The next experiment must make both
DPUs perform nonzero useful work before timing or planner calibration is
meaningful.

## Updated Near-Term Plan

### M2.1: balanced useful-slice acceptance

1. Extend the bounded fixture to a deterministic real quantum/TN case where
   both contraction-index slices have nonzero partial norms. Prefer the
   smallest multi-operation circuit and explicit dependent-prefix contract;
   do not hide prefix contraction on the CPU.
2. Keep exactly two DPUs and one tasklet per DPU. Preserve fixed slice ownership
   and host sum reconstruction so only workload usefulness changes.
3. Record per-slice output norm, nonzero element count, operation count, useful
   element/FLOP estimate, and completion. Reject the acceptance run unless both
   slices contribute to the result.
4. Correct the normalized slicing, parallelism, hardware-functionality, and
   bring-up fields. Give reused inner packages a truthful nested identity
   instead of presenting their one-DPU profile as the outer route identity.
5. Add stage timing where the SDK boundary supports it: allocation, binary
   load, H2D, launch/synchronize, D2H, package validation, and reconstruction.
   Preserve nulls where a boundary cannot be measured reliably.
6. Repeat the physical acceptance on ETH. The gate is correctness, two nonzero
   partial contributions, exact ownership/completion, no fallback, valid byte
   accounting, and clean release. It is still not a speedup experiment.

### M3: first operation-aware kernel vertical slice

After M2.1 passes, implement the deterministic operation classifier and
provider registry with explicit generic fallback. The first specialized route
remains a PIMutation-inspired permutation/row-swap kernel. Compare its output,
arithmetic skipped, bytes avoided, and diagnostic timing with the generic
contraction for the same operation. Keep SimplePIM qualification/integration as
the first external provider lane and continue PID-Comm, ATiM, and SparseP
qualification independently rather than blocking the classifier.

### M2 expansion in parallel

Extend fixed terminal slicing to a small multi-task graph and then to multiple
independent ready tasks. Do not call fixed slice ownership a general scheduler.
General placement, PID-Comm reduction, large-contraction distribution, and
strong/weak scaling remain M5-M7 work.

## Claims Allowed From This Run

- Two physical UPMEM DPUs were allocated and each executed one validated
  resident slice package.
- The DPU set was launched asynchronously and synchronized once.
- Both partial outputs were transferred and host reconstruction matched the
  CPU/reference result for the three bounded cases.
- No simulator or CPU execution fallback was used.

## Claims Not Allowed

- balanced or useful two-DPU parallel work;
- speedup, scaling, efficiency, or energy benefit;
- general multi-task TaskGraph execution;
- general complex or quantized multi-DPU execution;
- planner quality or hardware calibration; or
- superiority of slicing or UPMEM over CPU/GPU/TN baselines.
