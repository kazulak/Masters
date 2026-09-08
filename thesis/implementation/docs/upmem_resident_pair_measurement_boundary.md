# P4 Test-Only Pair Measurement Boundary

This patch prepares an isolated measurement path. It is not hardware admission,
a frozen experiment, a production backend, or evidence of whole-DAG residency.
No production source, DPU code, or private protocol header changes are required.
The default host invocation remains the SDK-simulator diagnostic protocol with
full-arena replies; replies are now explicitly flushed, without changing bytes.

## Fixed Contrast

Use only the retained Stress16 greedy `contract_121 -> contract_122` pair,
one DPU, one rank, eight tasklets, float32 four-product panel, right resident
operand, and no int8. Each contraction uses one four-product launch; the pair
is not fused into one launch. Producer geometry is `(16,64,4)`, consumer
geometry `(16,256,4)`, with 93,184 live MRAM bytes and no span recycling.

The shared `selected_case()` CPU oracle uses the existing corpus test's retained
path and plan checks, not a new path search:

- Candidate pool SHA256: `d95150ddf89f6aafa861000b0db2d8447d64456a035c5404463a878c3a319049`.
- Candidate: `31a9997e87f38005e081aad56952a30bd31277dafe425c44de78613365a45e02`.
- Physical plan: `90b181769e4c2c061f2b44e7fa5f61df39ac0a9354b86ad38f6b0db65c4bb3df`.

`tests.upmem_resident_probe_client.measure_selected(host, binary, arm, ...)`
prepares this CPU oracle before starting a subprocess. Each invocation opens one
session and makes exactly two synchronous launches. There is no retry or
management-init launch. `host_roundtrip` reads actual producer products and
reconstructs the second request; `resident` reads producer completion only.
Both read final consumer completion and the same four consumer product planes.
Both require exact CPU-oracle product bytes and finite float32 reconstruction.
Oracle byte comparisons occur only after host reaping. Only the actual producer
float32 reconstruction needed to form the second request stays inside the host
roundtrip pair interval; oracle comparison is not attributed as residency savings.
Positive-zero assembly precedes RR-II / RI+IR in the existing decoder, including
its signed-zero policy. Full-state injection remains in the existing diagnostic
corpus test, not an additional native launch or readback in measurement.

| Arm | H2D bytes / calls | D2H bytes / calls | Launches |
| --- | ---: | ---: | ---: |
| Resident | 93,832 / 5 | 65,680 / 3 | 2 |
| Host roundtrip | 102,024 / 6 | 82,064 / 4 | 2 |

These are successful SDK transfer payloads, including both plans, commands and
completions, not bus traffic or Python pipe bytes. The 24,576-byte difference
is producer products plus the retained operand upload. No full-arena diagnostic
readback occurs in measurement mode. Binary and output hashes accompany results.

## Timing and Lifecycle

`launch_wall_s` measures synchronous SDK launch calls, **not kernel time**.
`pair_s` includes the interactive producer reply and host reconstruction but
ends after the final reply flush, before the final EOF wait. Therefore
`session_open_s + pair_s + session_close_s` is **not** the complete attempt.
Host `attempt_wall_s` includes final EOF handling and resource release, but not
its final report write. Client `subprocess_wall_s` spans spawn, transfer,
actual output decoding, EOF and reaping; `validation_s` separately covers
post-reap oracle comparisons and terminal metadata validation. `client_total_s`
includes both intervals. CPU-oracle preparation and binary hashing precede these
client timings. Simulator durations are explicitly diagnostic only.

Measurement mode ignores SIGPIPE so write/flush failures reach cleanup. INT/TERM
use `sigaction` without `SA_RESTART` to interrupt blocked stdio. The client uses
one deadline and exact partial I/O, closes input on failure, signals only its
owned process group, and reaps it. Forced SIGKILL after the cleanup grace period
does not prove SDK release and never produces an accepted result.
Failures re-raise the original exception (including cancellation) with a
JSON-serializable `resident_probe_failure` attribute for the outer collector.
It retains the failure stage, return code, elapsed wall, the last 64 KiB of
native stderr with a truncation flag, and any available native metrics. Those
metrics are explicitly unvalidated, not an acceptance or release assertion.
Missing metrics, cleanup errors and SIGKILL remain failures; there is no retry.

Physical mode requires explicit `physical=True`, environment
`UPMEM_ALLOW_PHYSICAL_HARDWARE=1`, and exactly `/dev/dpu_rank1`. Conflicting
backend selectors are rejected. The caller must also pass the controller's
already exclusively locked `lock_fd`. The client verifies the descriptor's
device/inode against `/home/tkazulak/evidence/upmem-experiment.lock`, private
regular-file ownership, and Linux fdinfo's whole-file exclusive flock on that
same open file description. It neither creates nor acquires/unlocks this lock.
`pass_fds` retains the descriptor in the native host despite the new process
group: client SIGKILL cannot release the last lock reference while the host is
alive. The controller must retain its normal close-only lock lifetime, never
explicitly unlock the shared description while native work is live.
The host checks the character device, actual
allocated DPU/rank counts, rank enumeration and ownership, loaded symbol sizes,
successful T8-correlated completions, and ownership release after `dpu_free`.
It uses the production SDK allocation/profile pattern without modifying it.
There is no hardware-to-simulator fallback. These checks do not replace fresh
external admission, ownership locking, binary qualification or preregistration.

## Qualification Limit

The focused lifecycle suite compiles the host only against a local in-memory
`dpu.h` stub: no SDK headers/libraries, DPU compiler, devices or native numeric
execution. Stub products are zero-filled mock observations, never performance
or corpus evidence. Pipe closure, blocked partial reads, final EOF interrupts,
release, exact accounting, client cleanup, and CPU reconstruction are tested.
The mock-client SIGKILL test also proves lock exclusion while the inherited host
is alive and availability after host cleanup/exit, with both processes reaped.
The existing corpus module additionally checks both actual SDK-simulator
`--measure` arms against the same fixed CPU product/decoded-output hashes and
exact resource/transfer/launch/release metrics, without timing thresholds.
Existing SDK diagnostic parity tests are preserved. Qualification logs and
JUnit records distinguish these from mock coverage and report full-suite
dependency failures without exclusions. Any physical measurement still needs
separate admission. No P4 gate or schedule is frozen.

Local qualification on 2026-09-08 passed 1,726 tests with zero failures, errors,
or skips after restoring the pinned SimplePIM/QuEST inputs and rebuilding
`quest_runner`. All 159 earlier dependency-related failures/errors disappeared
without changing test IDs or weakening tests. The 45 focused tests include both
real SDK measurement arms. Project CI-scoped Ruff and `git diff --check` passed;
the restored upstream SimplePIM source is not included in the project's Ruff
scope. This environment uses SDK 2025.1.0, not the target ETH SDK 2023.1.0;
target-toolchain qualification and physical admission remain separate gates.
The independent failure-evidence review is resolved. No physical probe result
or performance claim follows from these software/simulator checks.
