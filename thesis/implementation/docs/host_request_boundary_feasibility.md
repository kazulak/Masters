# Host Request Boundary Reduction Feasibility v1

## Decision

The host-only feasibility gate is **GO for a bounded packed-operation
prototype**. It is not a physical performance result and it does not change
the production execution path.

The prototype preserves ABI-v4 sidecar bytes, payload bytes, request order and
work-unit order while representing one contraction operation as one validated
`UPOENV1` envelope. Across the six modeled cells, the probe reduced the
synthetic preparation/transport time by 2.75x to 9.20x. The measured
operation boundary changed as follows:

```text
current path: one request directory and file set per embedded request
candidate:    one packed operation envelope
```

The result justifies a separate, bounded production feasibility task against
the existing persistent C host process. It does not justify a simulator-wide C
rewrite, a new scheduler, an ABI change, or a physical speedup claim.

## Scope and source

The inventory is source-only and uses the accepted request-template baseline.
The host-only probe uses deterministic synthetic ABI-v4 float32 requests with
the same sidecar and payload layout, request ordering, and one- or four-DPU
topologies represented by the inventory. It does not allocate a DPU, load a
kernel, or execute a circuit.

The probe is intentionally an executable boundary test rather than a new
runtime. Its packed format is private to the probe until a production
implementation is separately approved.

The six-cell probe fixture uses source-derived aggregate contraction, wave and
request counts. It does not replay every real contraction node's exact tile
distribution. The fixture therefore establishes boundary mechanics and
operation-count sensitivity, not a physical workload prediction.

## Current boundary inventory

The current Python route prepares four sequential real-product passes per
active rank wave. The native host process is persistent for one rank session;
it is not started once per request. Each submitted request contains one dense
ABI record per configured DPU, including zero-work records when required. The
active logical work-unit count is therefore reported separately.

| Circuit | DPUs | Contractions | Active work units | Waves | Embedded requests | Dense ABI records | Request dirs | Payload files | Metadata files | Total request files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stress18 | 1 | 141 | 222 | 222 | 888 | 888 | 888 | 1776 | 1776 | 3552 |
| Stress18 | 4 | 141 | 222 | 159 | 636 | 2544 | 636 | 5088 | 1272 | 6360 |
| HS18 | 1 | 53 | 56 | 56 | 224 | 224 | 224 | 448 | 448 | 896 |
| HS18 | 4 | 53 | 56 | 53 | 212 | 848 | 212 | 1696 | 424 | 2120 |
| GHZ18 | 1 | 35 | 371 | 371 | 1484 | 1484 | 1484 | 2968 | 2968 | 5936 |
| GHZ18 | 4 | 35 | 371 | 116 | 464 | 1856 | 464 | 3712 | 928 | 4640 |

The current Python client has one `V4Session.submit` call site. The proposed
packed boundary estimates one operation-level submission per contraction,
which is an estimate for a future implementation, not a current runtime
fact.

## Call-graph boundary

One current physical request follows this path:

```text
UpmemEngine._submit_wave
  -> _build_work_unit
  -> build_v4_request
       -> write manifest, sidecar, A/B payload files
  -> V4Session.submit
       -> validate manifest, sidecar and payload hashes
       -> write `SUBMIT <manifest> <sha256>` to the persistent host process
  -> native host execute_request
       -> request.c loads manifest, sidecar and payload files
       -> SDK H2D copies
       -> synchronous DPU launch
       -> SDK D2H copies
       -> response and output files
  -> Python response validation and reconstruction
```

Planning, lowering, contraction-path selection, `ContractionDAG` creation and
`UpmemPlan` construction remain Python-owned. The probe only tests replacing
the repeated request-artifact boundary with one operation envelope. It does
not move scientific planning into C.

## Host-only equivalence

The probe checks all of the following for every measured cell:

```text
Python-built sidecar bytes == directly prepared sidecar bytes
Python-built payload bytes == directly prepared payload bytes
request order is deterministic
work-unit order is deterministic
packed envelope is deterministic
C validation summary == independent Python summary
```

The C validator also rejects truncated, overlapping, reordered, invalid-count
and digest-corrupted envelopes. The existing DPU ABI remains unchanged.

## Host-only measurements

The run used two warmup iterations and ten measured iterations per cell. The
timed arms were host-only preparation and transport probes, not
`steady_execution_v1` and not session-inclusive UPMEM attempts.

| Circuit | DPUs | Current median (s) | Packed median (s) | Boundary speedup | Current files | Packed files | Current bytes | Packed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stress18 | 1 | 0.003974 | 0.001102 | 3.61x | 28 | 1 | 4431 | 3260 |
| Stress18 | 4 | 0.005463 | 0.001349 | 4.05x | 50 | 1 | 9450 | 4336 |
| HS18 | 1 | 0.002805 | 0.001021 | 2.75x | 20 | 1 | 3165 | 2356 |
| HS18 | 4 | 0.004370 | 0.001227 | 3.56x | 40 | 1 | 7560 | 3488 |
| GHZ18 | 1 | 0.023603 | 0.002566 | 9.20x | 172 | 1 | 27219 | 19532 |
| GHZ18 | 4 | 0.015301 | 0.002273 | 6.73x | 140 | 1 | 26460 | 11968 |

These values are descriptive results from the local host-only run. They are
not whole-route UPMEM timings. The packed arm includes one subprocess
invocation to validate the envelope, so the observed reduction is conservative
with respect to the proposed persistent host integration but is not a model
of SDK wait or DPU execution.

## Cost model and next gate

The relevant host model is:

```text
T_host = N_process * C_process
       + N_request * C_request
       + N_file * C_file
       + bytes_copied * C_byte
       + T_sdk_wait
```

The probe demonstrates a reduction in `N_request`, `N_file`, and repeated
metadata handling. It does not measure `T_sdk_wait`, DPU transfer time, kernel
time, or Python reconstruction time.

The next production feasibility task must therefore verify, before any
physical A/B campaign:

```text
the existing persistent C host can consume one envelope per operation
outputs and output ownership remain equivalent
session-inclusive time does not merely move work outside steady timing
resource-general cases remain valid
malformed envelopes fail closed
the implementation does not change ABI-v4 or the DPU kernel
```

Three boundary choices were considered. Extending the existing persistent C
host process is the preferred production direction because it preserves SDK
ownership and process isolation. A shared-library FFI is deferred because it
adds pointer-lifetime, failure-isolation and build concerns before the actual
boundary benefit is measured. Keeping the current boundary remains the
fallback if a real host integration cannot preserve outputs and lifecycle
semantics. No alternative is implemented by this feasibility commit.

For a prior measured affected fraction `p` and host-only speedup `s`, the
whole-route upper bound is:

```text
S_max = 1 / ((1 - p) + p / s)
```

The physical decision must use the observed `session_open_s`,
`steady_execution_v1`, `session_close_s`, session-inclusive attempt time and
kernel time. Moving envelope preparation into session opening is not an
execution-time reduction unless it is explicitly amortized over a persistent
session and the break-even count is reported.

## Explicit non-goals

This milestone does not implement:

```text
production packed-envelope execution
new Python FFI
new C runtime
DPU ABI or kernel changes
asynchronous overlap
request batching in the physical runtime
multi-rank or slice execution
another serializer or cache framework
```

The production boundary decision remains conditional on an actual host
integration prototype. The current result is a positive feasibility signal,
not permission to bypass byte-equivalence, lifecycle, resource-general,
session-inclusive, SDK and physical qualification gates.
