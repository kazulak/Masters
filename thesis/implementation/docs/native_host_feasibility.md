# Native Host Execution Feasibility v1

## Decision

The current checkpoint is a **no-go for a C-only record-construction
migration**. The host-only prototype produced byte-equivalent output and
canonical SHA-256 values, but the C prepared-stage probe's repeated
copy-and-hash loop was slower than the Python fixture assembly for every
tested fixture. It also removed no process, pipe,
filesystem, SDK, or request boundary. A production native executor is
therefore not justified by this evidence.

This is not evidence that Python is the dominant cost of the complete UPMEM
sample. It is a bounded result for deterministic serialization and record
construction. A future stage-level boundary would require a separate
prototype that measures eliminated crossings and file operations.

## Provenance

```text
accepted source: c4efb3f17e29672e91a0a844881ead53ccf9f2c7
F1 analyzer:     558dcc25dc3f7e9d5751a397d982f5e75d268c1a
F2 probe:        c57043a7762db0a079653d9130cc659827df60e4
scope:           host-only; no UPMEM SDK or DPU allocation
iterations:      30
```

The F2 probe is a standalone executable under
`native/upmem/runtime/prepared_stage_probe.c`. It uses the existing v4
work-unit width as a layout guard but does not change ABI-v4 or production
runtime dispatch.

## Equivalence

The Python arm assembled the deterministic fixture records and packet. The C
probe then copied the packet's record and payload regions, hashed the packet and
canonical output, and wrote the canonical bytes. Those C-produced packet,
canonical and output hashes matched the Python fixture hashes for every
fixture. Record order, payload bytes, packet bytes and output bytes were
deterministic. The probe reports setup separately from repeated steady copying
and hashing and includes process-inclusive elapsed time.

## Host-only observations

| Fixture | Records | Payload bytes | Python steady (s) | C setup (s) | C steady (s) | C process (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stress_1d_t8 | 888 | 1,528,416 | 0.034176 | 0.006683 | 0.174378 | 0.182981 |
| stress_4d_t8 | 2,544 | 1,528,416 | 0.042502 | 0.007783 | 0.189349 | 0.199161 |
| hs_1d_t8 | 224 | 1,528,416 | 0.029262 | 0.006505 | 0.173432 | 0.181768 |
| ghz_4d_t8 | 1,856 | 1,528,416 | 0.051709 | 0.007033 | 0.182476 | 0.191347 |

These are feasibility measurements, not physical performance results. The
probe performs the same repeated copy and hashing work in both arms, so it
does not estimate the benefit of eliminating the current per-stage request
protocol.

## Cost-boundary interpretation

The existing canonical evidence confirms hundreds of request submissions per
sample and thousands of payload files in the tested Stress18 routes. The F1
analyzer reports those counts and the current nested timing hierarchy. It
does not establish that the Python interpreter, rather than file operations,
native parsing, hashing, or SDK waiting, is the dominant part of that host
time. No production C migration should be based on that assumption.

## Qualification status

```text
F1 focused tests:       passed
F2 focused tests:       passed
F2 standalone C build:  passed
F2 equivalence:         passed for all four fixtures
physical execution:     not performed
production runtime:     unchanged
```

The pinned hosted qualification at the accepted source remains the software
authority. A local full-suite attempt outside that environment reported
`810 passed, 37 failed`; the failures were caused by unavailable Quimb/
cotengra packages and missing local SimplePIM/UPMEM build headers. Tests were
not weakened or skipped.

## Claim boundary

This checkpoint supports only the following statement:

> A standalone C prepared-stage copy and hashing probe was shown to preserve
> the deterministic fixture bytes, but it did not provide a host-only speed
> advantage in the tested fixtures. The result does not justify replacing the
> production Python control path.

It does not support claims about full simulation speed, physical UPMEM
performance, C/ Python crossing reduction, persistent sessions, or general
circuit behavior.
