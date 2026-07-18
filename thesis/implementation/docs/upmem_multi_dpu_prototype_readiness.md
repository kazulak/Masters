# Historical UPMEM Multi-DPU Readiness Notes

The former frontier and multi-DPU execution prototypes are retired from the
active Phase B command and test surface. Their normalized records, assignment
schemas, and report labels remain readable for historical snapshots.

The active UPMEM simulator evidence is the strict generic-only sequential SDK
simulator path. The active physical evidence is the guarded one-DPU, one-tasklet
MRAM-resident TaskGraph route documented in
[upmem_hardware_taskgraph_resident_runbook.md](upmem_hardware_taskgraph_resident_runbook.md).

Historical modeled assignment artifacts may describe frontier waves and DPU
ownership, but they do not prove execution, hardware timing, or speedup. Do not
invoke retired frontier, hybrid, dense, or SimplePIM runners from this document.
Use the resident runbook for manual hardware acceptance and the snapshot reader
for old evidence.

The retained claim boundaries are:

- modeled assignment is not executed parallelism;
- SDK simulator timing is not hardware timing or hardware speedup;
- physical evidence is one-DPU functionality/diagnostic evidence only;
- strict UPMEM execution never falls back to CPU contraction.
