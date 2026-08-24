# M4.3 physical SimplePIM TaskGraph adapter

M4.3 transports one real, thesis-owned rank-1 `ContractionTask` into the
existing M4.2 SimplePIM operator. It uses two physical DPUs, synchronous
execution, one warmup and five measured repetitions. It is a functionality
fixture, not a general TN, speedup, energy, PID-Comm, ATiM, or scaling study.

The checkout must include the `simplepim_rank1_task` target adapter exposing
`build_rank1_taskgraph_workload()` and `validate_rank1_task(workload)`.
The native M4.2 host must support `--operands-file`; an older host is rejected
so its fixed deterministic operands cannot be mistaken for TaskGraph evidence.

```sh
make upmem-hw-m4-3-plan
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m4-3
UPMEM_HW_M4_3_RUN=runs/evidence/upmem_hardware_simplepim_taskgraph_m4_3/upmem_hw_m4_3/latest make upmem-hw-m4-3-report
```

The plan command builds and parses native code without allocating a DPU. The
execution command fails closed without explicit physical opt-in, rejects
simulator selectors, uses no CPU fallback, and keeps the run under
`runs/evidence/`. Copy ETH runs to the ignored evidence inbox before reporting
locally. The report writes only derived CSV/README/inspection artifacts under
`runs/comparisons/upmem_hardware_simplepim_taskgraph_m4_3/`.

Expected evidence includes `operands.bin`, `input_manifest.json`, execution
bundle hashes, native response, five measured normalized rows, exact int64
reference agreement, physical allocation/release flags, and transfer-byte
invariants. Environment metadata explicitly records `simplepim_integration`,
the pinned SimplePIM commit, adapter patch identity, and SDK/compiler paths.
Normalized rows distinguish `thesis_source_commit` from
`simplepim_source_commit`; `per_iteration_operator_time_s` is separate from
native session/setup timing. The report verdict is
`taskgraph_derived_operand_adapter_functionality_evidence` when all checks pass.
SimplePIM intermediate table contents are not independently traced, and the
fixture is one synthetic TaskGraph task. No result is promoted automatically
into `thesis_results/current`.
