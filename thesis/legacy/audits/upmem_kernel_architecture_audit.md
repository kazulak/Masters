# UPMEM Kernel Architecture Audit

Wave 2E.22 audit date: 2026-07-02.

## Verdict

**Bounded generic UPMEM contraction exists, but fully general UPMEM TN contraction does not yet exist.**

The current implementation has real UPMEM SDK simulator DPU programs for:

- dense int8 x int8 to int32 GEMM, with L1 direct and L2 single-DPU tiled strategies;
- a bounded generic binary tensor-contraction loop over compact integer index metadata.

The generic loop is mathematically broader than GEMM for small binary contractions, but it is not architecturally general yet: it copies full operands into WRAM-sized local buffers, uses one DPU, has rank and element caps, has no WRAM/MRAM tiling, no multi-DPU partitioning, no native complex kernel, no native quantize/dequantize, and no communication/reduction layer.

## Verdict Categories

| Category | Meaning |
| --- | --- |
| `locally_revalidated` | Source exists and was exercised by a command run during this audit. |
| `test_covered_but_not_locally_revalidated` | Source exists and has explicit tests, but no local command in this audit directly executed the path. |
| `exists_but_not_adequately_tested` | Source exists, but coverage is too narrow for the claimed capability. |
| `cpu_side_only` | Implemented on CPU/host side only, not as a UPMEM DPU kernel. |
| `missing` | No current implementation found in the implementation tree. |

## Generality Terms

| Term | Meaning |
| --- | --- |
| Mathematical generality | Whether the formula can represent arbitrary binary tensor contractions. |
| Implementation generality under caps | Whether current code supports that formula within explicit rank, shape, dtype, and execution caps. |
| UPMEM architectural generality | Whether the implementation uses UPMEM memory/parallelism generally: MRAM/WRAM tiling, multiple DPUs, reductions, layout transforms, and communication. |

## Validation Snapshot

These commands should be treated as local audit evidence only when they pass in the current checkout.

| Command | Audit purpose | Result |
| --- | --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q` | Revalidate unit/integration tests covering generic prep, dense bridge, generic bridge, strict runtime, and simulation compare. | Passed: `261 passed in 26.88s`. |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml` | Confirm normal benchmark smoke behavior is still intact. | Passed; run directory `runs/20260702_003901_smoke`. |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-taskgraph-runtime --case bell_2q --policy dense-then-generic --quantization-mode per_task_input_quantize --execute-external` | Revalidate strict sequential UPMEM SDK simulator TaskGraph runtime on a small real case. | Passed with `status: completed`; run directory `runs/20260702_003906_bell_2q_upmem_taskgraph_runtime`. |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simulation-backend-compare --suite configs/suites/simulation_backend_compare_upmem_sdk_simulator.yml --artifact-retention compact` | Revalidate the normalized comparison route `upmem_tn_sdk_simulator_quantized`. | Passed with `status: completed`, `case_count: 2`, and 6 normalized records in `runs/20260702_003911_simulation_backend_compare_upmem_sdk_simulator_simulation_backend_compare/normalized_records.jsonl`. |

## Wave 2E.23 Boundary Result

Wave 2E.23 added and executed `generic_rank3_boundary`, a developer-only non-GEMM binary contraction:

```text
abc,cde->abde
left shape:  (2, 3, 4)
right shape: (4, 5, 2)
output shape: (2, 3, 5, 2)
```

| Evidence command | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench generic-task-bridge --case generic_rank3_boundary --task-index 0 --backend upmem_sdk_simulator_generic_loop --execute-external` | Passed with `status: completed`; run directory `runs/20260702_142056_generic_rank3_boundary_generic_task_bridge`. The summary records `bridge_execution_status=upmem_sdk_simulator_generic_loop_executed`, `dpu_program_invocations=1`, `upmem_program_executed=true`, `cpu_fallback_used=false`, and validation against `expected_quantized_reference_output` with `max_abs_error=5.551115123125783e-17`. |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-taskgraph-runtime --case generic_rank3_boundary --policy generic-only --quantization-mode per_task_input_quantize --execute-external` | Passed with `status: completed`; current run directory `runs/20260702_154721_generic_rank3_boundary_upmem_taskgraph_runtime`. The summary records `dpu_program_invocations=1`, `runtime_tensor_sources_all_upmem_output_blobs=true`, `generic_only_all_tasks_used_generic_backend=true`, `valid_primary_upmem_codepath_result=true`, and `final_validation.reference_kind=generic_quantized_taskgraph_replay` with primary `max_abs_error=5.551115123125783e-17`. The secondary full-precision accuracy record reports `max_abs_error=0.0022317899307078837`. |

Native index metadata was preserved in the evidence:

```text
output_to_left_axes:      [0, 1, -1, -1]
output_to_right_axes:     [-1, -1, 1, 2]
contracted_to_left_axes:  [2]
contracted_to_right_axes: [0]
contracted_dims:          [4]
output_element_count:     60
```

Boundary verdict: the existing generic UPMEM SDK simulator fallback can execute at least one concrete tensor-network-shaped, non-GEMM rank-3 x rank-3 -> rank-4 binary contraction correctly under current caps. This strengthens the claim that the UPMEM path is not limited to GEMM-like dense bridge tasks. It does not change the broader verdict: bounded generic UPMEM contraction exists, but fully general UPMEM TN contraction does not yet exist.

## Wave 2E.24 Generic-Only TaskGraph Coverage Result

Wave 2E.24 changed strict `generic-only` TaskGraph validation so the primary final reference is the CPU-side generic quantized/dequantized TaskGraph replay, not the full-precision CPU einsum replay. Full-precision CPU TaskGraph output is still recorded as secondary accuracy evidence.

The generic quantized replay is implemented by `build_generic_quantized_taskgraph_reference` in `src/quantum_bench/targets/upmem/taskgraph_runtime.py`. It uses `prepare_generic_task` for each real task, combines complex tasks with the same split-real-imag four-component contract as the runtime, and inserts each quantized/dequantized task result into the reference tensor map.

| Evidence command | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-taskgraph-runtime --case bell_2q --policy generic-only --quantization-mode per_task_input_quantize --execute-external` | Passed with `status: completed`; run directory `runs/20260702_154516_bell_2q_upmem_taskgraph_runtime`. The summary records `total_tasks=3`, `dpu_program_invocations=3`, `generic_only_all_tasks_used_generic_backend=true`, `runtime_tensor_sources_all_upmem_output_blobs=true`, `cpu_fallback_used=false`, `valid_primary_upmem_codepath_result=true`, and `final_validation.reference_kind=generic_quantized_taskgraph_replay`. |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-taskgraph-runtime --case ghz_chain --n-qubits 3 --policy generic-only --quantization-mode per_task_input_quantize --execute-external` | Passed with `status: completed`; run directory `runs/20260702_154516_ghz_chain_upmem_taskgraph_runtime`. The summary records `total_tasks=5`, `dpu_program_invocations=5`, `generic_only_all_tasks_used_generic_backend=true`, `cpu_fallback_used=false`, and `valid_primary_upmem_codepath_result=true`. |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-taskgraph-runtime --case bv --n-qubits 3 --policy generic-only --quantization-mode per_task_input_quantize --execute-external` | Passed with `status: completed`; run directory `runs/20260702_154516_bv_upmem_taskgraph_runtime`. The summary records `total_tasks=9`, `dpu_program_invocations=9`, `generic_only_all_tasks_used_generic_backend=true`, `cpu_fallback_used=false`, and `valid_primary_upmem_codepath_result=true`. Secondary full-precision max absolute error was `1.1102230246251565e-16`. |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench generic-task-bridge --case generic_rank3_boundary --task-index 0 --backend upmem_sdk_simulator_generic_loop --execute-external` | Passed with `status: completed`; run directory `runs/20260702_154529_generic_rank3_boundary_generic_task_bridge`. |

Coverage verdict: the current generic UPMEM SDK simulator path can execute complete small sequential TaskGraphs in `generic-only` mode when every task fits the bounded generic contract. This is a real full-TaskGraph UPMEM SDK simulator codepath result under per-task quantization, not a hardware result and not an optimized runtime. The implementation remains bounded by one-DPU execution, rank/element caps, full local buffers, CPU-side quantization/dequantization, and CPU-side split-complex recombination.

## Direct Answers

### 1. What UPMEM Kernels Exist Today?

| Kernel/module | Current status | Evidence |
| --- | --- | --- |
| Dense L1 direct GEMM | Real UPMEM SDK DPU program, simulator path. | `native/upmem/simplepim/upmem_sdk_dense/dpu.c:25` defines `run_l1_direct`; lines 33-48 read MRAM operands, compute nested GEMM loops, and write `DENSE_C`. `native/upmem/simplepim/upmem_sdk_dense/common.h:6` caps L1 dimensions at `UPMEM_DENSE_MAX_DIM=16`. |
| Dense L2 single-DPU tiled GEMM | Real UPMEM SDK DPU program, simulator path, real-valued only. | `native/upmem/simplepim/upmem_sdk_dense/dpu.c:51` defines `run_l2_tiled`; lines 59-109 iterate output tiles and K tiles, use WRAM tile buffers, and write each C tile once. `native/upmem/simplepim/upmem_sdk_dense/common.h:10` and `:14` cap L2 dimensions and tile dimensions. `native/upmem/simplepim/upmem_sdk_dense_runner.py:408` rejects split-complex L2 with `complex_l2_not_implemented`. |
| Generic binary tensor-contraction loop | Real bounded UPMEM SDK DPU program, simulator path. | `native/upmem/simplepim/upmem_sdk_generic_loop/dpu.c:39` loops over output elements, `:43` loops over contracted combinations, `:47-72` maps compact integer axes/strides and accumulates int8 x int8 into int32, and `:77` writes output. `common.h:6` and `:10` cap rank and element count. |
| Split-complex generic fallback | CPU-side orchestration of four real generic UPMEM calls, not a native complex DPU kernel. | `src/quantum_bench/targets/upmem/taskgraph_runtime.py:414` starts `_execute_generic_split_complex_task`; `:431-436` defines ArBr, AiBi, ArBi, AiBr; `:477-478` combines the four real outputs into one complex output. |
| Split-complex dense L1 | Native dense runner supports a split-complex four-GEMM path for supported L1 cases. | `native/upmem/simplepim/upmem_sdk_dense_runner.py:408-426` detects split-complex operands, rejects L2, and records `split_complex_four_gemm` with `execution_class` `L1_WRAM`. |
| Quantize/dequantize | CPU-side only. | `src/quantum_bench/formats/fixed_point.py:71` implements `quantize_fixed_point`; `:125` implements `dequantize_fixed_point`; no native UPMEM quantization source appears in `rg --files native/upmem/simplepim`, which lists only dense, generic-loop, and stub sources. |
| Reduction/accumulation | Exists inside dense/generic kernels only as local contraction accumulation, not as a standalone reduction/collective kernel. | Dense accumulation is in `native/upmem/simplepim/upmem_sdk_dense/dpu.c:38-42` and `:90-96`. Generic accumulation is in `native/upmem/simplepim/upmem_sdk_generic_loop/dpu.c:41-72`. No standalone reduction source appears in `rg --files native/upmem/simplepim`. |
| Transpose/layout transform | Missing as a native UPMEM kernel. | `rg --files native/upmem/simplepim` lists dense, generic-loop, and stub sources only; no transpose/layout program exists. |
| Slicing/partitioning | Missing as a native UPMEM kernel. | Same native source listing; no slicing/partitioning DPU program exists. |
| Multi-DPU communication/collectives | Missing. | Dense and generic host runners allocate one DPU: `native/upmem/simplepim/upmem_sdk_dense/host.c:145` and `native/upmem/simplepim/upmem_sdk_generic_loop/host.c:103`. |

### 2. Is the Generic Kernel Truly General TN Contraction?

No.

| Generality axis | Verdict | Evidence |
| --- | --- | --- |
| Mathematical generality | Broad for binary contractions representable by output-axis and contracted-axis mappings. | `src/quantum_bench/routing/generic_prepare.py:138` defines `generic_loop_reference_int32`; `:159-168` loops over every output element and contracted combination using axis/stride maps. |
| Implementation generality under caps | Bounded: binary, rank <= 6, tensor element count <= 4096, contracted combinations <= 4096 by default, int8 inputs, int32 accumulation. | `GenericTaskPreparationCaps` at `src/quantum_bench/routing/generic_prepare.py:25-30`; native caps in `native/upmem/simplepim/upmem_sdk_generic_loop/common.h:6-12`; rejection checks in `src/quantum_bench/routing/generic_prepare.py:326-349`. |
| UPMEM architectural generality | Not general: one DPU, full operands loaded into local WRAM arrays, no tiling, no partitioning, no distributed reduction. | `native/upmem/simplepim/upmem_sdk_generic_loop/dpu.c:13-15` declares full local A/B/C arrays; `:36-37` reads full A/B; `native/upmem/simplepim/upmem_sdk_generic_loop/host.c:103` allocates one DPU. |
| Complex support | Not native generic complex. Runtime supports split-complex by four real generic UPMEM calls plus CPU host combine. | Generic preparation rejects complex at `src/quantum_bench/routing/generic_prepare.py:177-180`; runtime four-call split-complex path is `src/quantum_bench/targets/upmem/taskgraph_runtime.py:414-505`. |
| Correctness tests | Unit tests cover quantized reference, caps, label-free manifest, CPU-feed prevention, split-complex orchestration, and compare artifacts. | `tests/test_generic_task_prepare.py:62-150`, `tests/test_generic_bridge.py:52-115`, and `tests/test_upmem_taskgraph_runtime.py:189-289`. |

## System Architecture Table

| Module | Code capability | Validated capability | Runs on CPU/UPMEM | Evidence citation | Missing work |
| --- | --- | --- | --- | --- | --- |
| TaskGraph construction and CPU exact reference | Builds sequential contraction tasks and computes CPU reference. | `locally_revalidated` through smoke/runtime commands; test coverage is broad via pytest. | CPU | Runtime uses `execute_task_sequence_np_einsum` for references in `tests/test_upmem_taskgraph_runtime.py:191` and `:229`; strict runtime takes `reference_output` at `src/quantum_bench/targets/upmem/taskgraph_runtime.py:87`. | Parallel TaskGraph frontier execution is not implemented in strict UPMEM runtime. |
| Strict UPMEM TaskGraph runtime | Sequentially executes every task through dense/generic UPMEM SDK bridge paths, rejects missing inputs or unsupported tasks, updates runtime tensor map from output blobs. | `locally_revalidated` on `bell_2q` during this audit; unit tests cover no CPU feed. | CPU orchestrator plus UPMEM SDK simulator DPU programs | `execute_upmem_taskgraph_runtime` at `src/quantum_bench/targets/upmem/taskgraph_runtime.py:78`; task loop at `:114-180`; no CPU-feed assertions in summary at `:675-690`; unit test at `tests/test_upmem_taskgraph_runtime.py:189-224`. | No parallel scheduling, hardware mode, L3 distribution, or persistent network quantization. |
| Dense bridge backend | Writes dense bridge manifests, runs UPMEM SDK dense runner, validates output. | `locally_revalidated` for small supported runtime/compare cases; L2/split-complex behavior is mainly test-covered. | CPU host plus UPMEM SDK simulator DPU program | Runtime dense policy calls `prepare_dense_task`, `write_dense_bridge_input_manifest`, and `execute_dense_bridge` at `src/quantum_bench/targets/upmem/taskgraph_runtime.py:296-318`; dense native source at `native/upmem/simplepim/upmem_sdk_dense/dpu.c:25-118`; tests at `tests/test_dense_bridge.py:711-966`. | L2 complex unsupported; no multi-DPU GEMM; no hardware validation in this audit. |
| Generic bridge backend | Writes generic manifests with compact integer index metadata and runs UPMEM SDK generic loop runner. | `locally_revalidated` through strict runtime and compare suite; manifest and disabled-external behavior test-covered. | CPU host plus UPMEM SDK simulator DPU program | Generic real component writes manifest and executes bridge at `src/quantum_bench/targets/upmem/taskgraph_runtime.py:524-547`; manifest fields in `src/quantum_bench/targets/upmem/generic_bridge.py:150-228`; tests at `tests/test_generic_bridge.py:52-115`. | No tiling, batching, multi-DPU, or native complex loop. |
| Fixed-point conversion | Per-task input quantization/dequantization and error records. | `locally_revalidated` as part of strict UPMEM runtime; direct implementation is CPU-side only. | CPU | `quantize_fixed_point` at `src/quantum_bench/formats/fixed_point.py:71`; `dequantize_fixed_point` at `:125`; generic preparation uses quantized operands and expected quantized reference at `src/quantum_bench/routing/generic_prepare.py:189-209`. | No native DPU quantize/dequantize kernel; no persistent network quantization. |
| Split-complex generic fallback | Four real generic calls plus host-side complex combine. | `test_covered_but_not_locally_revalidated` for complex-specific path in this audit. | UPMEM for real components; CPU for orchestration/combine | Runtime implementation at `src/quantum_bench/targets/upmem/taskgraph_runtime.py:414-505`; tests at `tests/test_upmem_taskgraph_runtime.py:227-278`. | Native split-complex generic kernel absent; four calls increase runtime overhead. |
| UPMEM comparison route | Emits normalized `upmem_tn_sdk_simulator_quantized` rows only when strict UPMEM runtime assertions pass. | `locally_revalidated` by UPMEM simulator comparison suite during this audit. | CPU orchestration plus UPMEM SDK simulator | Route ID in `src/quantum_bench/providers/exact_tn/upmem_sdk_simulator.py:39`; row semantics at `:324-336`; strict assertions at `:289-307`; tests at `tests/test_simulation_backend_compare.py:926-1202`. | Quantized, not exact; currently minimal validation suite only. |
| SimplePIM compute path | Stub/capability lane only; no SimplePIM GEMM runtime. | `test_covered_but_not_locally_revalidated` for stub/capability behavior only. | CPU/subprocess stub, no real SimplePIM kernel | Native source listing includes `native/upmem/simplepim/simplepim_dense_stub.py`; generic manifest records `simplepim_api_used=false` in `tests/test_generic_bridge.py:59-61`. | Real SimplePIM kernel integration missing. |
| PID-Comm / collectives | Candidate external library only; no runtime integration. | `missing` for execution. | Not applicable | No PID-Comm execution source appears in `native/upmem/simplepim`; one-DPU runners cite `host.c:103` and dense `host.c:145`. | Communication/orchestration substrate across L1/L2/L3 remains future work. |

## Kernel Coverage Table

| Kernel | Needed for TN? | Mathematical generality | Implementation generality under caps | UPMEM architectural generality | Validated capability | Evidence citation | Current limitations | Next minimal test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 dense GEMM | Yes, for GEMM-lowered contractions. | Matrix multiply only. | Int8 inputs, int32 output, dimensions capped at 16 by default. | WRAM-resident single-DPU direct GEMM. | `locally_revalidated` for small cases through strict runtime/compare; broader cases test-covered. | `native/upmem/simplepim/upmem_sdk_dense/common.h:6-24`; `native/upmem/simplepim/upmem_sdk_dense/dpu.c:25-49`; `tests/test_dense_bridge.py:711-966`. | Small max dim, simulator-only in current validation. | Add explicit command-level L1 task fixture with recorded output manifest checks. |
| L2 dense tiled GEMM | Yes, for larger GEMM-lowered contractions. | Matrix multiply only. | Real-valued, max dim 512, tile max 64, one DPU. | MRAM-resident operands with WRAM output/K tiling, still single DPU. | `test_covered_but_not_locally_revalidated` in this audit. | `native/upmem/simplepim/upmem_sdk_dense/dpu.c:51-110`; `native/upmem/simplepim/upmem_sdk_dense_runner.py:479-538`; `tests/test_dense_bridge.py:756-779` and `:953-966`. | Complex L2 rejected; no multi-DPU distribution. | Run one synthetic L2 real task through `pim-bridge-eval` or `dense-task-bridge` and record manifest metadata. |
| Generic binary contraction loop | Yes, as fallback coverage for small non-specialized contractions. | Binary contractions with arbitrary free/contracted labels. | Rank <= 6, each tensor <= 4096 elements, contracted combinations <= 4096, real int8 only at preparation level. | Single DPU; full inputs and output local arrays; no tiling. | `locally_revalidated` through strict runtime/compare; generic-specific unit tests revalidated by pytest. | `src/quantum_bench/routing/generic_prepare.py:25-30`, `:138-168`, `:322-349`; `native/upmem/simplepim/upmem_sdk_generic_loop/dpu.c:9-15`, `:36-77`; `tests/test_generic_task_prepare.py:62-150`. | Not scalable; not native complex; no WRAM/MRAM tiling. | Add one non-GEMM rank-3/rank-4 real task command-level execution with real SDK simulator output. |
| Generic split-complex fallback | Yes, because quantum tensors are complex. | Correct split formula for binary complex contraction. | Four real generic calls within real generic caps. | UPMEM executes real components; CPU combines complex output. | `test_covered_but_not_locally_revalidated` for complex path. | `src/quantum_bench/targets/upmem/taskgraph_runtime.py:431-505`; `tests/test_upmem_taskgraph_runtime.py:227-278`. | CPU-side combine; four launches; no native complex DPU program. | Run one small complex circuit through strict runtime and inspect task metrics for four components. |
| Quantize/dequantize | Yes for current int8 route. | Applies per tensor/operand, not a contraction kernel. | CPU supports signed symmetric int8/int16; current runtime uses per-task int8. | No UPMEM kernel. | `cpu_side_only`. | `src/quantum_bench/formats/fixed_point.py:71-122` and `:125-152`; runtime quantization mode check at `src/quantum_bench/targets/upmem/taskgraph_runtime.py:91-96`. | CPU overhead; no persistent network quantization. | Add timing breakdown check for quantize/dequantize share in strict runtime. |
| Reduction/accumulation | Yes. | Accumulation is embedded in GEMM/generic loops. | Per-output local int32 accumulation only. | No standalone or distributed reduction. | `exists_but_not_adequately_tested` as standalone capability. | Dense accumulation at `native/upmem/simplepim/upmem_sdk_dense/dpu.c:38-42` and `:90-96`; generic accumulation at `native/upmem/simplepim/upmem_sdk_generic_loop/dpu.c:41-72`. | No all-reduce, no partial-output reduction tree. | Add future L3 reduction prototype or explicit host-mediated reduction model. |
| Transpose/layout transform | Often needed for efficient TN lowering and scheduling. | Missing. | Missing. | Missing. | `missing`. | Native source inventory from `rg --files native/upmem/simplepim` lists only dense, generic-loop, README, stub, and runners. | CPU currently handles layout/lowering. | Add a small layout-transform design note before implementation. |
| Slicing/partitioning | Needed for L2/L3 memory hierarchy and distribution. | Missing as kernel. | Missing. | Missing. | `missing`. | Native source inventory from `rg --files native/upmem/simplepim`; strict runtime has no partitioning path in `src/quantum_bench/targets/upmem/taskgraph_runtime.py:114-180`. | No tensor slicing or scatter/gather kernel. | Add one host-side partition metadata model before native DPU slicing. |
| Multi-DPU distribution/collectives | Needed for L3. | Missing. | Missing. | Missing. | `missing`. | Generic host allocates one DPU at `native/upmem/simplepim/upmem_sdk_generic_loop/host.c:103`; dense host allocates one DPU at `native/upmem/simplepim/upmem_sdk_dense/host.c:145`. | No PID-Comm integration, no native collectives, no multi-DPU scheduler. | Implement a model-only L3 communication contract, then a two-DPU simulator proof if SDK supports it. |

## Next Minimal Work After 2E.24

The next execution-focused step should expand the generic boundary by one dimension of architectural pressure:

1. Add one generic boundary case that exceeds the current full-local-buffer approach but still fits one DPU MRAM.
2. Confirm it rejects before launch with an explicit cap/WRAM-local-buffer reason.
3. Use that evidence to define the minimum L2 generic tiling requirement.

Keep hardware, multi-DPU, PID-Comm, SimplePIM, and native complex-generic execution out of this immediate follow-up.
