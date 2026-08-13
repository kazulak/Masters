PYTHON ?= ../.venv/bin/python
PYTHON_BIN_DIR := $(abspath $(dir $(PYTHON)))
export PATH := $(PYTHON_BIN_DIR):$(PATH)
PY_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
BENCH := $(PY_ENV) $(PYTHON) -m quantum_bench.bench
BENCH_CPU_THREADS ?=
THESIS_BENCH_ENV := \
	BENCH_CPU_THREADS=$(BENCH_CPU_THREADS) \
	OMP_NUM_THREADS=$(BENCH_CPU_THREADS) \
	OPENBLAS_NUM_THREADS=$(BENCH_CPU_THREADS) \
	MKL_NUM_THREADS=$(BENCH_CPU_THREADS) \
	NUMEXPR_NUM_THREADS=$(BENCH_CPU_THREADS) \
	OMP_DYNAMIC=FALSE \
	OMP_PROC_BIND=close \
	OMP_PLACES=cores

CPU_SUITE ?= configs/suites/cpu_evidence.yml
GPU_SUITE ?= configs/suites/gpu_evidence.yml
UPMEM_SIM_SUITE ?= configs/suites/upmem_sim_evidence.yml
UPMEM_HW_TASKGRAPH_RESIDENT_SUITE ?= configs/suites/upmem_hardware_taskgraph_resident_path_quantization.yml
UPMEM_HW_TASKGRAPH_RESIDENT_RUN ?= runs/evidence/upmem_hardware_taskgraph_resident_path_quantization/upmem_hw_taskgraph_resident/latest
UPMEM_HW_M4_6_SUITE ?= configs/suites/upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling.yml
# The parent is intentional: the scaling target creates one immutable child
# evidence run per tasklet value, while the report aggregates that sweep.
UPMEM_HW_M4_6_RUN ?= runs/evidence/upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling/upmem_hw_taskgraph_resident
UPMEM_HW_M4_6_REPORT ?= runs/comparisons/upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling/latest
UPMEM_HW_M4_6_TASKLETS ?= 1 2 4 8 16
UPMEM_HW_SLICED_RESIDENT_MVP_SUITE ?= configs/suites/upmem_hardware_sliced_resident_mvp.yml
UPMEM_HW_SLICED_RESIDENT_M2_1_SUITE ?= configs/suites/upmem_hardware_sliced_resident_m2_1.yml
UPMEM_HW_SLICED_RESIDENT_M2_2_SUITE ?= configs/suites/upmem_hardware_sliced_resident_m2_2.yml
UPMEM_HW_M2_2_RUN ?= runs/evidence/upmem_hardware_sliced_resident_m2_2/upmem_hw_sliced_resident/latest
UPMEM_HW_SLICED_RESIDENT_M2_3_SUITE ?= configs/suites/upmem_hardware_sliced_resident_m2_3.yml
UPMEM_HW_FRONTIER_M3_1_SUITE ?= configs/suites/upmem_hardware_frontier_m3_1.yml
UPMEM_HW_TASKGRAPH_M4_1_SUITE ?= configs/suites/upmem_hardware_taskgraph_m4_1.yml
UPMEM_HW_TASKGRAPH_M4_1_RUN ?= runs/evidence/upmem_hardware_taskgraph_m4_1/upmem_hw_taskgraph_m4_1/latest
UPMEM_HW_M4_2_SUITE ?= configs/suites/upmem_hardware_simplepim_rank1_m4_2.yml
UPMEM_HW_M4_2_RUN ?= runs/evidence/upmem_hardware_simplepim_rank1_m4_2/upmem_hw_m4_2/latest
UPMEM_HW_M4_3_SUITE ?= configs/suites/upmem_hardware_simplepim_taskgraph_m4_3.yml
UPMEM_HW_M4_3_RUN ?= runs/evidence/upmem_hardware_simplepim_taskgraph_m4_3/upmem_hw_m4_3/latest
UPMEM_HW_M4_4_SUITE ?= configs/suites/upmem_hardware_simplepim_chain_m4_4.yml
UPMEM_SIMPLEPIM_TASKGRAPH_SUITE ?= configs/suites/upmem_hardware_simplepim_taskgraph.yml
UPMEM_HW_M2_3_RUN ?= runs/evidence/upmem_hardware_sliced_resident_m2_3/upmem_hw_sliced_resident/latest
UPMEM_PROVIDER_CATALOG ?= configs/qualification/upmem_provider_m1.yml
UPMEM_HW_M5_1_RUN ?= runs/evidence/upmem_hardware_distributed_m5_1/upmem_hw_m5_1/latest
UPMEM_HW_M5_1_REPORT ?= runs/comparisons/upmem_hardware_distributed_m5_1/latest
UPMEM_HW_M5_2_RUN ?= runs/evidence/upmem_hardware_distributed_m5_2/upmem_hw_m5_2/latest
UPMEM_HW_M5_2_REPORT ?= runs/comparisons/upmem_hardware_distributed_m5_2/latest
UPMEM_HW_M5_SUITE ?= configs/suites/upmem_hardware_distributed_m5.yml
UPMEM_HW_M5_RUN ?= runs/evidence/upmem_hardware_distributed_m5/upmem_hw_m5/latest
UPMEM_HW_M5_DPU_COUNTS ?=
UPMEM_HW_M5_TASKLETS ?=
UPMEM_HW_M5_ARGS = $(if $(UPMEM_HW_M5_DPU_COUNTS),--dpu-counts $(UPMEM_HW_M5_DPU_COUNTS),) $(if $(UPMEM_HW_M5_TASKLETS),--tasklets $(UPMEM_HW_M5_TASKLETS),)
EVIDENCE_INBOX ?= runs/inbox/eth
GPU_VERIFY ?= quest-hip
CLEAN_RUNS ?= 0
THESIS_PACK ?= runs/comparisons/research_pack/latest
THESIS_SNAPSHOT ?= thesis_results/current
APPLY ?= 0

PUBLIC_TARGETS := \
	help \
	test \
	setup \
	build-quest-cpu \
	doctor \
	bench-cpu \
	bench-gpu \
	bench-upmem-sim \
	upmem-hw-taskgraph-resident-plan \
	upmem-hw-taskgraph-resident \
	upmem-hw-taskgraph-resident-report \
	upmem-hw-m4-6-tasklet-scaling-plan \
	upmem-hw-m4-6-tasklet-scaling \
	upmem-hw-m4-6-tasklet-scaling-report \
	upmem-hw-sliced-resident-plan \
	upmem-hw-sliced-resident \
	upmem-hw-m2-1-plan \
	upmem-hw-m2-1 \
	upmem-hw-m2-2-plan \
	upmem-hw-m2-2 \
	upmem-hw-m2-2-report \
	upmem-hw-m2-3-plan \
	upmem-hw-m2-3 \
	upmem-hw-m2-3-report \
	upmem-hw-frontier-m3-1-plan \
	upmem-hw-frontier-m3-1 \
	upmem-hw-m4-1-plan \
	upmem-hw-m4-1 \
	upmem-hw-m4-1-report \
	upmem-hw-m4-2-plan \
	upmem-hw-m4-2 \
	upmem-hw-m4-2-report \
	upmem-hw-m4-3-plan \
	upmem-hw-m4-3 \
	upmem-hw-m4-3-report \
	upmem-hw-m4-4-plan \
	upmem-hw-m4-4 \
	upmem-simplepim-plan \
	upmem-simplepim-run \
	upmem-hw-m5-1-plan \
	upmem-hw-m5-1 \
	upmem-hw-m5-1-report \
	upmem-hw-m5-2-plan \
	upmem-hw-m5-2 \
	upmem-hw-m5-2-report \
	upmem-hw-m5-plan \
	upmem-hw-m5 \
	upmem-hw-m5-report \
	upmem-pidcomm-compatibility \
	upmem-pidcomm-plan \
	upmem-pidcomm-run \
	upmem-provider-plan \
	upmem-provider-qualify \
	evidence-inbox \
	research-plan \
	planner-evidence \
	planner-report \
	thesis-run \
	thesis-promote \
	thesis-promote-historical \
	thesis-verify \
	thesis-report \
	list-runs \
	thesis-clean \
	archive-evidence \
	thesis-release \
	clean-generated

.PHONY: $(PUBLIC_TARGETS)

help:
	@printf '%s\n' \
		'Thesis evidence shortcuts:' \
		'  make help              Show this command summary' \
		'  make setup             Require uv, create/reuse ../.venv, install dev dependencies, and run doctor' \
		'  make test              Run the unified active Python test suite' \
		'  make doctor             Check local benchmark prerequisites without creating evidence' \
		'  make build-quest-cpu    Build the native QuEST CPU runner' \
		'  make bench-cpu          Run the CPU evidence suite' \
		'  make bench-gpu          Verify the GPU and run the GPU evidence suite' \
		'  make bench-upmem-sim    Check UPMEM simulator and run its evidence suite' \
		'  make upmem-hw-taskgraph-resident-plan  Prepare/build the active one-DPU physical MRAM-resident TaskGraph route; never allocates a DPU' \
		'  make upmem-hw-taskgraph-resident  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; run the active one-DPU physical MRAM-resident TaskGraph route' \
		'  make upmem-hw-taskgraph-resident-report  Generate a report from explicitly supplied one-DPU physical MRAM-resident diagnostic evidence' \
		'  make upmem-hw-m4-6-tasklet-scaling-plan  Prepare the versioned M4.6 tile-2 resident profile for tasklets 1,2,4,8,16' \
		'  make upmem-hw-m4-6-tasklet-scaling  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; run the one-DPU M4.6 tasklet scaling suite' \
		'  make upmem-hw-m4-6-tasklet-scaling-report  Report one explicitly supplied M4.6 scaling run' \
		'  make upmem-hw-sliced-resident-plan  Prepare/build the internal/research M2 two-DPU sliced-resident MVP; never allocates a DPU' \
		'  make upmem-hw-sliced-resident  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; run the internal/research M2 two-DPU sliced-resident MVP' \
		'  make upmem-hw-m2-1-plan  Prepare/build the M2.1 H-X useful-slice fixture; never allocates a DPU' \
		'  make upmem-hw-m2-1  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; run the fixed M2.1 H-X two-DPU fixture' \
		'  make upmem-hw-m2-2-plan  Prepare/build the M2.2 same-plan numeric-mode fixture; never allocates a DPU' \
		'  make upmem-hw-m2-2  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; run float32 and resident-requantized M2.2 modes' \
		'  make upmem-hw-m2-2-report  Report M2.2 evidence; for copied ETH data use UPMEM_HW_M2_2_RUN=runs/inbox/eth/m2_2/<timestamp> make upmem-hw-m2-2-report' \
		'  make upmem-hw-m2-3-plan  Prepare/build the M2.3 two-path, two-mode study; never allocates a DPU' \
		'  make upmem-hw-m2-3  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; run the fixed M2.3 physical study' \
		'  make upmem-hw-m2-3-report  Strictly report M2.3 evidence; for copied ETH data use UPMEM_HW_M2_3_RUN=runs/inbox/eth/m2_3/<timestamp> make upmem-hw-m2-3-report' \
		'  make upmem-hw-frontier-m3-1-plan  Prepare/build the strict M3.1 two-DPU frontier fixture; never allocates a DPU' \
		'  make upmem-hw-frontier-m3-1  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the strict M3.1 two-DPU frontier fixture' \
		'  make upmem-hw-m4-1-plan  Prepare/build the M4.1 raw/SimplePIM differential fixture; never allocates a DPU' \
		'  make upmem-hw-m4-1  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the M4.1 differential fixture' \
		'  make upmem-hw-m4-1-report  Inspect M4.1 differential evidence; no speedup report is generated' \
		'  make upmem-hw-m4-2-plan  Prepare/build/parser-check the fixed SimplePIM rank-1 qualification; never allocates a DPU' \
		'  make upmem-hw-m4-2  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the M4.2 SimplePIM operator qualification' \
		'  make upmem-hw-m4-2-report  Inspect M4.2 functionality evidence; no speedup report is generated' \
		'  make upmem-hw-m4-3-plan  Prepare/build the one real TaskGraph M4.3 adapter; never allocates a DPU' \
		'  make upmem-hw-m4-3  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the bounded M4.3 TaskGraph fixture' \
		'  make upmem-hw-m4-3-report  Validate M4.3 functionality evidence; no speedup report is generated' \
		'  make upmem-hw-m4-4-plan  Prepare/build the one-DPU SimplePIM chain; never allocates a DPU' \
		'  make upmem-hw-m4-4  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the one-DPU SimplePIM chain' \
		'  make upmem-simplepim-plan  Prepare/validate the fixed M4.5 one/two-DPU plans; never allocates a DPU' \
		'  make upmem-simplepim-run  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the fixed M4.5 route' \
		'  make upmem-hw-m5-1-plan  Prepare/native-validate one real float32 output-partition contraction on 1,2,4 DPUs; never allocates a DPU' \
		'  make upmem-hw-m5-1  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the M5.1 output-partition contraction' \
		'  make upmem-hw-m5-1-report  Report the explicitly supplied M5.1 evidence run' \
		'  make upmem-hw-m5-2-plan  Prepare/native-validate one real float32 contracted-partition reduction on 1,2,4 DPUs; never allocates a DPU' \
		'  make upmem-hw-m5-2  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; execute the M5.2 host-mediated reduction' \
		'  make upmem-hw-m5-2-report  Report the explicitly supplied M5.2 evidence run' \
		'  make upmem-hw-m5-plan  Prepare/build the M5 execution-plan-v3 distributed study; never allocates or launches a DPU' \
		'  make upmem-hw-m5  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1 and UPMEM_HW_RANK_PATH; execute the M5 study' \
		'  make upmem-hw-m5-report  Report the explicitly supplied M5 evidence run' \
		'  make upmem-pidcomm-compatibility  Stage and compile/link PID-Comm; never allocates a DPU' \
		'  make upmem-pidcomm-plan  Record PID-Comm CPU/SDK/source facts and stage no hardware; never allocates a DPU' \
		'  make upmem-pidcomm-run  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; qualify PID-Comm physical all-reduce candidates' \
		'  make upmem-provider-plan  Prepare a unique M1 provider qualification plan; never builds or allocates hardware' \
		'  make upmem-provider-qualify PROVIDER=<id>  Execute one selected physical provider qualification' \
		'  make evidence-inbox     Create the ignored inbox for copied ETH evidence bundles' \
		'  make research-plan      Show the research benchmark plan' \
		'  make planner-evidence   Run modeled planner v2 evidence into runs/comparisons/planner_v2/' \
		'  make planner-report     Regenerate the labeled planner v2 report from latest planner evidence' \
		'  make thesis-run BENCH_CPU_THREADS=<physical cores>' \
		'                           Run the controlled thesis benchmark matrix' \
		'  make thesis-promote     Promote the latest research pack into tracked thesis_results/current' \
		'  make thesis-promote-historical THESIS_SNAPSHOT=thesis_results/<name>' \
		'                           Explicitly promote clean historical evidence into a named snapshot; never use current' \
		'  make thesis-verify      Verify tracked thesis results without rerunning benchmarks' \
		'  make thesis-report      Regenerate tracked tables and plots from thesis_results/current' \
		'  make thesis-clean       Preview stale generated run cleanup; add APPLY=1 to remove' \
		'  make archive-evidence RUN=<path> APPLY=1  Checksum-archive one raw run outside Git, then remove the source' \
		'  make thesis-release NAME=<name>  Freeze current results as a named milestone' \
		'  make list-runs          Show generated evidence/comparisons and current selection' \
		'  make clean-generated    Remove generated build/cache files; keeps runs/ by default'

setup:
	@python3 scripts/bootstrap_env.py
	$(MAKE) -C native/quest_cpu clean-quest
	$(MAKE) build-quest-cpu
	$(MAKE) doctor

build-quest-cpu:
	$(MAKE) -C native/quest_cpu

doctor:
	$(PY_ENV) $(PYTHON) scripts/doctor.py

test:
	$(PY_ENV) $(PYTHON) -m pytest -q

bench-cpu: build-quest-cpu
	$(BENCH) simulation-backend-compare --suite $(CPU_SUITE) --artifact-retention compact

bench-gpu:
	$(BENCH) simulation-backend-probe --verify-gpu $(GPU_VERIFY)
	$(BENCH) simulation-backend-compare --suite $(GPU_SUITE) --artifact-retention compact
	$(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py check-gpu runs/latest

bench-upmem-sim:
	$(BENCH) upmem-env-check --run-sample --target simulator
	$(BENCH) simulation-backend-compare --suite $(UPMEM_SIM_SUITE) --artifact-retention compact
	$(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py check-upmem runs/latest

upmem-provider-plan:
	$(BENCH) provider-qualification --catalog $(UPMEM_PROVIDER_CATALOG) --prepare-only

upmem-provider-qualify:
	@test -n "$(PROVIDER)" || { echo 'Set PROVIDER=<id>, for example: make upmem-provider-qualify PROVIDER=simplepim' >&2; exit 2; }
	$(BENCH) provider-qualification --catalog $(UPMEM_PROVIDER_CATALOG) --execute --provider $(PROVIDER)

evidence-inbox:
	@mkdir -p "$(EVIDENCE_INBOX)"
	@printf '%s\n' "$(EVIDENCE_INBOX)" \
		'Copy raw ETH archives here. Extract validated run directories under runs/evidence/ and promote only reviewed compact results into thesis_results/.'

research-plan:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py plan

planner-evidence:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py run --full --suite planner_paths --suite planner_sensitivity --label planner_v2

planner-report:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py report --suite planner_paths --suite planner_sensitivity --label planner_v2

thesis-run: build-quest-cpu
	@test -n "$(BENCH_CPU_THREADS)" || { echo 'Set BENCH_CPU_THREADS to the physical CPU core count, for example: BENCH_CPU_THREADS=6 make thesis-run' >&2; exit 2; }
	@case "$(BENCH_CPU_THREADS)" in *[!0-9]*|'') echo 'BENCH_CPU_THREADS must be a positive integer' >&2; exit 2;; esac
	@test "$(BENCH_CPU_THREADS)" -gt 0 || { echo 'BENCH_CPU_THREADS must be a positive integer' >&2; exit 2; }
	$(PY_ENV) $(THESIS_BENCH_ENV) $(PYTHON) scripts/research_benchmark_pack.py run --full

thesis-promote:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py promote --pack $(THESIS_PACK) --out $(THESIS_SNAPSHOT)

thesis-promote-historical:
	@case "$(THESIS_SNAPSHOT)" in thesis_results/*) ;; *) echo 'Set THESIS_SNAPSHOT=thesis_results/<named-snapshot>' >&2; exit 2;; esac
	@case "$(THESIS_SNAPSHOT)" in */*/*|*..*) echo 'THESIS_SNAPSHOT must be a direct named child of thesis_results' >&2; exit 2;; esac
	@test "$(THESIS_SNAPSHOT)" != "thesis_results/current" && test "$(notdir $(THESIS_SNAPSHOT))" != "current" || { echo 'Set THESIS_SNAPSHOT=thesis_results/<named-snapshot>; historical promotion cannot target current' >&2; exit 2; }
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py promote --historical --pack $(THESIS_PACK) --out $(THESIS_SNAPSHOT)

thesis-verify:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py verify --snapshot $(THESIS_SNAPSHOT)

thesis-report:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py report --snapshot $(THESIS_SNAPSHOT)

upmem-hw-taskgraph-resident-plan:
	$(BENCH) upmem-hardware-taskgraph-resident --suite $(UPMEM_HW_TASKGRAPH_RESIDENT_SUITE) --prepare-only --build

upmem-hw-taskgraph-resident:
	$(BENCH) upmem-hardware-taskgraph-resident --suite $(UPMEM_HW_TASKGRAPH_RESIDENT_SUITE) --execute

upmem-hw-m4-6-tasklet-scaling-plan:
	@for tasklets in $(UPMEM_HW_M4_6_TASKLETS); do \
		$(BENCH) upmem-hardware-taskgraph-resident --suite $(UPMEM_HW_M4_6_SUITE) --prepare-only --build --tasklets $$tasklets || exit $$?; \
	done

upmem-hw-m4-6-tasklet-scaling:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	@test -n "$(UPMEM_HW_RANK_PATH)" || { echo 'Set UPMEM_HW_RANK_PATH=/dev/dpu_rankN to select a physical UPMEM rank' >&2; exit 2; }
	@for tasklets in $(UPMEM_HW_M4_6_TASKLETS); do \
		$(BENCH) upmem-hardware-taskgraph-resident --suite $(UPMEM_HW_M4_6_SUITE) --execute --tasklets $$tasklets || exit $$?; \
	done

upmem-hw-m4-6-tasklet-scaling-report:
	@test -n "$(UPMEM_HW_M4_6_RUN)" || { echo 'Set UPMEM_HW_M4_6_RUN=<physical M4.6 evidence directory>' >&2; exit 2; }
	@test -n "$(UPMEM_HW_M4_6_REPORT)" || { echo 'Set UPMEM_HW_M4_6_REPORT=<comparison output directory>' >&2; exit 2; }
	$(BENCH) report-run --input $(UPMEM_HW_M4_6_RUN) --out $(UPMEM_HW_M4_6_REPORT) --no-output-plots

upmem-hw-sliced-resident-plan:
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_MVP_SUITE) --prepare-only --build

upmem-hw-sliced-resident:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_MVP_SUITE) --execute

upmem-hw-m2-1-plan:
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_M2_1_SUITE) --prepare-only --build

upmem-hw-m2-1:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_M2_1_SUITE) --execute

upmem-hw-m2-2-plan:
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_M2_2_SUITE) --prepare-only --build

upmem-hw-m2-2:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_M2_2_SUITE) --execute

upmem-hw-m2-2-report:
	$(PY_ENV) $(PYTHON) scripts/upmem_m2_2_report.py --input $(UPMEM_HW_M2_2_RUN)

upmem-hw-m2-3-plan:
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_M2_3_SUITE) --prepare-only --build

upmem-hw-m2-3:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-sliced-resident-mvp --suite $(UPMEM_HW_SLICED_RESIDENT_M2_3_SUITE) --execute

upmem-hw-m2-3-report:
	$(PY_ENV) $(PYTHON) scripts/upmem_m2_3_report.py --input $(UPMEM_HW_M2_3_RUN)

upmem-hw-frontier-m3-1-plan:
	$(BENCH) upmem-hardware-frontier-m3-1 --suite $(UPMEM_HW_FRONTIER_M3_1_SUITE) --prepare-only --build

upmem-hw-frontier-m3-1:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-frontier-m3-1 --suite $(UPMEM_HW_FRONTIER_M3_1_SUITE) --execute

upmem-hw-m4-1-plan:
	$(BENCH) upmem-hardware-taskgraph-m4-1 --suite $(UPMEM_HW_TASKGRAPH_M4_1_SUITE) --prepare-only --build

upmem-hw-m4-1:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-taskgraph-m4-1 --suite $(UPMEM_HW_TASKGRAPH_M4_1_SUITE) --execute

upmem-hw-m4-1-report:
	$(PY_ENV) $(PYTHON) scripts/upmem_m4_1_report.py --input $(UPMEM_HW_TASKGRAPH_M4_1_RUN)

upmem-hw-m4-2-plan:
	$(BENCH) upmem-hardware-simplepim-rank1-m4-2 --suite $(UPMEM_HW_M4_2_SUITE) --prepare-only --build

upmem-hw-m4-2:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-simplepim-rank1-m4-2 --suite $(UPMEM_HW_M4_2_SUITE) --execute

upmem-hw-m4-2-report:
	@test -n "$(UPMEM_HW_M4_2_RUN)" || { echo 'Set UPMEM_HW_M4_2_RUN=<physical M4.2 evidence directory>' >&2; exit 2; }
	$(PY_ENV) $(PYTHON) scripts/upmem_m4_2_report.py --input $(UPMEM_HW_M4_2_RUN)

upmem-hw-m4-3-plan:
	$(BENCH) upmem-hardware-simplepim-taskgraph-m4-3 --suite $(UPMEM_HW_M4_3_SUITE) --prepare-only --build

upmem-hw-m4-3:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-simplepim-taskgraph-m4-3 --suite $(UPMEM_HW_M4_3_SUITE) --execute

upmem-hw-m4-3-report:
	@test -n "$(UPMEM_HW_M4_3_RUN)" || { echo 'Set UPMEM_HW_M4_3_RUN=<physical M4.3 evidence directory>' >&2; exit 2; }
	$(PY_ENV) $(PYTHON) scripts/upmem_m4_3_report.py --input $(UPMEM_HW_M4_3_RUN)

upmem-hw-m4-4-plan:
	$(BENCH) upmem-hardware-simplepim-chain-m4-4 --suite $(UPMEM_HW_M4_4_SUITE) --prepare-only --build

upmem-hw-m4-4:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-simplepim-chain-m4-4 --suite $(UPMEM_HW_M4_4_SUITE) --execute

upmem-simplepim-plan:
	$(BENCH) upmem-simplepim-taskgraph --suite $(UPMEM_SIMPLEPIM_TASKGRAPH_SUITE) --prepare-only --build

upmem-simplepim-run:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(BENCH) upmem-simplepim-taskgraph --suite $(UPMEM_SIMPLEPIM_TASKGRAPH_SUITE) --execute

upmem-hw-m5-1-plan:
	$(BENCH) upmem-hardware-distributed-m5-1 --prepare-only --build

upmem-hw-m5-1:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	@test -n "$(UPMEM_HW_RANK_PATH)" || { echo 'Set UPMEM_HW_RANK_PATH=/dev/dpu_rankN to select a physical UPMEM rank' >&2; exit 2; }
	$(BENCH) upmem-hardware-distributed-m5-1 --execute

upmem-hw-m5-1-report:
	$(BENCH) report-run --input $(UPMEM_HW_M5_1_RUN) --out $(UPMEM_HW_M5_1_REPORT) --no-output-plots

upmem-hw-m5-2-plan:
	$(BENCH) upmem-hardware-distributed-m5-2 --prepare-only --build

upmem-hw-m5-2:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	@test -n "$(UPMEM_HW_RANK_PATH)" || { echo 'Set UPMEM_HW_RANK_PATH=/dev/dpu_rankN to select a physical UPMEM rank' >&2; exit 2; }
	$(BENCH) upmem-hardware-distributed-m5-2 --execute

upmem-hw-m5-2-report:
	$(BENCH) report-run --input $(UPMEM_HW_M5_2_RUN) --out $(UPMEM_HW_M5_2_REPORT) --no-output-plots

upmem-hw-m5-plan:
	$(BENCH) upmem-hardware-distributed-m5 --suite $(UPMEM_HW_M5_SUITE) --prepare-only --build $(UPMEM_HW_M5_ARGS)

upmem-hw-m5:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	@test -n "$(UPMEM_HW_RANK_PATH)" || { echo 'Set UPMEM_HW_RANK_PATH=/dev/dpu_rankN to select a physical UPMEM rank' >&2; exit 2; }
	$(BENCH) upmem-hardware-distributed-m5 --suite $(UPMEM_HW_M5_SUITE) --execute $(UPMEM_HW_M5_ARGS)

upmem-hw-m5-report:
	$(PY_ENV) $(PYTHON) scripts/upmem_m5_report.py --input $(UPMEM_HW_M5_RUN) --output-root .

upmem-pidcomm-compatibility:
	$(PY_ENV) $(PYTHON) native/upmem/pidcomm_qualification/pidcomm_qualification_runner.py --compatibility-only --workdir .

upmem-pidcomm-plan:
	$(PY_ENV) $(PYTHON) native/upmem/pidcomm_qualification/pidcomm_qualification_runner.py --prepare-only --workdir .

upmem-pidcomm-run:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to execute physical hardware' >&2; exit 2; }
	$(PY_ENV) $(PYTHON) native/upmem/pidcomm_qualification/pidcomm_qualification_runner.py --execute --workdir .

upmem-hw-taskgraph-resident-report:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py check-resident-evidence --input $(UPMEM_HW_TASKGRAPH_RESIDENT_RUN)
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py report --input $(UPMEM_HW_TASKGRAPH_RESIDENT_RUN) --label upmem_hw_taskgraph_resident

list-runs:
	$(PY_ENV) $(PYTHON) scripts/thesis_runs.py list

thesis-clean:
	$(PY_ENV) $(PYTHON) scripts/thesis_runs.py prune $(if $(filter 1,$(APPLY)),--apply,)

archive-evidence:
	@test -n "$(RUN)" || { echo 'Set RUN=<raw evidence directory>'; exit 2; }
	$(PY_ENV) $(PYTHON) scripts/thesis_runs.py archive $(RUN) $(if $(filter 1,$(APPLY)),--apply,)

thesis-release:
	@if [ -z "$(NAME)" ]; then echo 'Set NAME=<milestone-name>'; exit 2; fi
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py release --name $(NAME) --snapshot $(THESIS_SNAPSHOT)

clean-generated:
	rm -rf build .pytest_cache native/quest_cpu/bin native/quest_gpu/bin native/quest_gpu/build
	find . -path ./external -prune -o \( -name "__pycache__" -type d -o -name "*.pyc" -type f \) -print -exec rm -rf {} +
ifeq ($(CLEAN_RUNS),1)
	rm -rf runs
else
	@echo 'Keeping runs/. Use CLEAN_RUNS=1 to remove benchmark evidence and comparisons.'
endif
