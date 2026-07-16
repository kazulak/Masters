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
UPMEM_HW_MVP_SUITE ?= configs/suites/upmem_hardware_mvp.yml
UPMEM_HW_GENERIC_MVP_SUITE ?= configs/suites/upmem_hardware_generic_mvp.yml
UPMEM_HW_TASKGRAPH_SUITE ?= configs/suites/upmem_hardware_taskgraph_correctness.yml
UPMEM_HW_TASKGRAPH_RUN ?= runs/evidence/upmem_hardware_taskgraph_correctness/upmem_hw_taskgraph/latest
UPMEM_HW_TASKGRAPH_STUDY_SUITE ?= configs/suites/upmem_hardware_taskgraph_path_quantization.yml
UPMEM_HW_TASKGRAPH_STUDY_RUN ?= runs/evidence/upmem_hardware_taskgraph_path_quantization/upmem_hw_taskgraph_study/latest
UPMEM_HW_TASKGRAPH_RESIDENT_SUITE ?= configs/suites/upmem_hardware_taskgraph_resident_path_quantization.yml
UPMEM_HW_TASKGRAPH_RESIDENT_RUN ?= runs/evidence/upmem_hardware_taskgraph_resident_path_quantization/upmem_hw_taskgraph_resident/latest
EVIDENCE_INBOX ?= runs/inbox/eth
GPU_VERIFY ?= quest-hip
CLEAN_RUNS ?= 0
THESIS_PACK ?= runs/comparisons/research_pack/latest
THESIS_SNAPSHOT ?= thesis_results/current
APPLY ?= 0

PUBLIC_TARGETS := \
	help \
	setup \
	build-quest-cpu \
	doctor \
	bench-cpu \
	bench-gpu \
	bench-upmem-sim \
	upmem-hw-mvp-plan \
	upmem-hw-mvp \
	upmem-hw-generic-plan \
	upmem-hw-generic-mvp \
	upmem-hw-taskgraph-plan \
	upmem-hw-taskgraph \
	upmem-hw-taskgraph-report \
	upmem-hw-taskgraph-study-plan \
	upmem-hw-taskgraph-study \
	upmem-hw-taskgraph-study-report \
	upmem-hw-taskgraph-resident-plan \
	upmem-hw-taskgraph-resident \
	upmem-hw-taskgraph-resident-report \
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
		'  make doctor             Check local benchmark prerequisites without creating evidence' \
		'  make build-quest-cpu    Build the native QuEST CPU runner' \
		'  make bench-cpu          Run the CPU evidence suite' \
		'  make bench-gpu          Verify the GPU and run the GPU evidence suite' \
		'  make bench-upmem-sim    Check UPMEM simulator and run its evidence suite' \
		'  make upmem-hw-mvp-plan  Prepare/build the guarded one-DPU hardware dense MVP; never allocates a DPU' \
		'  make upmem-hw-mvp       Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; runs the one-DPU hardware dense MVP' \
		'  make upmem-hw-generic-plan  Prepare/build the guarded generic TaskGraph MVP; never allocates a DPU' \
		'  make upmem-hw-generic-mvp   Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; runs one real generic TaskGraph contraction' \
		'  make upmem-hw-taskgraph-plan  Prepare/build the physical TaskGraph correctness route; never allocates a DPU' \
		'  make upmem-hw-taskgraph  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; runs correctness-only TaskGraph evidence' \
		'  make upmem-hw-taskgraph-report  Generate correctness/quantization plots from the latest physical TaskGraph evidence' \
		'  make upmem-hw-taskgraph-study-plan  Prepare/build the physical TaskGraph path/quantization study; never allocates a DPU' \
		'  make upmem-hw-taskgraph-study  Requires UPMEM_ALLOW_PHYSICAL_HARDWARE=1; runs the physical TaskGraph path/quantization study' \
		'  make upmem-hw-taskgraph-study-report  Generate the labeled report from the latest physical TaskGraph study evidence' \
		'  make upmem-hw-taskgraph-resident-plan  Show the future resident-route suite/config without running a resident runtime' \
		'  make upmem-hw-taskgraph-resident  Reserved future resident-route execution target; runtime is not implemented' \
		'  make upmem-hw-taskgraph-resident-report  Generate a report from explicitly supplied future resident-route evidence' \
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

upmem-hw-mvp-plan:
	$(BENCH) upmem-hardware-mvp --suite $(UPMEM_HW_MVP_SUITE) --prepare-only --build

upmem-hw-mvp:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to run physical UPMEM hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-mvp --suite $(UPMEM_HW_MVP_SUITE) --execute
	$(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py check-upmem-hardware runs/latest

upmem-hw-generic-plan:
	$(BENCH) upmem-hardware-generic-mvp --suite $(UPMEM_HW_GENERIC_MVP_SUITE) --prepare-only --build

upmem-hw-generic-mvp:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to run physical UPMEM hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-generic-mvp --suite $(UPMEM_HW_GENERIC_MVP_SUITE) --execute

upmem-hw-taskgraph-plan:
	$(BENCH) upmem-hardware-taskgraph --suite $(UPMEM_HW_TASKGRAPH_SUITE) --prepare-only --build

upmem-hw-taskgraph:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to run physical UPMEM hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-taskgraph --suite $(UPMEM_HW_TASKGRAPH_SUITE) --execute

upmem-hw-taskgraph-report:
	@test -f "$(UPMEM_HW_TASKGRAPH_RUN)/normalized_records.jsonl" || { echo 'Set UPMEM_HW_TASKGRAPH_RUN=<physical TaskGraph evidence directory>' >&2; exit 2; }
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py report --input $(UPMEM_HW_TASKGRAPH_RUN) --label upmem_hw_taskgraph

upmem-hw-taskgraph-study-plan:
	$(BENCH) upmem-hardware-taskgraph-study --suite $(UPMEM_HW_TASKGRAPH_STUDY_SUITE) --prepare-only --build

upmem-hw-taskgraph-study:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1 to run physical UPMEM hardware' >&2; exit 2; }
	$(BENCH) upmem-hardware-taskgraph-study --suite $(UPMEM_HW_TASKGRAPH_STUDY_SUITE) --execute

upmem-hw-taskgraph-study-report:
	@test -f "$(UPMEM_HW_TASKGRAPH_STUDY_RUN)/normalized_records.jsonl" || { echo 'Set UPMEM_HW_TASKGRAPH_STUDY_RUN=<physical TaskGraph study evidence directory>' >&2; exit 2; }
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py report --input $(UPMEM_HW_TASKGRAPH_STUDY_RUN) --label upmem_hw_taskgraph_study

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
	@test "$(THESIS_SNAPSHOT)" != "thesis_results/current" && test "$(notdir $(THESIS_SNAPSHOT))" != "current" || { echo 'Set THESIS_SNAPSHOT=thesis_results/<named-snapshot>; historical promotion cannot target current' >&2; exit 2; }
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py promote --historical --pack $(THESIS_PACK) --out $(THESIS_SNAPSHOT)

thesis-verify:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py verify --snapshot $(THESIS_SNAPSHOT)

thesis-report:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py report --snapshot $(THESIS_SNAPSHOT)

upmem-hw-taskgraph-resident-plan:
	@printf '%s\n' 'Future resident TaskGraph route plan (no runtime invoked):' '  suite=$(UPMEM_HW_TASKGRAPH_RESIDENT_SUITE)' '  run=$(UPMEM_HW_TASKGRAPH_RESIDENT_RUN)' '  route=upmem_tn_hardware_taskgraph_resident'

upmem-hw-taskgraph-resident:
	@echo 'Resident TaskGraph runtime is reserved for a future implementation; no runtime was invoked.' >&2
	@exit 2

upmem-hw-taskgraph-resident-report:
	@test -f "$(UPMEM_HW_TASKGRAPH_RESIDENT_RUN)/normalized_records.jsonl" || { echo 'Set UPMEM_HW_TASKGRAPH_RESIDENT_RUN=<future resident-route evidence directory>' >&2; exit 2; }
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
