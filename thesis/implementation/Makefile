PYTHON ?= ../.venv/bin/python
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
GPU_VERIFY ?= quest-hip
CLEAN_RUNS ?= 0
THESIS_PACK ?= runs/comparisons/research_pack/latest
THESIS_SNAPSHOT ?= thesis_results/current
APPLY ?= 0

PUBLIC_TARGETS := \
	help \
	build-quest-cpu \
	doctor \
	bench-cpu \
	bench-gpu \
	bench-upmem-sim \
	research-plan \
	thesis-run \
	thesis-promote \
	thesis-verify \
	thesis-report \
	list-runs \
	thesis-clean \
	thesis-release \
	clean-generated

.PHONY: $(PUBLIC_TARGETS)

help:
	@printf '%s\n' \
		'Thesis evidence shortcuts:' \
		'  make help              Show this command summary' \
		'  make doctor             Check local benchmark prerequisites without creating evidence' \
		'  make build-quest-cpu    Build the native QuEST CPU runner' \
		'  make bench-cpu          Run the CPU evidence suite' \
		'  make bench-gpu          Verify the GPU and run the GPU evidence suite' \
		'  make bench-upmem-sim    Check UPMEM simulator and run its evidence suite' \
		'  make research-plan      Show the research benchmark plan' \
		'  make thesis-run BENCH_CPU_THREADS=<physical cores>' \
		'                           Run the controlled thesis benchmark matrix' \
		'  make thesis-promote     Promote the latest research pack into tracked thesis_results/current' \
		'  make thesis-verify      Verify tracked thesis results without rerunning benchmarks' \
		'  make thesis-report      Regenerate tracked tables and plots from thesis_results/current' \
		'  make thesis-clean       Preview stale generated run cleanup; add APPLY=1 to remove' \
		'  make thesis-release NAME=<name>  Freeze current results as a named milestone' \
		'  make list-runs          Show generated evidence/comparisons and current selection' \
		'  make clean-generated    Remove generated build/cache files; keeps runs/ by default'

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

research-plan:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py plan

thesis-run: build-quest-cpu
	@test -n "$(BENCH_CPU_THREADS)" || { echo 'Set BENCH_CPU_THREADS to the physical CPU core count, for example: BENCH_CPU_THREADS=6 make thesis-run' >&2; exit 2; }
	@case "$(BENCH_CPU_THREADS)" in *[!0-9]*|'') echo 'BENCH_CPU_THREADS must be a positive integer' >&2; exit 2;; esac
	@test "$(BENCH_CPU_THREADS)" -gt 0 || { echo 'BENCH_CPU_THREADS must be a positive integer' >&2; exit 2; }
	$(PY_ENV) $(THESIS_BENCH_ENV) $(PYTHON) scripts/research_benchmark_pack.py run --full

thesis-promote:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py promote --pack $(THESIS_PACK) --out $(THESIS_SNAPSHOT)

thesis-verify:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py verify --snapshot $(THESIS_SNAPSHOT)

thesis-report:
	$(PY_ENV) $(PYTHON) scripts/thesis_snapshot.py report --snapshot $(THESIS_SNAPSHOT)

list-runs:
	$(PY_ENV) $(PYTHON) scripts/thesis_runs.py list

thesis-clean:
	$(PY_ENV) $(PYTHON) scripts/thesis_runs.py prune $(if $(filter 1,$(APPLY)),--apply,)

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
