PYTHON ?= ../.venv/bin/python
PY_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
BENCH := $(PY_ENV) $(PYTHON) -m quantum_bench.bench

CPU_SUITE ?= configs/suites/cpu_evidence.yml
GPU_SUITE ?= configs/suites/gpu_evidence.yml
UPMEM_SIM_SUITE ?= configs/suites/upmem_sim_evidence.yml
GPU_VERIFY ?= quest-hip
CLEAN_RUNS ?= 0
THESIS_PACK ?= runs/comparisons/research_pack/latest
THESIS_SNAPSHOT ?= thesis_results/current
APPLY ?= 0

.PHONY: help build-quest-cpu doctor bench-cpu bench-gpu bench-upmem-sim thesis-report research-plan clean-generated thesis-run thesis-promote thesis-verify thesis-clean thesis-release list-runs

help:
	@printf '%s\n' \
		'Thesis evidence shortcuts:' \
		'  make doctor             Check local benchmark prerequisites without creating evidence' \
		'  make thesis-run         Run the complete local thesis benchmark matrix' \
		'  make thesis-promote     Promote the latest research pack into tracked thesis_results/current' \
		'  make thesis-report      Regenerate tracked tables and plots from thesis_results/current' \
		'  make thesis-clean       Preview stale generated run cleanup; add APPLY=1 to remove' \
		'  make thesis-release NAME=<name>  Freeze current results as a named milestone' \
		'  make list-runs          Show generated evidence/comparisons and current selection' \
		'  make build-quest-cpu    Build the native QuEST CPU runner' \
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
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py run --full

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
