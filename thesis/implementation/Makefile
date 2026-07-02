PYTHON ?= ../.venv/bin/python
PY_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
BENCH := $(PY_ENV) $(PYTHON) -m quantum_bench.bench

CPU_SUITE ?= configs/suites/simulation_backend_compare_compute_medium.yml
GPU_SUITE ?= configs/suites/simulation_backend_compare_gpu_execution_only.yml
UPMEM_SIM_SUITE ?= configs/suites/simulation_backend_compare_upmem_sdk_simulator.yml
GPU_VERIFY ?= quest-hip
CLEAN_RUNS ?= 0

.PHONY: help bench-cpu bench-gpu bench-upmem-sim report-latest compare-latest clean-generated

help:
	@printf '%s\n' \
		'Thesis evidence shortcuts:' \
		'  make bench-cpu          Run CPU/QuEST/Quimb evidence suite' \
		'  make bench-gpu          Verify QuEST HIP GPU and run GPU execution-only suite' \
		'  make bench-upmem-sim    Check UPMEM SDK simulator and run strict UPMEM simulator suite' \
		'  make report-latest      Regenerate reports for runs/latest' \
		'  make compare-latest     Write a derived comparison under runs/comparisons' \
		'  make clean-generated    Remove generated build/cache files; keeps runs/ by default'

bench-cpu:
	$(BENCH) simulation-backend-compare --suite $(CPU_SUITE) --artifact-retention compact

bench-gpu:
	$(BENCH) simulation-backend-probe --verify-gpu $(GPU_VERIFY)
	$(BENCH) simulation-backend-compare --suite $(GPU_SUITE) --artifact-retention compact
	$(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py check-gpu runs/latest

bench-upmem-sim:
	$(BENCH) upmem-env-check --run-sample --target simulator
	$(BENCH) simulation-backend-compare --suite $(UPMEM_SIM_SUITE) --artifact-retention compact
	$(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py check-upmem runs/latest

report-latest:
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	suite_id=$$($(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py suite-id runs/latest); \
	out_dir="runs/comparisons/$$suite_id/report_run/$$timestamp"; \
	$(BENCH) report-run --input runs/latest --out "$$out_dir"

compare-latest:
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	suite_id=$$($(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py suite-id runs/latest); \
	out_dir="runs/comparisons/$$suite_id/latest_single_run/$$timestamp"; \
	$(BENCH) compare-results --inputs runs/latest --comparison-type latest_single_run --out "$$out_dir"

clean-generated:
	rm -rf build .pytest_cache native/quest_cpu/bin native/quest_gpu/bin native/quest_gpu/build
	find . -path ./external -prune -o \( -name "__pycache__" -type d -o -name "*.pyc" -type f \) -print -exec rm -rf {} +
ifeq ($(CLEAN_RUNS),1)
	rm -rf runs
else
	@echo 'Keeping runs/. Use CLEAN_RUNS=1 to remove benchmark evidence and comparisons.'
endif
