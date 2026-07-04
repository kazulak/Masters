PYTHON ?= ../.venv/bin/python
PY_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
BENCH := $(PY_ENV) $(PYTHON) -m quantum_bench.bench

CPU_SUITE ?= configs/suites/cpu_evidence.yml
GPU_SUITE ?= configs/suites/gpu_evidence.yml
UPMEM_SIM_SUITE ?= configs/suites/upmem_sim_evidence.yml
GPU_VERIFY ?= quest-hip
CLEAN_RUNS ?= 0

.PHONY: help build-quest-cpu doctor bench-cpu bench-gpu bench-upmem-sim thesis-benchmark research-plan research-benchmarks research-report report-latest compare-latest clean-generated

help:
	@printf '%s\n' \
		'Thesis evidence shortcuts:' \
		'  make doctor             Check local benchmark prerequisites without creating evidence' \
		'  make build-quest-cpu    Build the native QuEST CPU runner' \
		'  make bench-cpu          Run CPU/QuEST/Quimb evidence suite' \
		'  make bench-gpu          Verify QuEST HIP GPU and run GPU execution-only suite' \
		'  make bench-upmem-sim    Check UPMEM SDK simulator and run strict UPMEM simulator suite' \
		'  make thesis-benchmark   Run CPU + UPMEM simulator evidence and compare them' \
		'  make research-plan      Print research benchmark pack commands and checks' \
		'  make research-benchmarks Create a lightweight research pack; set RUN_RESEARCH=1 for long runs' \
		'  make research-report    Build a research pack from existing evidence' \
		'  make report-latest      Regenerate reports for runs/latest' \
		'  make compare-latest     Write a derived comparison under runs/comparisons' \
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

thesis-benchmark:
	@set -e; \
	timestamp=$$(date +%Y%m%d_%H%M%S); \
	make bench-cpu; \
	cpu_run=$$(readlink -f runs/latest); \
	cpu_suite=$$($(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py suite-id "$$cpu_run"); \
	cpu_report="runs/comparisons/$$cpu_suite/report_run/$${timestamp}_cpu"; \
	$(BENCH) report-run --input "$$cpu_run" --out "$$cpu_report"; \
	make bench-upmem-sim; \
	upmem_run=$$(readlink -f runs/latest); \
	upmem_suite=$$($(PY_ENV) $(PYTHON) scripts/evidence_shortcuts.py suite-id "$$upmem_run"); \
	upmem_report="runs/comparisons/$$upmem_suite/report_run/$${timestamp}_upmem"; \
	$(BENCH) report-run --input "$$upmem_run" --out "$$upmem_report"; \
	compare_out="runs/comparisons/thesis_benchmark/cpu_upmem/$$timestamp"; \
	$(BENCH) compare-results --inputs "$$cpu_run" "$$upmem_run" --comparison-type thesis_benchmark_cpu_upmem --out "$$compare_out"; \
	printf '%s\n' "CPU evidence: $$cpu_run" "UPMEM evidence: $$upmem_run" "CPU report: $$cpu_report" "UPMEM report: $$upmem_report" "Comparison: $$compare_out"

research-plan:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py plan

research-benchmarks:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py run $(if $(filter 1,$(RUN_RESEARCH)),--full,)

research-report:
	$(PY_ENV) $(PYTHON) scripts/research_benchmark_pack.py report

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
