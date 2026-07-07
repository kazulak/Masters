PYTHON ?= ../.venv/bin/python
PY_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
BENCH := $(PY_ENV) $(PYTHON) -m quantum_bench.bench

CPU_SUITE ?= configs/suites/cpu_evidence.yml
GPU_SUITE ?= configs/suites/gpu_evidence.yml
UPMEM_SIM_SUITE ?= configs/suites/upmem_sim_evidence.yml
THESIS_FULL_STATE_SUITE ?= configs/suites/manual/thesis_full_state_cpu_gpu.yml
THESIS_TN_QUIMB_SUITE ?= configs/suites/manual/thesis_cpu_tn_quimb.yml
THESIS_TN_QUANT_SUITE ?= configs/suites/manual/thesis_tn_paths_quantization.yml
THESIS_UPMEM_SUITE ?= configs/suites/manual/thesis_upmem_quantization_boundary.yml
PARALLELISM_SLICING_FRONTIER_SUITE ?= configs/suites/diagnostics/cpu_slicing_vs_frontier_quick.yml
PARALLELISM_HYBRID_SUITE ?= configs/suites/diagnostics/cpu_hybrid_quick.yml
PARALLELISM_UPMEM_ASSIGNMENT_SUITE ?= configs/suites/upmem_sim_evidence.yml
PARALLELISM_DPU_GROUPS ?= 4
PARALLELISM_INCLUDE_UPMEM_FRONTIER ?= 0
PARALLELISM_UPMEM_FRONTIER_CASE ?= bell_2q
PARALLELISM_UPMEM_FRONTIER_QUANTIZATION ?= per_task_input_quantize
GPU_VERIFY ?= quest-hip
CLEAN_RUNS ?= 0

.PHONY: help build-quest-cpu doctor bench-cpu bench-gpu bench-upmem-sim thesis-benchmark thesis-results thesis-results-resume thesis-report parallelism-report research-plan research-benchmarks research-report report-latest compare-latest clean-generated

help:
	@printf '%s\n' \
		'Thesis evidence shortcuts:' \
		'  make doctor             Check local benchmark prerequisites without creating evidence' \
		'  make build-quest-cpu    Build the native QuEST CPU runner' \
		'  make bench-cpu          Run CPU/QuEST/Quimb evidence suite' \
		'  make bench-gpu          Verify QuEST HIP GPU and run GPU execution-only suite' \
		'  make bench-upmem-sim    Check UPMEM SDK simulator and run strict UPMEM simulator suite' \
		'  make thesis-benchmark   Run CPU + UPMEM simulator evidence and compare them' \
		'  make thesis-results     Run thesis CPU/GPU + TN + UPMEM evidence and generate report' \
		'  make thesis-results-resume FULL_STATE_RUN=...  Continue after CPU/GPU evidence exists' \
		'  make thesis-report      Build thesis report from explicit THESIS_INPUTS evidence paths' \
		'  make parallelism-report Build current slicing/frontier/hybrid/modeled-UPMEM comparison' \
		'  make parallelism-report PARALLELISM_INCLUDE_UPMEM_FRONTIER=1  Include executed UPMEM SDK simulator frontier row' \
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

thesis-results:
	@set -e; \
	timestamp=$$(date +%Y%m%d_%H%M%S); \
	printf '%s\n' 'Running thesis full-state CPU/GPU evidence...'; \
	$(BENCH) simulation-backend-probe --verify-gpu $(GPU_VERIFY); \
	$(BENCH) simulation-backend-compare --suite $(THESIS_FULL_STATE_SUITE) --artifact-retention compact; \
	full_state_run=$$(readlink -f runs/latest); \
	printf '%s\n' "Full-state CPU/GPU evidence: $$full_state_run"; \
	printf '%s\n' 'Running thesis CPU TN path/quantization evidence...'; \
	$(BENCH) simulation-backend-compare --suite $(THESIS_TN_QUIMB_SUITE) --artifact-retention compact; \
	tn_quimb_run=$$(readlink -f runs/latest); \
	printf '%s\n' "Quimb CPU TN evidence: $$tn_quimb_run"; \
	$(BENCH) simulation-backend-compare --suite $(THESIS_TN_QUANT_SUITE) --artifact-retention compact; \
	tn_quant_run=$$(readlink -f runs/latest); \
	printf '%s\n' "TN path/quantization diagnostic evidence: $$tn_quant_run"; \
	printf '%s\n' 'Running thesis UPMEM SDK simulator boundary evidence...'; \
	$(BENCH) upmem-mvp-benchmark --suite $(THESIS_UPMEM_SUITE) --policies generic-only --quantization-modes none,per_task_input_quantize --execute-external; \
	upmem_run=$$(readlink -f runs/latest); \
	printf '%s\n' "UPMEM SDK simulator evidence: $$upmem_run"; \
	out_dir="runs/comparisons/thesis/$$timestamp"; \
	$(PY_ENV) $(PYTHON) scripts/thesis_report.py --inputs "$$full_state_run" "$$tn_quimb_run" "$$tn_quant_run" "$$upmem_run" --out "$$out_dir"; \
	printf '%s\n' "Thesis report: $$out_dir" "Plots: $$out_dir/plots"

thesis-results-resume:
	@if [ -z "$(FULL_STATE_RUN)" ]; then \
		echo 'Set FULL_STATE_RUN=runs/evidence/thesis_full_state_cpu_gpu/simulation_backend_compare/<run_id>'; \
		exit 2; \
	fi
	@set -e; \
	timestamp=$$(date +%Y%m%d_%H%M%S); \
	full_state_run=$$(readlink -f "$(FULL_STATE_RUN)"); \
	printf '%s\n' "Reusing full-state CPU/GPU evidence: $$full_state_run"; \
	printf '%s\n' 'Running thesis CPU TN path/quantization evidence...'; \
	$(BENCH) simulation-backend-compare --suite $(THESIS_TN_QUIMB_SUITE) --artifact-retention compact; \
	tn_quimb_run=$$(readlink -f runs/latest); \
	printf '%s\n' "Quimb CPU TN evidence: $$tn_quimb_run"; \
	$(BENCH) simulation-backend-compare --suite $(THESIS_TN_QUANT_SUITE) --artifact-retention compact; \
	tn_quant_run=$$(readlink -f runs/latest); \
	printf '%s\n' "TN path/quantization diagnostic evidence: $$tn_quant_run"; \
	printf '%s\n' 'Running thesis UPMEM SDK simulator boundary evidence...'; \
	$(BENCH) upmem-mvp-benchmark --suite $(THESIS_UPMEM_SUITE) --policies generic-only --quantization-modes none,per_task_input_quantize --execute-external; \
	upmem_run=$$(readlink -f runs/latest); \
	printf '%s\n' "UPMEM SDK simulator evidence: $$upmem_run"; \
	out_dir="runs/comparisons/thesis/$$timestamp"; \
	$(PY_ENV) $(PYTHON) scripts/thesis_report.py --inputs "$$full_state_run" "$$tn_quimb_run" "$$tn_quant_run" "$$upmem_run" --out "$$out_dir"; \
	printf '%s\n' "Thesis report: $$out_dir" "Plots: $$out_dir/plots"

thesis-report:
	@if [ -z "$(THESIS_INPUTS)" ]; then \
		echo 'Set THESIS_INPUTS="runs/evidence/... runs/evidence/..."'; \
		exit 2; \
	fi
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	out_dir="$${THESIS_OUT:-runs/comparisons/thesis/$$timestamp}"; \
	$(PY_ENV) $(PYTHON) scripts/thesis_report.py --inputs $(THESIS_INPUTS) --out "$$out_dir"

parallelism-report: build-quest-cpu
	@set -e; \
	timestamp=$$(date +%Y%m%d_%H%M%S); \
	printf '%s\n' 'Running CPU slicing/frontier diagnostic evidence...'; \
	$(BENCH) simulation-backend-compare --suite $(PARALLELISM_SLICING_FRONTIER_SUITE) --artifact-retention compact; \
	slicing_frontier_run=$$(readlink -f runs/latest); \
	printf '%s\n' "CPU slicing/frontier evidence: $$slicing_frontier_run"; \
	printf '%s\n' 'Running CPU hybrid diagnostic evidence...'; \
	$(BENCH) simulation-backend-compare --suite $(PARALLELISM_HYBRID_SUITE) --artifact-retention compact; \
	hybrid_run=$$(readlink -f runs/latest); \
	printf '%s\n' "CPU hybrid evidence: $$hybrid_run"; \
	printf '%s\n' 'Running modeled UPMEM DPU assignment evidence...'; \
	$(BENCH) upmem-multi-dpu-assignment --suite $(PARALLELISM_UPMEM_ASSIGNMENT_SUITE) --dpu-groups $(PARALLELISM_DPU_GROUPS); \
	upmem_assignment_run=$$(readlink -f runs/latest); \
	printf '%s\n' "UPMEM modeled assignment evidence: $$upmem_assignment_run"; \
	inputs="$$slicing_frontier_run $$hybrid_run $$upmem_assignment_run"; \
	if [ "$(PARALLELISM_INCLUDE_UPMEM_FRONTIER)" = "1" ]; then \
		printf '%s\n' 'Running UPMEM SDK simulator frontier evidence...'; \
		$(BENCH) upmem-taskgraph-frontier-runtime --case $(PARALLELISM_UPMEM_FRONTIER_CASE) --policy generic-only --quantization-mode $(PARALLELISM_UPMEM_FRONTIER_QUANTIZATION) --dpu-groups $(PARALLELISM_DPU_GROUPS) --frontier-worker-count 1 --execute-external; \
		upmem_frontier_run=$$(readlink -f runs/latest); \
		printf '%s\n' "UPMEM SDK simulator frontier evidence: $$upmem_frontier_run"; \
		inputs="$$inputs $$upmem_frontier_run"; \
	fi; \
	out_dir="runs/comparisons/parallelism/current/$$timestamp"; \
	$(BENCH) compare-results --inputs $$inputs --comparison-type parallelism_evidence --out "$$out_dir"; \
	printf '%s\n' "Parallelism comparison: $$out_dir" "Summary: $$out_dir/parallelism_comparison_summary.md"

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
