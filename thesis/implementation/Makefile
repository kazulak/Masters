PYTHON ?= ../.venv/bin/python
PYTHON_BIN_DIR := $(abspath $(dir $(PYTHON)))
export PATH := $(PYTHON_BIN_DIR):$(PATH)

PY_ENV := PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
CLI := $(PY_ENV) $(PYTHON) -m quantum_bench.cli
RUN_STAMP := $(shell date +%Y-%m-%d_%H-%M-%S)

CONFIG ?= configs/tn_benchmark_reset.yml
PHYSICAL_CONFIG ?= configs/tn_benchmark_physical.yml
OUTPUT ?= runs/reset/$(RUN_STAMP)
INPUT ?=
REPORT_OUTPUT ?= runs/reports/$(RUN_STAMP)
UPMEM_TASKLETS ?= 1
CLEAN_RUNS ?= 0

PUBLIC_TARGETS := \
	help setup doctor test \
	build-quest-cpu build-upmem-runtime \
	plan run report verify qualify \
	clean-generated

.PHONY: $(PUBLIC_TARGETS)

help:
	@printf '%s\n' \
		'Thesis tensor-network benchmark:' \
		'  make setup                         Create/update ../.venv, build QuEST CPU, run doctor' \
		'  make test                          Run the active software test suite' \
		'  make plan CONFIG=<yaml>             Compile a deterministic experiment plan; no execution' \
		'  make run CONFIG=<yaml>              Run CPU/TN/simulator routes into a new evidence directory' \
		'  make report INPUT=<evidence-dir>    Generate tables and plots without executing workloads' \
		'  make verify INPUT=<evidence-dir>    Verify canonical evidence and identities' \
		'  make build-upmem-runtime             Build the active ABI-v4 host and DPU binaries' \
		'  UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make qualify' \
		'                                      Run only explicit physical routes from a clean worktree' \
		'' \
		'Overrides: OUTPUT=<dir>, REPORT_OUTPUT=<dir>, PHYSICAL_CONFIG=<yaml>, UPMEM_TASKLETS=<n>, PYTHON=<path>'

setup:
	@python3 scripts/bootstrap_env.py
	$(MAKE) -C native/quest_cpu clean-quest
	$(MAKE) build-quest-cpu
	$(MAKE) doctor

doctor:
	$(PY_ENV) $(PYTHON) scripts/doctor.py

test:
	$(PY_ENV) $(PYTHON) -m pytest -q

build-quest-cpu:
	$(MAKE) -C native/quest_cpu

build-upmem-runtime:
	$(MAKE) -C native/upmem/runtime v4 NR_TASKLETS=$(UPMEM_TASKLETS)

plan:
	$(CLI) plan --config $(CONFIG) --output $(OUTPUT)

run:
	$(CLI) run --config $(CONFIG) --output $(OUTPUT)

report:
	@test -n "$(INPUT)" || { echo 'Set INPUT=<canonical evidence directory>' >&2; exit 2; }
	$(CLI) report --input $(INPUT) --output $(REPORT_OUTPUT)

verify:
	@test -n "$(INPUT)" || { echo 'Set INPUT=<canonical evidence directory>' >&2; exit 2; }
	$(CLI) verify --input $(INPUT)

qualify:
	@test "$(UPMEM_ALLOW_PHYSICAL_HARDWARE)" = "1" || { echo 'Set UPMEM_ALLOW_PHYSICAL_HARDWARE=1' >&2; exit 2; }
	$(CLI) qualify --config $(PHYSICAL_CONFIG) --output $(OUTPUT) --allow-physical

clean-generated:
	rm -rf build .pytest_cache native/quest_cpu/bin native/quest_gpu/bin native/quest_gpu/build
	find . -path ./external -prune -o \( -name '__pycache__' -type d -o -name '*.pyc' -type f \) -print -exec rm -rf {} +
ifeq ($(CLEAN_RUNS),1)
	rm -rf runs
else
	@echo 'Keeping runs/. Use CLEAN_RUNS=1 to remove generated run data.'
endif
