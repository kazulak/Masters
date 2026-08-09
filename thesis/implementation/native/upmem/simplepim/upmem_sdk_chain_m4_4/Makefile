SHELL := /bin/sh

SIMPLEPIM_ROOT ?= ../../../../external/SimplePIM
SIMPLEPIM_SOURCE_COMMIT ?= 1d639c53532555f01e9f71d872e7712b166d6cba
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc
UPMEM_ALLOW_PHYSICAL_HARDWARE ?= 0
EXECUTE_TIMEOUT_S ?= 120
OPERANDS_FILE ?=
INPUT_SHA256 ?=
GRAPH_BINDING ?=
GRAPH_BINDING_SHA256 ?=
UPMEM_PREFIX ?= $(shell dpu-pkg-config --variable=prefix dpu)
DPU_SYSLIB_INCLUDE ?= $(UPMEM_PREFIX)/share/upmem/include/syslib

ROUTE_ROOT := $(abspath .)
STAGE_ROOT ?= build/simplepim_chain_m4_4/staged
STAGE_LIB := $(STAGE_ROOT)/lib
STAGE_BENCH := $(STAGE_ROOT)/benchmarks/chain_m4_4
STAGE_MANIFEST := $(STAGE_ROOT)/simplepim_stage_manifest.json
HOST_BINARY := $(STAGE_BENCH)/bin/chain_host
INIT_BINARY := $(STAGE_BENCH)/bin/dpu_init_binary
EXECUTE_RESPONSE := $(abspath build/simplepim_chain_m4_4/execute_response.json)

.PHONY: all stage build execute clean

all: build

stage: $(STAGE_MANIFEST)

build: $(HOST_BINARY) $(INIT_BINARY)

$(STAGE_MANIFEST): simplepim_rank1_hardening.patch simplepim_chain_tasklets.patch benchmark/product_i8_i8/map.h benchmark/product_i32_i8/map.h benchmark/reduce_i32/map_to_val_func.h benchmark/reduce_i32/init_combine_func.h
	set -eu; \
	test -d "$(SIMPLEPIM_ROOT)/.git" -o -f "$(SIMPLEPIM_ROOT)/.git"; \
	actual_commit=$$(git -C "$(SIMPLEPIM_ROOT)" rev-parse --verify HEAD); \
	test "$$actual_commit" = "$(SIMPLEPIM_SOURCE_COMMIT)" || { echo "SimplePIM commit mismatch" >&2; exit 1; }; \
	dirty=$$(git -C "$(SIMPLEPIM_ROOT)" status --porcelain --untracked-files=all); test -z "$$dirty" || { echo "SimplePIM source is dirty" >&2; exit 1; }; \
	patch_sha256=$$(sha256sum simplepim_rank1_hardening.patch | cut -d ' ' -f1); \
	rm -rf "$(STAGE_ROOT)"; mkdir -p "$(STAGE_LIB)" "$(STAGE_BENCH)"; \
	cp -R "$(SIMPLEPIM_ROOT)/lib/." "$(STAGE_LIB)/"; \
	(cd "$(STAGE_ROOT)" && patch --batch --forward --strip=1 < "$(ROUTE_ROOT)/simplepim_rank1_hardening.patch"); \
	(cd "$(STAGE_ROOT)" && patch --batch --forward --strip=1 < "$(ROUTE_ROOT)/simplepim_chain_tasklets.patch"); \
	cp -R benchmark/. "$(STAGE_BENCH)/"; mkdir -p "$(STAGE_BENCH)/bin"; \
	tasklet_patch_sha256=$$(sha256sum simplepim_chain_tasklets.patch | cut -d ' ' -f1); \
	printf '{"schema_version":"simplepim_chain_m4_4_stage_v1","source_commit":"%s","expected_source_commit":"%s","source_worktree_dirty":false,"hardening_patch_sha256":"%s","tasklet_patch_sha256":"%s","patches_applied":true,"effective_operator_tasklets":1,"effective_initialization_tasklets":1}\n' "$$actual_commit" "$(SIMPLEPIM_SOURCE_COMMIT)" "$$patch_sha256" "$$tasklet_patch_sha256" > "$(STAGE_MANIFEST)"

$(HOST_BINARY): host.c $(STAGE_MANIFEST) | $(STAGE_BENCH)/bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -fopenmp -I$(STAGE_LIB) -I$(STAGE_LIB)/management -I$(STAGE_LIB)/processing -I$(DPU_SYSLIB_INCLUDE) -DNR_TASKLETS=1 -DTHESIS_SIMPLEPIM_OPERATOR_TASKLETS=1 -o $@ host.c \
		$(STAGE_LIB)/communication/CommHelper.c $(STAGE_LIB)/communication/CommOps.c \
		$(STAGE_LIB)/management/Management.c $(STAGE_LIB)/management/SmallTableInit.c \
		$(STAGE_LIB)/processing/ProcessingHelperHost.c $(STAGE_LIB)/processing/map/Map.c \
		$(STAGE_LIB)/processing/zip/Zip.c $(STAGE_LIB)/processing/gen_red/GenRed.c \
		$$(dpu-pkg-config --cflags --libs dpu) -ldl -lm -fopenmp

$(INIT_BINARY): $(STAGE_MANIFEST) | $(STAGE_BENCH)/bin
	$(DPU_CC) -O2 -DNR_TASKLETS=1 -o $@ $(STAGE_LIB)/management/SmallTableInit_dpu.c

$(STAGE_BENCH)/bin:
	mkdir -p $@

execute: build
	@test -n "$(OPERANDS_FILE)" || { echo 'Set OPERANDS_FILE=<768-byte int8 operand file>' >&2; exit 2; }
	@test -n "$(INPUT_SHA256)" || { echo 'Set INPUT_SHA256=<sha256 of OPERANDS_FILE>' >&2; exit 2; }
	@test -n "$(GRAPH_BINDING)" || { echo 'Set GRAPH_BINDING=<fixed M44 graph binding file>' >&2; exit 2; }
	@test -n "$(GRAPH_BINDING_SHA256)" || { echo 'Set GRAPH_BINDING_SHA256=<sha256 of GRAPH_BINDING>' >&2; exit 2; }
	cd "$(STAGE_BENCH)" && UPMEM_ALLOW_PHYSICAL_HARDWARE=1 timeout --signal=TERM --kill-after=5s "$(EXECUTE_TIMEOUT_S)s" ./bin/chain_host --mode execute --response "$(EXECUTE_RESPONSE)" --stage-manifest "../../simplepim_stage_manifest.json" --operands-file "$(abspath $(OPERANDS_FILE))" --input-sha256 "$(INPUT_SHA256)" --graph-binding "$(abspath $(GRAPH_BINDING))" --graph-binding-sha256 "$(GRAPH_BINDING_SHA256)"

clean:
	rm -rf build/simplepim_chain_m4_4
