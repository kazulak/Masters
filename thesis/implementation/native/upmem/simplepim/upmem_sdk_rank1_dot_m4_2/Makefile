SHELL := /bin/sh

SIMPLEPIM_ROOT ?= ../../../../external/SimplePIM
SIMPLEPIM_SOURCE_COMMIT ?= 1d639c53532555f01e9f71d872e7712b166d6cba
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc
INITIALIZATION_TASKLETS ?= 1
OPERATOR_TASKLETS ?= 12
UPMEM_ALLOW_PHYSICAL_HARDWARE ?= 0
EXECUTE_TIMEOUT_S ?= 120
UPMEM_PREFIX ?= $(shell dpu-pkg-config --variable=prefix dpu)
DPU_SYSLIB_INCLUDE ?= $(UPMEM_PREFIX)/share/upmem/include/syslib

ROUTE_ROOT := $(abspath .)
STAGE_ROOT ?= build/simplepim_rank1_dot_m4_2/staged
STAGE_LIB := $(STAGE_ROOT)/lib
STAGE_BENCH := $(STAGE_ROOT)/benchmarks/rank1_dot
STAGE_MANIFEST := $(STAGE_ROOT)/simplepim_stage_manifest.json
HOST_BINARY := $(STAGE_BENCH)/bin/rank1_dot_host
INIT_BINARY := $(STAGE_BENCH)/bin/dpu_init_binary
PARSER_RESPONSE := $(abspath build/simplepim_rank1_dot_m4_2/parser_response.json)
EXECUTE_RESPONSE := $(abspath build/simplepim_rank1_dot_m4_2/execute_response.json)

ifeq ($(INITIALIZATION_TASKLETS),1)
else
$(error rank-1 SimplePIM qualification requires exactly INITIALIZATION_TASKLETS=1)
endif

ifeq ($(OPERATOR_TASKLETS),12)
else
$(error rank-1 SimplePIM qualification requires exactly OPERATOR_TASKLETS=12)
endif

.PHONY: all stage build parser execute clean

all: build

stage: $(STAGE_MANIFEST)

build: $(HOST_BINARY) $(INIT_BINARY)

$(STAGE_MANIFEST): simplepim_rank1_hardening.patch benchmark/dot_funcs/map.h benchmark/dot_reduce_funcs/map_to_val_func.h benchmark/dot_reduce_funcs/init_combine_func.h
	set -eu; \
	test -d "$(SIMPLEPIM_ROOT)/.git" -o -f "$(SIMPLEPIM_ROOT)/.git"; \
	actual_commit=$$(git -C "$(SIMPLEPIM_ROOT)" rev-parse --verify HEAD); \
	test "$$actual_commit" = "$(SIMPLEPIM_SOURCE_COMMIT)" || { echo "SimplePIM commit mismatch: expected $(SIMPLEPIM_SOURCE_COMMIT), got $$actual_commit" >&2; exit 1; }; \
	dirty=$$(git -C "$(SIMPLEPIM_ROOT)" status --porcelain --untracked-files=all); \
	test -z "$$dirty" || { echo "SimplePIM submodule is dirty; refusing to stage it" >&2; printf '%s\n' "$$dirty" >&2; exit 1; }; \
	patch_sha256=$$(sha256sum simplepim_rank1_hardening.patch | cut -d ' ' -f1); \
	rm -rf "$(STAGE_ROOT)"; \
	mkdir -p "$(STAGE_LIB)" "$(STAGE_BENCH)"; \
	cp -R "$(SIMPLEPIM_ROOT)/lib/." "$(STAGE_LIB)/"; \
	(cd "$(STAGE_ROOT)" && patch --batch --forward --strip=1 < "$(ROUTE_ROOT)/simplepim_rank1_hardening.patch"); \
	cp -R benchmark/. "$(STAGE_BENCH)/"; \
	mkdir -p "$(STAGE_BENCH)/bin"; \
	lib_tree_sha256=$$(find "$(STAGE_LIB)" -type f -print | LC_ALL=C sort | while IFS= read -r file; do rel=$${file#$(STAGE_LIB)/}; printf '%s  %s\n' "$$(sha256sum "$$file" | cut -d ' ' -f1)" "$$rel"; done | sha256sum | cut -d ' ' -f1); \
	overlay_tree_sha256=$$(find "$(STAGE_BENCH)" -type f -print | LC_ALL=C sort | while IFS= read -r file; do rel=$${file#$(STAGE_BENCH)/}; printf '%s  %s\n' "$$(sha256sum "$$file" | cut -d ' ' -f1)" "$$rel"; done | sha256sum | cut -d ' ' -f1); \
	printf '{"schema_version":"simplepim_rank1_stage_v1","source_commit":"%s","expected_source_commit":"%s","source_worktree_dirty":false,"staged_source_tree_sha256":"%s","staged_overlay_tree_sha256":"%s","patch_sha256":"%s","patch_applied":true}\n' "$$actual_commit" "$(SIMPLEPIM_SOURCE_COMMIT)" "$$lib_tree_sha256" "$$overlay_tree_sha256" "$$patch_sha256" > "$(STAGE_MANIFEST)"

$(HOST_BINARY): host.c $(STAGE_MANIFEST) | $(STAGE_BENCH)/bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -fopenmp \
		-I$(STAGE_LIB) -I$(STAGE_LIB)/management -I$(STAGE_LIB)/processing -I$(DPU_SYSLIB_INCLUDE) \
		-DNR_TASKLETS=$(OPERATOR_TASKLETS) \
		-o $@ host.c \
		$(STAGE_LIB)/communication/CommHelper.c $(STAGE_LIB)/communication/CommOps.c \
		$(STAGE_LIB)/management/Management.c $(STAGE_LIB)/management/SmallTableInit.c \
		$(STAGE_LIB)/processing/ProcessingHelperHost.c $(STAGE_LIB)/processing/map/Map.c \
		$(STAGE_LIB)/processing/zip/Zip.c $(STAGE_LIB)/processing/gen_red/GenRed.c \
		$$(dpu-pkg-config --cflags --libs dpu) -ldl -lm -fopenmp

$(INIT_BINARY): $(STAGE_MANIFEST) | $(STAGE_BENCH)/bin
	$(DPU_CC) -O2 -DNR_TASKLETS=$(INITIALIZATION_TASKLETS) -o $@ $(STAGE_LIB)/management/SmallTableInit_dpu.c

$(STAGE_BENCH)/bin:
	mkdir -p $@

parser: build
	$(HOST_BINARY) --mode parser --response "$(PARSER_RESPONSE)" --stage-manifest "$(abspath $(STAGE_MANIFEST))"

execute: build
	cd "$(STAGE_BENCH)" && timeout --signal=TERM --kill-after=5s "$(EXECUTE_TIMEOUT_S)s" ./bin/rank1_dot_host --mode execute --response "$(EXECUTE_RESPONSE)" --stage-manifest "../../simplepim_stage_manifest.json"

clean:
	rm -rf build/simplepim_rank1_dot_m4_2
