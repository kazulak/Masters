MAX_RANK ?= 16
MAX_ELEMS ?= 256
NR_TASKLETS ?= 1
UPMEM_GENERIC_HARDWARE_MVP ?= 1
RESIDENT_MAX_LOGICAL_TASKS ?= 32
RESIDENT_MAX_COMPONENT_OPS ?= 128
RESIDENT_MAX_SLOT_DESCRIPTORS ?= 128
RESIDENT_MRAM_POOL_BYTES ?= 524288
RESIDENT_OUTPUT_TILE_ELEMS ?= 256
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc
SIMPLEPIM_ROOT ?= ../../../../external/SimplePIM
SIMPLEPIM_SOURCE_COMMIT ?= 1d639c53532555f01e9f71d872e7712b166d6cba
SIMPLEPIM_SOURCE_DIR ?= ../upmem_sdk_generic_loop_frontier_two_dpu
SIMPLEPIM_STAGE ?= build/simplepim_execution_plan/staged/SimplePIM
SIMPLEPIM_STAGE_MARKER := $(SIMPLEPIM_STAGE)/management_profile_manifest.json
SIMPLEPIM_MANAGEMENT_C := $(SIMPLEPIM_STAGE)/lib/management/Management.c
SIMPLEPIM_MANAGEMENT_H := $(SIMPLEPIM_STAGE)/lib/management/Management.h
SIMPLEPIM_SMALLTABLE_C := $(SIMPLEPIM_STAGE)/lib/management/SmallTableInit.c

ifneq ($(filter 1 2,$(NR_TASKLETS)),1)
$(error execution-plan profile requires NR_TASKLETS=1)
endif
ifneq ($(UPMEM_GENERIC_HARDWARE_MVP),1)
$(error execution-plan profile requires UPMEM_GENERIC_HARDWARE_MVP=1)
endif

.PHONY: all simplepim-stage clean

all: bin/host_upmem_execution_plan bin/dpu_resident bin/dpu_simplepim_management_init

bin:
	mkdir -p bin

bin/dpu_resident: $(SIMPLEPIM_SOURCE_DIR)/../upmem_sdk_generic_loop_resident/dpu.c $(SIMPLEPIM_SOURCE_DIR)/../upmem_sdk_generic_loop_resident/common.h | bin
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -o $@ $<

bin/dpu_simplepim_management_init: $(SIMPLEPIM_ROOT)/lib/management/SmallTableInit_dpu.c | bin
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -o $@ $<

bin/host_upmem_execution_plan: host.c plan_request.c plan_request.h plan_schedule.c plan_schedule.h execution_plan_common.h execution_plan_provider.c execution_plan_provider.h $(SIMPLEPIM_SOURCE_DIR)/../upmem_sdk_generic_loop_resident/session_protocol.c $(SIMPLEPIM_SOURCE_DIR)/../upmem_sdk_generic_loop_resident/session_protocol.h $(SIMPLEPIM_SOURCE_DIR)/../upmem_sdk_generic_loop_resident/common.h $(SIMPLEPIM_STAGE_MARKER) | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -I$(SIMPLEPIM_STAGE)/lib/management -o $@ host.c plan_request.c plan_schedule.c execution_plan_provider.c $(SIMPLEPIM_SOURCE_DIR)/../upmem_sdk_generic_loop_resident/session_protocol.c $(SIMPLEPIM_MANAGEMENT_C) $(SIMPLEPIM_SMALLTABLE_C) $$(dpu-pkg-config --cflags --libs dpu) -ldl

$(SIMPLEPIM_STAGE_MARKER): $(SIMPLEPIM_ROOT)/lib/management/Management.c $(SIMPLEPIM_ROOT)/lib/management/Management.h $(SIMPLEPIM_SOURCE_DIR)/simplepim_management_profile.patch
	set -eu; \
	actual_commit=$$(git -C "$(SIMPLEPIM_ROOT)" rev-parse --verify HEAD 2>/dev/null) || { echo "SimplePIM source is not a Git checkout" >&2; exit 1; }; \
	test "$$actual_commit" = "$(SIMPLEPIM_SOURCE_COMMIT)" || { echo "SimplePIM commit mismatch: expected $(SIMPLEPIM_SOURCE_COMMIT), got $$actual_commit" >&2; exit 1; }; \
	dirty=$$(git -C "$(SIMPLEPIM_ROOT)" status --porcelain --untracked-files=all); \
	test -z "$$dirty" || { echo "SimplePIM source worktree is dirty; refusing to stage it" >&2; printf '%s\n' "$$dirty" >&2; exit 1; }; \
	patch_path="$(abspath $(SIMPLEPIM_SOURCE_DIR)/simplepim_management_profile.patch)"; \
	patch_sha256=$$(sha256sum "$$patch_path" | cut -d ' ' -f1); \
	rm -rf "$(SIMPLEPIM_STAGE)"; mkdir -p "$(SIMPLEPIM_STAGE)/lib"; cp -R "$(SIMPLEPIM_ROOT)/lib/." "$(SIMPLEPIM_STAGE)/lib/"; \
	(cd "$(SIMPLEPIM_STAGE)" && GIT_CEILING_DIRECTORIES=$$(realpath ..) git apply --recount --no-index --verbose "$$patch_path"); \
	stage_root=$$(cd "$(SIMPLEPIM_STAGE)" && pwd); \
	tree_sha256=$$(find "$$stage_root" -type f ! -name management_profile_manifest.json -print | LC_ALL=C sort | while IFS= read -r file; do rel="$${file#$$stage_root/}"; printf '%s  %s\n' "$$(sha256sum "$$file" | cut -d ' ' -f1)" "$$rel"; done | sha256sum | cut -d ' ' -f1); \
	printf '{"schema_version":"simplepim_management_stage_v1","source_commit":"%s","expected_source_commit":"%s","source_worktree_dirty":false,"staged_source_tree_sha256":"%s","patch_sha256":"%s","patch_applied":true}\n' "$$actual_commit" "$(SIMPLEPIM_SOURCE_COMMIT)" "$$tree_sha256" "$$patch_sha256" > "$@"

simplepim-stage:
	$(MAKE) -B $(SIMPLEPIM_STAGE_MARKER)

clean:
	rm -rf bin build/simplepim_execution_plan
