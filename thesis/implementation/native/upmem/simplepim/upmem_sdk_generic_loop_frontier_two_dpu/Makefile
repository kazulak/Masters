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
SIMPLEPIM_STAGE ?= build/simplepim_frontier_two_dpu/staged/SimplePIM
SIMPLEPIM_STAGE_MARKER := $(SIMPLEPIM_STAGE)/management_profile_manifest.json
SIMPLEPIM_MANAGEMENT_C := $(SIMPLEPIM_STAGE)/lib/management/Management.c
SIMPLEPIM_MANAGEMENT_H := $(SIMPLEPIM_STAGE)/lib/management/Management.h
SIMPLEPIM_SMALLTABLE_C := $(SIMPLEPIM_STAGE)/lib/management/SmallTableInit.c

ifneq ($(NR_TASKLETS),1)
$(error frontier two-DPU profile requires NR_TASKLETS=1)
endif

ifneq ($(UPMEM_GENERIC_HARDWARE_MVP),1)
$(error frontier two-DPU profile requires UPMEM_GENERIC_HARDWARE_MVP=1)
endif

.PHONY: all simplepim-all clean simplepim-stage

all: bin/host_frontier_two_dpu bin/dpu_frontier_two_dpu

simplepim-all: simplepim-stage bin/host_frontier_two_dpu_simplepim_management bin/dpu_frontier_two_dpu_simplepim_management bin/dpu_simplepim_management_init

bin:
	mkdir -p bin

bin/dpu_frontier_two_dpu: ../upmem_sdk_generic_loop_resident/dpu.c ../upmem_sdk_generic_loop_resident/common.h | bin
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -o $@ ../upmem_sdk_generic_loop_resident/dpu.c

bin/dpu_frontier_two_dpu_simplepim_management: ../upmem_sdk_generic_loop_resident/dpu.c ../upmem_sdk_generic_loop_resident/common.h | bin
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -o $@ ../upmem_sdk_generic_loop_resident/dpu.c

bin/host_frontier_two_dpu: host.c common.h ../upmem_sdk_generic_loop_resident/session_protocol.c ../upmem_sdk_generic_loop_resident/session_protocol.h ../upmem_sdk_generic_loop_resident/common.h | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -o $@ host.c ../upmem_sdk_generic_loop_resident/session_protocol.c $$(dpu-pkg-config --cflags --libs dpu)

bin/host_frontier_two_dpu_simplepim_management: host.c common.h simplepim_provider.c simplepim_provider.h simplepim_management_profile.patch ../upmem_sdk_generic_loop_resident/session_protocol.c ../upmem_sdk_generic_loop_resident/session_protocol.h ../upmem_sdk_generic_loop_resident/common.h $(SIMPLEPIM_STAGE_MARKER) | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -DFRONTIER_PROVIDER_SIMPLEPIM_MANAGEMENT=1 -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -I$(SIMPLEPIM_STAGE)/lib/management -o $@ host.c simplepim_provider.c $(SIMPLEPIM_MANAGEMENT_C) $(SIMPLEPIM_SMALLTABLE_C) ../upmem_sdk_generic_loop_resident/session_protocol.c $$(dpu-pkg-config --cflags --libs dpu) -ldl

bin/dpu_simplepim_management_init: $(SIMPLEPIM_ROOT)/lib/management/SmallTableInit_dpu.c | bin
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -o $@ $<

$(SIMPLEPIM_STAGE_MARKER): $(SIMPLEPIM_ROOT)/lib/management/Management.c $(SIMPLEPIM_ROOT)/lib/management/Management.h simplepim_management_profile.patch
	set -eu; \
	actual_commit=$$(git -C "$(SIMPLEPIM_ROOT)" rev-parse --verify HEAD 2>/dev/null) || { echo "SimplePIM source is not a Git checkout" >&2; exit 1; }; \
	test "$$actual_commit" = "$(SIMPLEPIM_SOURCE_COMMIT)" || { echo "SimplePIM commit mismatch: expected $(SIMPLEPIM_SOURCE_COMMIT), got $$actual_commit" >&2; exit 1; }; \
	dirty=$$(git -C "$(SIMPLEPIM_ROOT)" status --porcelain --untracked-files=all); \
	test -z "$$dirty" || { echo "SimplePIM source worktree is dirty; refusing to stage it" >&2; printf '%s\n' "$$dirty" >&2; exit 1; }; \
	patch_sha256=$$(sha256sum simplepim_management_profile.patch | cut -d ' ' -f1); \
	rm -rf "$(SIMPLEPIM_STAGE)"; \
	mkdir -p "$(SIMPLEPIM_STAGE)/lib"; \
	cp -R "$(SIMPLEPIM_ROOT)/lib/." "$(SIMPLEPIM_STAGE)/lib/"; \
	(cd "$(SIMPLEPIM_STAGE)" && GIT_CEILING_DIRECTORIES=$$(realpath ..) git apply --recount --no-index --verbose "$(CURDIR)/simplepim_management_profile.patch"); \
	stage_root=$$(cd "$(SIMPLEPIM_STAGE)" && pwd); \
	tree_sha256=$$(find "$$stage_root" -type f ! -name management_profile_manifest.json -print | LC_ALL=C sort | while IFS= read -r file; do \
		rel="$${file#$$stage_root/}"; \
		printf '%s  %s\n' "$$(sha256sum "$$file" | cut -d ' ' -f1)" "$$rel"; \
	done | sha256sum | cut -d ' ' -f1); \
	printf '{"schema_version":"simplepim_management_stage_v1","source_commit":"%s","expected_source_commit":"%s","source_worktree_dirty":false,"staged_source_tree_sha256":"%s","patch_sha256":"%s","patch_applied":true}\n' "$$actual_commit" "$(SIMPLEPIM_SOURCE_COMMIT)" "$$tree_sha256" "$$patch_sha256" > "$@"

simplepim-stage:
	$(MAKE) -B $(SIMPLEPIM_STAGE_MARKER)

clean:
	rm -rf bin build/simplepim_frontier_two_dpu
