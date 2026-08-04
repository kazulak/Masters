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

ifneq ($(NR_TASKLETS),1)
$(error frontier two-DPU profile requires NR_TASKLETS=1)
endif

ifneq ($(UPMEM_GENERIC_HARDWARE_MVP),1)
$(error frontier two-DPU profile requires UPMEM_GENERIC_HARDWARE_MVP=1)
endif

.PHONY: all clean

all: bin/host_frontier_two_dpu bin/dpu_frontier_two_dpu

bin:
	mkdir -p bin

bin/dpu_frontier_two_dpu: ../upmem_sdk_generic_loop_resident/dpu.c ../upmem_sdk_generic_loop_resident/common.h | bin
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -o $@ ../upmem_sdk_generic_loop_resident/dpu.c

bin/host_frontier_two_dpu: host.c common.h ../upmem_sdk_generic_loop_resident/session_protocol.c ../upmem_sdk_generic_loop_resident/session_protocol.h ../upmem_sdk_generic_loop_resident/common.h | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -DNR_TASKLETS=$(NR_TASKLETS) -DUPMEM_GENERIC_HARDWARE_MVP=$(UPMEM_GENERIC_HARDWARE_MVP) -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -DRESIDENT_MAX_LOGICAL_TASKS=$(RESIDENT_MAX_LOGICAL_TASKS) -DRESIDENT_MAX_COMPONENT_OPS=$(RESIDENT_MAX_COMPONENT_OPS) -DRESIDENT_MAX_SLOT_DESCRIPTORS=$(RESIDENT_MAX_SLOT_DESCRIPTORS) -DRESIDENT_MRAM_POOL_BYTES=$(RESIDENT_MRAM_POOL_BYTES) -DRESIDENT_OUTPUT_TILE_ELEMS=$(RESIDENT_OUTPUT_TILE_ELEMS) -o $@ host.c ../upmem_sdk_generic_loop_resident/session_protocol.c $$(dpu-pkg-config --cflags --libs dpu)

clean:
	rm -rf bin
