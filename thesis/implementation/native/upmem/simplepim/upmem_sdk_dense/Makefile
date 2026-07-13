MAX_DIM ?= 16
L2_MAX_DIM ?= 512
L2_TILE_MAX_DIM ?= 64
NR_TASKLETS ?= 1
UPMEM_DENSE_HARDWARE_MVP ?= 0
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc

# The physical MVP is deliberately single-tasklet. Keep the configurable
# tasklet count for simulator/non-MVP builds.
DPU_NR_TASKLETS := $(if $(filter 1,$(UPMEM_DENSE_HARDWARE_MVP)),1,$(NR_TASKLETS))
DENSE_DEFINES := -DUPMEM_DENSE_MAX_DIM=$(MAX_DIM) -DUPMEM_DENSE_L2_MAX_DIM=$(L2_MAX_DIM) -DUPMEM_DENSE_L2_TILE_MAX_DIM=$(L2_TILE_MAX_DIM) -DNR_TASKLETS=$(DPU_NR_TASKLETS) -DUPMEM_DENSE_HARDWARE_MVP=$(UPMEM_DENSE_HARDWARE_MVP)

.PHONY: all clean

all: bin/host bin/dpu_dense

bin:
	mkdir -p bin

bin/dpu_dense: dpu.c common.h | bin
	$(DPU_CC) -O2 $(DENSE_DEFINES) -o $@ dpu.c

bin/host: host.c common.h | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra $(DENSE_DEFINES) -o $@ host.c `dpu-pkg-config --cflags --libs dpu`

clean:
	rm -rf bin
