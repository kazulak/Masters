MAX_DIM ?= 16
L2_MAX_DIM ?= 512
L2_TILE_MAX_DIM ?= 64
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc

.PHONY: all clean

all: bin/host bin/dpu_dense

bin:
	mkdir -p bin

bin/dpu_dense: dpu.c common.h | bin
	$(DPU_CC) -O2 -DUPMEM_DENSE_MAX_DIM=$(MAX_DIM) -DUPMEM_DENSE_L2_MAX_DIM=$(L2_MAX_DIM) -DUPMEM_DENSE_L2_TILE_MAX_DIM=$(L2_TILE_MAX_DIM) -o $@ dpu.c

bin/host: host.c common.h | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -DUPMEM_DENSE_MAX_DIM=$(MAX_DIM) -DUPMEM_DENSE_L2_MAX_DIM=$(L2_MAX_DIM) -DUPMEM_DENSE_L2_TILE_MAX_DIM=$(L2_TILE_MAX_DIM) -o $@ host.c `dpu-pkg-config --cflags --libs dpu`

clean:
	rm -rf bin
