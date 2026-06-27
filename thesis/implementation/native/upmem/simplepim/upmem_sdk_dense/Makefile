MAX_DIM ?= 16
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc

.PHONY: all clean

all: bin/host bin/dpu_dense

bin:
	mkdir -p bin

bin/dpu_dense: dpu.c common.h | bin
	$(DPU_CC) -O2 -DUPMEM_DENSE_MAX_DIM=$(MAX_DIM) -o $@ dpu.c

bin/host: host.c common.h | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -DUPMEM_DENSE_MAX_DIM=$(MAX_DIM) -o $@ host.c `dpu-pkg-config --cflags --libs dpu`

clean:
	rm -rf bin
