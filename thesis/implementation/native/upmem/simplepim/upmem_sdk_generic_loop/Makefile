MAX_RANK ?= 7
MAX_ELEMS ?= 4096
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc

.PHONY: all clean

all: bin/host bin/dpu_generic

bin:
	mkdir -p bin

bin/dpu_generic: dpu.c common.h | bin
	$(DPU_CC) -O2 -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -o $@ dpu.c

bin/host: host.c common.h | bin
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -DUPMEM_GENERIC_MAX_RANK=$(MAX_RANK) -DUPMEM_GENERIC_MAX_ELEMS=$(MAX_ELEMS) -o $@ host.c `dpu-pkg-config --cflags --libs dpu`

clean:
	rm -rf bin
