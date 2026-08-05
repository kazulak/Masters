HOST_CC := gcc
DPU_CC ?= dpu-upmem-dpurte-clang

HOST_CFLAGS = --std=c99 -O2 -Wall -Wextra -DNR_TASKLETS=12 -DQUALIFICATION_ELEMENTS=256
DPU_CFLAGS = -O2 -DNR_TASKLETS=12
LIB_HOST = ../../lib/communication/CommHelper.c \
           ../../lib/communication/CommOps.c \
           ../../lib/management/Management.c \
           ../../lib/processing/map/Map.c \
           ../../lib/processing/zip/Zip.c

.PHONY: all clean verify-patch

all: verify-patch bin/host bin/dpu_init_binary bin/dpu_zip bin/dpu_map_va_funcs

verify-patch:
	@test "$$(grep -c 'copy_block_size-unroll_block_size' ../../lib/processing/map/MapProcessing.h)" -eq 2
	@test "$$(grep -c 'copy_block_size-unroll_block_rest' ../../lib/processing/map/MapProcessing.h)" -eq 0

bin:
	mkdir -p bin

bin/host: host.c Param.h small_table_init.c $(LIB_HOST) | bin
	$(HOST_CC) $(HOST_CFLAGS) -fopenmp -o $@ host.c small_table_init.c $(LIB_HOST) `dpu-pkg-config --cflags --libs dpu` -lm -ldl

bin/dpu_init_binary: ../../lib/management/SmallTableInit_dpu.c | bin
	$(DPU_CC) $(DPU_CFLAGS) -o $@ $<

bin/dpu_zip: ../../lib/processing/zip/zip_dpu.c ../../lib/processing/zip/ZipProcessing.c ../../lib/processing/ProcessingHelper.c | bin
	$(DPU_CC) $(DPU_CFLAGS) -o $@ $^

bin/dpu_map_va_funcs: ../../lib/processing/map/map_dpu.c ../../lib/processing/ProcessingHelper.c va_funcs/map.h | bin
	$(DPU_CC) $(DPU_CFLAGS) -I. -D__mapfunc_filename__=\"va_funcs/map.h\" -o $@ ../../lib/processing/map/map_dpu.c ../../lib/processing/ProcessingHelper.c

clean:
	rm -rf bin
