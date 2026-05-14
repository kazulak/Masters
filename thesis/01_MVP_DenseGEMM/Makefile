.DEFAULT_GOAL := all
include ../common.mk

TARGET = mvp_host
DPU_TARGET = bin/dpu_gemm_int8
DPU_NR_TASKLETS = 8

all: $(DPU_TARGET) $(TARGET)

$(DPU_TARGET): kernels/gemm_int8/dpu.c kernels/gemm_int8/map.h Param.h
	@mkdir -p bin
	dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=$(DPU_NR_TASKLETS) -I. -o $@ kernels/gemm_int8/dpu.c

$(TARGET): src_c/main.c src_c/quantizer.c src_c/quantizer.h src_c/cjson/cJSON.c src_c/cjson/cJSON.h Param.h prep_spim
	@mkdir -p bin
	$(CC) $(CFLAGS) -O2 -o $@ src_c/main.c src_c/quantizer.c src_c/cjson/cJSON.c $(LDFLAGS)

clean:
	rm -rf bin $(TARGET)
