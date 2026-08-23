SIMPLEPIM_ROOT ?= ../../../external/SimplePIM
DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= gcc
NR_TASKLETS ?= 1

MAX_TASKLETS := 24
MAX_DPUS := 64

V4_GOAL := $(filter all v4 bin/host_upmem_execution_plan_v4_t% bin/dpu_gemm_tile_v4_t% bin/dpu_simplepim_management_init_t%,$(MAKECMDGOALS))
ifneq ($(V4_GOAL),)
ifneq ($(shell test "$(NR_TASKLETS)" -ge 1 2>/dev/null && test "$(NR_TASKLETS)" -le $(MAX_TASKLETS) 2>/dev/null && echo yes),yes)
$(error v4 requires NR_TASKLETS in [1,24])
endif
endif

.PHONY: all v4 clean

all: v4

v4: bin/host_upmem_execution_plan_v4_t$(NR_TASKLETS) \
    bin/dpu_gemm_tile_v4_t$(NR_TASKLETS) \
    bin/dpu_simplepim_management_init_t$(NR_TASKLETS)

bin:
	mkdir -p $@

bin/dpu_gemm_tile_v4_t%: dpu.c protocol.h | bin
	@test "$*" -ge 1 -a "$*" -le $(MAX_TASKLETS) || { echo "v4 DPU binary requires tasklets in [1,24]" >&2; exit 1; }
	$(DPU_CC) -O2 -DNR_TASKLETS=$* -o $@ dpu.c

bin/dpu_simplepim_management_init_t%: $(SIMPLEPIM_ROOT)/lib/management/SmallTableInit_dpu.c | bin
	@test "$*" -ge 1 -a "$*" -le $(MAX_TASKLETS) || { echo "v4 initialization binary requires tasklets in [1,24]" >&2; exit 1; }
	$(DPU_CC) -O2 -DNR_TASKLETS=$* -o $@ $<

bin/host_upmem_execution_plan_v4_t%: host.c dpu.c protocol.h request.c request.h plan.c plan.h simplepim_provider.c simplepim_provider.h | bin
	@test "$*" -ge 1 -a "$*" -le $(MAX_TASKLETS) || { echo "v4 host binary requires tasklets in [1,24]" >&2; exit 1; }
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -DNR_TASKLETS=$* -I$(SIMPLEPIM_ROOT)/lib/management -o $@ host.c request.c plan.c simplepim_provider.c $$(dpu-pkg-config --cflags --libs dpu) -ldl

clean:
	rm -rf bin
