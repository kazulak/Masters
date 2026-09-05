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

bin/dpu_gemm_tile_v4_t%: dpu.c panel_compute.h protocol.h | bin
	@test "$*" -ge 1 -a "$*" -le $(MAX_TASKLETS) || { echo "v4 DPU binary requires tasklets in [1,24]" >&2; exit 1; }
	$(DPU_CC) -O2 -DNR_TASKLETS=$* -o $@ dpu.c

# Experimental launch target; not selected by the active v4 host.
bin/dpu_wave_v5_t%: dpu_wave.c panel_compute.h outer_compute.h wave_protocol.h protocol.h | bin
	@test "$*" -ge 1 -a "$*" -le $(MAX_TASKLETS) || { echo "wave DPU binary requires tasklets in [1,24]" >&2; exit 1; }
	$(DPU_CC) -O2 -DNR_TASKLETS=$* -o $@ dpu_wave.c

bin/dpu_simplepim_management_init_t%: management_init.c $(SIMPLEPIM_ROOT)/lib/management/SmallTableInit_dpu.c | bin
	@test "$*" -ge 1 -a "$*" -le $(MAX_TASKLETS) || { echo "v4 initialization binary requires tasklets in [1,24]" >&2; exit 1; }
	$(DPU_CC) -O2 -DNR_TASKLETS=$* -I$(SIMPLEPIM_ROOT)/lib/management -o $@ $<

bin/host_upmem_execution_plan_v4_t%: host.c dpu.c protocol.h request.c request.h operation_envelope.c operation_envelope.h wave_envelope.c wave_envelope.h wave_protocol.h plan.c plan.h simplepim_provider.c simplepim_provider.h | bin
	@test "$*" -ge 1 -a "$*" -le $(MAX_TASKLETS) || { echo "v4 host binary requires tasklets in [1,24]" >&2; exit 1; }
	$(HOST_CC) -std=c99 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -DNR_TASKLETS=$* -I$(SIMPLEPIM_ROOT)/lib/management -o $@ host.c request.c operation_envelope.c wave_envelope.c plan.c simplepim_provider.c $$(dpu-pkg-config --cflags --libs dpu) -ldl

clean:
	rm -rf bin
