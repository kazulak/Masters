STAGE ?= .
HOST_CC ?= gcc
DPU_CC ?= dpu-upmem-dpurte-clang
DPU_PKG_CONFIG ?= dpu-pkg-config
DPU_CFLAGS := $(shell $(DPU_PKG_CONFIG) --cflags dpu)
DPU_LIBS := $(shell $(DPU_PKG_CONFIG) --libs dpu)

BIN := $(STAGE)/bin
INCLUDE := $(STAGE)/include
PIDCOMM_BIN := $(STAGE)/pidcomm_bin
HOST := $(BIN)/pidcomm_qualifier
DPU_USER := $(BIN)/dpu_user

.PHONY: all compatibility clean

all: $(HOST) $(DPU_USER)

# This target only compiles and links the staged host. It never runs it.
compatibility: $(HOST)
	@printf '%s\n' 'compatibility symbols checked: dpu_alloc_comm pidcomm_all_reduce init_hypercube_manager dpu_get_nr_dpus'

$(BIN):
	mkdir -p $@

$(HOST): $(STAGE)/host.c $(STAGE)/pidcomm_commlib.c $(INCLUDE)/pidcomm.h $(INCLUDE)/support.h $(INCLUDE)/pidcomm_binary_paths.h | $(BIN)
	$(HOST_CC) -std=c11 -Wall -Wextra -Werror=implicit-function-declaration -Werror=return-type -DINT32 -include $(INCLUDE)/pidcomm_binary_paths.h -I$(INCLUDE) $(DPU_CFLAGS) -Wl,--no-undefined -o $@ $(STAGE)/host.c $(STAGE)/pidcomm_commlib.c $(DPU_LIBS) -lpthread -ldl

$(DPU_USER): $(STAGE)/dpu_user.c | $(BIN)
	$(DPU_CC) -Wall -Wextra -O2 -DNR_TASKLETS=16 -I$(INCLUDE) -o $@ $<

clean:
	rm -rf $(BIN)
