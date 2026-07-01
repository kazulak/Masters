# Developer-only QuEST GPU runner build wrapper.
#
# This builds generated artifacts outside external/QuEST so the submodule stays
# clean. Select GPU_BACKEND=hip for AMD/ROCm or GPU_BACKEND=cuda for NVIDIA.

THESIS_ROOT := $(abspath ../../)
QUEST_DIR := $(THESIS_ROOT)/external/QuEST
QUEST_CPU_RUNNER_DIR := $(THESIS_ROOT)/native/quest_cpu
QUEST_GPU_BUILD ?= $(THESIS_ROOT)/build/external/QuEST-$(GPU_BACKEND)
QUEST_BUILD_TYPE ?= Release
QUEST_BUILD_JOBS ?= 2
GPU_BACKEND ?= hip
CMAKE ?= cmake
HIPCONFIG ?= hipconfig
HIPCC ?= $(shell command -v hipcc 2>/dev/null)
ROCM_PATH_FROM_HIPCONFIG := $(shell command -v $(HIPCONFIG) >/dev/null 2>&1 && $(HIPCONFIG) --rocmpath 2>/dev/null)
ROCM_PATH_FALLBACK := $(firstword $(wildcard /opt/rocm /opt/rocm-*))
ROCM_PATH ?= $(or $(ROCM_PATH_FROM_HIPCONFIG),$(ROCM_PATH_FALLBACK))
HIP_CONFIG_CMAKE := $(firstword $(wildcard $(ROCM_PATH)/lib/cmake/hip/hip-config.cmake $(ROCM_PATH)/hip/lib/cmake/hip/hip-config.cmake))
HIP_DIR ?= $(patsubst %/,%,$(dir $(HIP_CONFIG_CMAKE)))
HIP_ARCHITECTURES ?= gfx1032
ROCM_CLANGXX ?= $(firstword $(wildcard $(ROCM_PATH)/lib/llvm/bin/clang++ $(ROCM_PATH)/llvm/bin/clang++))
CMAKE_HIP_COMPILER ?= $(ROCM_CLANGXX)

ifeq ($(GPU_BACKEND),hip)
QUEST_GPU_FLAGS := -DENABLE_HIP=ON \
                   -DCMAKE_PREFIX_PATH=$(ROCM_PATH) \
                   -DHIP_DIR=$(HIP_DIR) \
                   -DCMAKE_HIP_COMPILER=$(CMAKE_HIP_COMPILER) \
                   -DCMAKE_HIP_ARCHITECTURES=$(HIP_ARCHITECTURES)
CC := $(HIPCC)
else ifeq ($(GPU_BACKEND),cuda)
QUEST_GPU_FLAGS := -DENABLE_CUDA=ON
CC := gcc
else
$(error GPU_BACKEND must be hip or cuda)
endif

CFLAGS = -O3 -march=native -fopenmp -Wall \
         -I$(QUEST_DIR) \
         -I$(QUEST_DIR)/quest/include \
         -I$(QUEST_GPU_BUILD) \
         -I$(QUEST_GPU_BUILD)/quest/include \
         -I$(QUEST_CPU_RUNNER_DIR)/src

LDFLAGS = -L$(QUEST_GPU_BUILD) \
          -L$(QUEST_GPU_BUILD)/quest \
          -Wl,-rpath=$(QUEST_GPU_BUILD) \
          -Wl,-rpath=$(QUEST_GPU_BUILD)/quest \
          -lQuEST -lm -fopenmp -Wl,-z,muldefs

SRC_DIR = $(QUEST_CPU_RUNNER_DIR)/src
CIRC_DIR = $(SRC_DIR)/circuits
VERIFY_DIR = $(SRC_DIR)/verification
V_CIRC_DIR = $(VERIFY_DIR)/verify_circuits
BIN_DIR = bin
BUILD_DIR = build/$(GPU_BACKEND)
OBJ_DIR = $(BUILD_DIR)/obj

SRCS = $(wildcard $(SRC_DIR)/*.c) \
       $(wildcard $(CIRC_DIR)/*.c) \
       $(wildcard $(VERIFY_DIR)/*.c) \
       $(wildcard $(V_CIRC_DIR)/*.c)
OBJS = $(patsubst $(QUEST_CPU_RUNNER_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

TARGET = $(BIN_DIR)/quest_gpu_runner

.PHONY: all prep quest-lib clean clean-quest clean-all

all: quest-lib prep $(TARGET)

quest-lib:
ifeq ($(GPU_BACKEND),hip)
	test -n "$(ROCM_PATH)"
	test -n "$(HIPCC)"
	test -n "$(HIP_CONFIG_CMAKE)"
	test -n "$(CMAKE_HIP_COMPILER)"
endif
	$(CMAKE) -S $(QUEST_DIR) -B $(QUEST_GPU_BUILD) -DCMAKE_BUILD_TYPE=$(QUEST_BUILD_TYPE) $(QUEST_GPU_FLAGS)
	$(CMAKE) --build $(QUEST_GPU_BUILD) -j$(QUEST_BUILD_JOBS)

prep:
	mkdir -p $(BIN_DIR)
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(QUEST_CPU_RUNNER_DIR)/%.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf build
	rm -rf bin

clean-quest:
	rm -rf $(QUEST_GPU_BUILD)

clean-all: clean clean-quest

-include $(DEPS)
