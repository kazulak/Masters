# --- GLOBAL PATHS ---
# Resolves the absolute path to the thesis root
THESIS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
EXTERN_DIR  := $(THESIS_ROOT)/extern
SPIM_DIR    := $(EXTERN_DIR)/SimplePIM

# Centralized build directory
GLOBAL_PATCH_DIR := $(THESIS_ROOT)/build/patched_spim

CC = gcc

# GLOBAL FLAGS: 
# 1. -I. (look in current project dir)
# 2. -I$(GLOBAL_PATCH_DIR) (look inside the patched framework)
CFLAGS = --std=c99 -O3 -fopenmp -I. -I$(GLOBAL_PATCH_DIR)
LDFLAGS = -lm `dpu-pkg-config --cflags --libs dpu`

# SimplePIM Source Files
SPIM_FILES = processing/ProcessingHelperHost.c \
             communication/CommHelper.c \
             communication/CommOps.c \
             management/SmallTableInit.c \
             management/Management.c \
             processing/map/Map.c \
             processing/zip/Zip.c

SPIM_PATCHED_SRCS = $(addprefix $(GLOBAL_PATCH_DIR)/, $(SPIM_FILES))

# The Global Patching Target
prep_spim:
	@if [ ! -d "$(GLOBAL_PATCH_DIR)" ]; then \
		echo "Global setup: Patching SimplePIM framework..."; \
		mkdir -p $(GLOBAL_PATCH_DIR); \
		cp -r $(SPIM_DIR)/lib/* $(GLOBAL_PATCH_DIR)/; \
		find $(GLOBAL_PATCH_DIR) -type f -name "*.c" -exec sed -i 's|../../lib|$(SPIM_DIR)/lib|g' {} +; \
	fi