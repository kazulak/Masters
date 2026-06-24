# SCIENTIFIC OBJECTIVE: Guarantee SOTA Hardware Utilization
#
# TO DO:
# 1. Define Compiler: gcc or clang.
# 2. Set Flags: -O3 (Max opt), -march=native (Enable CPU-specific AVX2/AVX-512), 
#    -fopenmp (Enable multi-threading).
# 3. Target 'quest_runner': Compile universal_runner.c and all files in src/circuits/*.c.
# 4. Link: -lm (Math) and the QuEST library objects.

# --- ARCHITECTURE PATHS ---
# Assuming this Makefile is in thesis/implementation/native/quest_cpu
# $(abspath ../..) resolves to the active implementation directory.
THESIS_ROOT := $(abspath ../../)
EXTERN_DIR  := $(THESIS_ROOT)/external
QUEST_DIR   := $(EXTERN_DIR)/QuEST
QUEST_BUILD := $(QUEST_DIR)/build

CC = gcc

# -O3: Max opt | -march=native: AVX2/AVX-512 | -fopenmp: Multithreading
CFLAGS = -O3 -march=native -fopenmp -Wall \
         -I$(QUEST_DIR) \
         -I$(QUEST_DIR)/quest/include \
         -I$(QUEST_BUILD) \
         -I$(QUEST_BUILD)/quest/include

# Link against the compiled QuEST library and standard math library
LDFLAGS = -L$(QUEST_BUILD) \
          -L$(QUEST_BUILD)/quest \
          -Wl,-rpath=$(QUEST_BUILD) \
          -Wl,-rpath=$(QUEST_BUILD)/quest \
          -lQuEST -lm -fopenmp -Wl,-z,muldefs

SRC_DIR = src
CIRC_DIR = $(SRC_DIR)/circuits
VERIFY_DIR = $(SRC_DIR)/verification
V_CIRC_DIR = $(VERIFY_DIR)/verify_circuits
BIN_DIR = bin
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Gather all C files
SRCS = $(wildcard $(SRC_DIR)/*.c) \
       $(wildcard $(CIRC_DIR)/*.c) \
       $(wildcard $(VERIFY_DIR)/*.c) \
       $(wildcard $(V_CIRC_DIR)/*.c)
OBJS = $(patsubst %.c,$(OBJ_DIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

TARGET = $(BIN_DIR)/quest_runner

all: prep $(TARGET)

prep:
	mkdir -p $(BIN_DIR)
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(CIRC_DIR)/*.o $(VERIFY_DIR)/*.o $(V_CIRC_DIR)/*.o
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)

-include $(DEPS)
