# SCIENTIFIC OBJECTIVE: Guarantee SOTA Hardware Utilization
#
# TO DO:
# 1. Define Compiler: gcc or clang.
# 2. Set Flags: -O3 (Max opt), -march=native (Enable CPU-specific AVX2/AVX-512), 
#    -fopenmp (Enable multi-threading).
# 3. Target 'quest_runner': Compile universal_runner.c and all files in src/circuits/*.c.
# 4. Link: -lm (Math) and the QuEST library objects.

# SCIENTIFIC OBJECTIVE: Guarantee SOTA Hardware Utilization

CC = gcc
# -O3: Max opt | -march=native: AVX2/AVX-512 | -fopenmp: Multithreading
CFLAGS = -O3 -march=native -fopenmp -Wall -I./external/QuEST -I./external/QuEST/quest/include -I./external/QuEST/build -I./external/QuEST/build/quest/include
# Link against the compiled QuEST library and standard math library
# UPDATED: Added rpath so the executable knows where to find the .so file at runtime
LDFLAGS = -L./external/QuEST/build -L./external/QuEST/build/quest -Wl,-rpath=$(shell pwd)/external/QuEST/build -Wl,-rpath=$(shell pwd)/external/QuEST/build/quest -lQuEST -lm -fopenmp -Wl,-z,muldefs
SRC_DIR = src
CIRC_DIR = $(SRC_DIR)/circuits
BIN_DIR = bin

# Gather all C files
SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(CIRC_DIR)/*.c)
OBJS = $(SRCS:.c=.o)

TARGET = $(BIN_DIR)/quest_runner

all: prep $(TARGET)

prep:
	mkdir -p $(BIN_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(CIRC_DIR)/*.o
	rm -rf $(BIN_DIR)