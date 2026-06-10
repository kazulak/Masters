THESIS_ROOT := $(abspath ../../..)
QUEST_DIR   := $(THESIS_ROOT)/extern/QuEST
QUEST_BUILD := $(QUEST_DIR)/build

CC = gcc

CFLAGS = -O3 -march=native -fopenmp -Wall \
         -I$(QUEST_DIR) \
         -I$(QUEST_DIR)/quest/include \
         -I$(QUEST_BUILD) \
         -I$(QUEST_BUILD)/quest/include

LDFLAGS = -L$(QUEST_BUILD) \
          -L$(QUEST_BUILD)/quest \
          -Wl,-rpath=$(QUEST_BUILD) \
          -Wl,-rpath=$(QUEST_BUILD)/quest \
          -lQuEST -lm -fopenmp -Wl,-z,muldefs

BIN_DIR = bin
TARGET = $(BIN_DIR)/quest_exact_runner
SRCS = src/quest_exact_runner.c

all: prep $(TARGET)

prep:
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all prep clean

