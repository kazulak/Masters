.DEFAULT_GOAL := all
include ../common.mk

TARGET = bin/hello_pim

all: $(TARGET)

$(TARGET): host.c prep_spim
	@mkdir -p bin/dpu_map_kernels
	$(CC) $(CFLAGS) -o $@ host.c $(SPIM_PATCHED_SRCS) $(LDFLAGS)

clean:
	rm -rf bin
