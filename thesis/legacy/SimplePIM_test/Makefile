.DEFAULT_GOAL := all

# Include the central logic
include ../common.mk

all: bin/host_app

bin/host_app: host.c prep_spim
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ host.c $(SPIM_PATCHED_SRCS) $(LDFLAGS)

clean:
	rm -rf bin