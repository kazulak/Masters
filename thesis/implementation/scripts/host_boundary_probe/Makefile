CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
TARGET ?= upoenv_probe

.PHONY: all clean

all: $(TARGET)

$(TARGET): upoenv_probe.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
