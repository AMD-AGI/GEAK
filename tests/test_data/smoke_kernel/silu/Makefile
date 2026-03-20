# Makefile

# Compiler
HIPCC = hipcc

# Source and target
SRC = silu.hip
TARGET = applications_silu

# Compiler flags
CFLAGS = -O3 -ffast-math

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(TARGET)


