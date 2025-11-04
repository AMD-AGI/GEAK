# Makefile

# Compiler
HIPCC = hipcc

# Source and target
SRC = main.hip
TARGET = applications_point_to_voxelidx

# Compiler flags
CFLAGS = -O3

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(TARGET)

