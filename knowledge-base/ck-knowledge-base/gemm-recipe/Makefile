# GEAK Test Harness - GEMM+Bias Kernel .so build rules
# Standalone build (independent of CK's CMake).
#
# Preferred: use compile.py which auto-detects GPU arch from PyTorch.
# Direct usage: make ARCH=gfx942 [target]

HIPCC      ?= /opt/rocm/bin/hipcc
CK_INCLUDE ?= /opt/rocm/include

ifndef ARCH
  $(error ARCH not set. Use compile.py for auto-detection, or: make ARCH=gfx942)
endif

HIPCC_FLAGS = -shared -fPIC -O3 \
              --offload-arch=$(ARCH) \
              -I$(CK_INCLUDE) \
              -std=c++17 \
              -MMD -MP

.PHONY: all baseline optimized clean

all: baseline optimized

baseline: libbaseline.so
optimized: liboptimized.so

libbaseline.so: baseline.cpp
	@echo "Building baseline for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -MF $(basename $@).d -o $@ $<

liboptimized.so: optimized.cpp
	@echo "Building optimized for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -MF $(basename $@).d -o $@ $<

clean:
	rm -f libbaseline.so liboptimized.so libbaseline.d liboptimized.d

-include libbaseline.d liboptimized.d
