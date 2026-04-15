# GEAK Test Harness - Conv2D Kernel .so build rules
# Standalone build (independent of CK's CMake).
#
# Preferred: use compile.py which auto-detects GPU arch from PyTorch.
# Direct usage: make ARCH=gfx942 [target]

THIS_MAKEFILE := $(abspath $(firstword $(MAKEFILE_LIST)))
IMPL_DIR      := $(dir $(THIS_MAKEFILE))
CK_INCLUDE    := $(abspath $(IMPL_DIR)../include)
HIPCC      ?= /opt/rocm/bin/hipcc

BASELINE_SO   := $(IMPL_DIR)libbaseline.so
OPTIMIZED_SO  := $(IMPL_DIR)liboptimized.so

ifndef ARCH
  $(error ARCH not set. Use compile.py for auto-detection, or: make ARCH=gfx950)
endif

HIPCC_FLAGS = -shared -fPIC -O3 \
              --offload-arch=$(ARCH) \
              -I$(CK_INCLUDE) \
              -std=c++17 \
              -MMD -MP

.PHONY: all baseline optimized clean

all: baseline optimized

baseline: $(BASELINE_SO)
optimized: $(OPTIMIZED_SO)

$(BASELINE_SO): $(IMPL_DIR)baseline.cpp
	@echo "Building baseline for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -MF $(IMPL_DIR)libbaseline.d -o $@ $<

$(OPTIMIZED_SO): $(IMPL_DIR)optimized.cpp
	@echo "Building optimized for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -MF $(IMPL_DIR)liboptimized.d -o $@ $<

clean:
	rm -f libbaseline.so liboptimized.so libbaseline.d liboptimized.d

# Pulled in after first compile; lists all headers the compiler actually opened.
-include $(IMPL_DIR)libbaseline.d $(IMPL_DIR)liboptimized.d
