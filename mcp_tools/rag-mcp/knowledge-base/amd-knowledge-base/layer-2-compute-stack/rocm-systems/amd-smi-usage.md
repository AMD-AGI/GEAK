---
layer: "2"
category: "rocm-systems"
subcategory: "monitoring"
tags: ["amd-smi", "monitoring", "gpu-management", "performance", "cli", "python", "go", "c++"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-04
---

# AMD System Management Interface (AMD SMI)

The AMD System Management Interface (AMD SMI) is a unified library and toolset for managing and monitoring AMD GPUs, particularly in high-performance computing and AI/ML environments. It provides comprehensive GPU monitoring, configuration, and performance management capabilities.

**Repository**: [https://github.com/ROCm/amdsmi](https://github.com/ROCm/amdsmi)

**Documentation**: [https://rocm.docs.amd.com/projects/amdsmi/en/latest](https://rocm.docs.amd.com/projects/amdsmi/en/latest)

**License**: MIT

## Overview

AMD SMI is the successor to `rocm_smi_lib` and `esmi_ib_library`, providing a modern, unified interface for:

- **GPU Monitoring**: Temperature, power consumption, utilization, memory usage
- **Performance Management**: Clock speeds, power limits, performance levels
- **Device Information**: GPU models, drivers, capabilities, firmware versions
- **Process Tracking**: Identify which processes are using GPUs
- **Error Reporting**: Monitor GPU errors and events
- **Multi-GPU Support**: Manage multiple GPUs in a system

### Supported Platforms

- **Linux Bare Metal**: Full support for AMD GPUs
- **Linux VM (Guest)**: Virtualization support with AMD-SMI Virtualization
- **GPU Support**: AMD ROCm-supported GPUs (MI300X, MI250X, MI210, etc.)
- **CPU Support**: AMD EPYC™ CPUs via `esmi_ib_library` integration

## Key Features

### 1. Multi-Language Support

AMD SMI provides interfaces for multiple programming languages:

- **C++ Library**: High-performance native API
- **Python Library**: Easy-to-use Python bindings
- **Go Library**: Golang interface for system tools
- **CLI Tool**: `amd-smi` command-line interface

### 2. Comprehensive Monitoring

```bash
# Monitor GPU utilization
amd-smi metric --gpu --usage

# Monitor temperature and power
amd-smi metric --temperature --power

# Monitor memory usage
amd-smi metric --memory-usage

# Real-time monitoring (update every 1 second)
amd-smi metric --gpu --watch 1000
```

### 3. GPU Management

```bash
# Show device information
amd-smi static --asic --bus --vbios

# Set performance level
sudo amd-smi set --perf-level high

# Set power limit (300W example)
sudo amd-smi set --power-cap 300

# Reset GPU
sudo amd-smi reset --gpu 0
```

### 4. Process Monitoring

```bash
# Show processes using GPUs
amd-smi process --gpu-process

# Show detailed process information
amd-smi process --pid <process_id>
```

## Installation

### Prerequisites

- **AMD GPU Driver**: `amdgpu` driver must be loaded
- **ROCm**: ROCm 7.0+ installed
- **Python**: Python 3.6.8+ (64-bit) for Python interface
- **Go**: Go 1.20+ for Go interface

### From ROCm Installation

AMD SMI is included with ROCm installation:

```bash
# Verify installation
which amd-smi
# Output: /opt/rocm/bin/amd-smi

# Check version
amd-smi version

# Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/ROCm/amdsmi.git
cd amdsmi

# Install dependencies (Ubuntu/Debian)
sudo apt install -y \
  cmake \
  libdrm-dev \
  libgtest-dev

# Build
mkdir -p build
cd build
cmake ..
make -j $(nproc)

# Install (requires sudo)
sudo make install

# Build packages
make package  # Creates .deb and .rpm packages
```

### Python Package Installation

```bash
# Install Python interface
pip install amdsmi

# Or from source
cd amdsmi
pip install .
```

### Docker Container Support

For Docker containers, include these configuration options:

```bash
docker run \
  --cap-add=SYS_MODULE \
  -v /lib/modules:/lib/modules \
  --device=/dev/kfd \
  --device=/dev/dri \
  your-image:tag
```

**Required for**:
- Memory partitioning operations
- Kernel driver management
- Full hardware access

---

## CLI Usage (`amd-smi`)

### Device Information

```bash
# Show all GPU information (comprehensive)
amd-smi list

# Show static device information
amd-smi static

# Show specific static info
amd-smi static --asic        # ASIC information
amd-smi static --bus         # PCIe bus info
amd-smi static --vbios       # VBIOS version
amd-smi static --limit       # Hardware limits
amd-smi static --driver      # Driver information

# Show firmware information
amd-smi firmware

# Show topology (multi-GPU systems)
amd-smi topology
```

### Real-Time Monitoring

```bash
# Monitor all metrics
amd-smi monitor

# Monitor specific GPU (GPU ID 0)
amd-smi monitor --gpu 0

# Monitor with custom interval (milliseconds)
amd-smi monitor --watch 500

# Monitor specific metrics
amd-smi metric --gpu --usage          # GPU utilization
amd-smi metric --memory-usage         # Memory usage
amd-smi metric --temperature          # Temperature
amd-smi metric --power                # Power consumption
amd-smi metric --clock                # Clock frequencies
amd-smi metric --fan                  # Fan speed

# Combined monitoring
amd-smi metric --gpu --memory --power --temperature --watch 1000
```

### Performance Management

```bash
# Show current performance level
amd-smi metric --perf-level

# Set performance level
sudo amd-smi set --perf-level auto    # Automatic
sudo amd-smi set --perf-level low     # Low power
sudo amd-smi set --perf-level high    # High performance
sudo amd-smi set --perf-level manual  # Manual control

# Set clock frequencies (manual mode)
sudo amd-smi set --perf-level manual
sudo amd-smi set --sclk 5 --gpu 0     # Set GPU clock to level 5
sudo amd-smi set --mclk 3 --gpu 0     # Set memory clock to level 3

# Show available clock levels
amd-smi metric --clock --show-levels
```

### Power Management

```bash
# Show current power consumption
amd-smi metric --power

# Show power cap (limit)
amd-smi metric --power-cap

# Set power cap (GPU 0, 300W)
sudo amd-smi set --power-cap 300 --gpu 0

# Show power profile
amd-smi metric --power-profile

# Set power profile
sudo amd-smi set --power-profile compute
```

### Process Management

```bash
# Show all GPU processes
amd-smi process

# Show processes on specific GPU
amd-smi process --gpu 0

# Show detailed process info
amd-smi process --pid 12345

# JSON output for scripting
amd-smi process --json
```

### Event Monitoring

```bash
# Show GPU events
amd-smi event

# Monitor for specific events
amd-smi event --gpu 0
```

### GPU Reset

```bash
# Reset specific GPU
sudo amd-smi reset --gpu 0

# Reset all GPUs
sudo amd-smi reset --gpu all
```

### Output Formats

```bash
# Default table format
amd-smi list

# JSON format (for scripting)
amd-smi list --json

# CSV format (for data analysis)
amd-smi list --csv

# Save output to file
amd-smi list --json > gpu_info.json
amd-smi monitor --csv > monitoring.csv
```

---

## Python API Usage

### Installation

```python
# Install
pip install amdsmi

# Import
import amdsmi
```

### Basic Device Enumeration

```python
import amdsmi

# Initialize AMD SMI
try:
    amdsmi.amdsmi_init()
    print("AMD SMI initialized successfully")
except amdsmi.AmdSmiException as e:
    print(f"Initialization failed: {e}")
    exit(1)

# Get list of GPU devices
try:
    devices = amdsmi.amdsmi_get_processor_handles()
    print(f"Found {len(devices)} GPU(s)")
    
    for i, device in enumerate(devices):
        print(f"\nGPU {i}:")
        
        # Get device name
        info = amdsmi.amdsmi_get_gpu_asic_info(device)
        print(f"  Name: {info['market_name']}")
        print(f"  Device ID: {hex(info['device_id'])}")
        print(f"  Vendor ID: {hex(info['vendor_id'])}")
        
except amdsmi.AmdSmiException as e:
    print(f"Error: {e}")

finally:
    # Cleanup
    amdsmi.amdsmi_shut_down()
```

### GPU Monitoring

```python
import amdsmi
import time

# Initialize
amdsmi.amdsmi_init()
devices = amdsmi.amdsmi_get_processor_handles()

# Monitor GPU 0
device = devices[0]

try:
    # GPU utilization
    utilization = amdsmi.amdsmi_get_gpu_activity(device)
    print(f"GPU Utilization: {utilization['gfx_activity']}%")
    print(f"Memory Activity: {utilization['umc_activity']}%")
    
    # Temperature
    temp = amdsmi.amdsmi_get_temp_metric(
        device, 
        amdsmi.AmdSmiTemperatureType.EDGE,
        amdsmi.AmdSmiTemperatureMetric.CURRENT
    )
    print(f"Temperature: {temp / 1000}°C")
    
    # Power consumption
    power = amdsmi.amdsmi_get_power_info(device)
    print(f"Power: {power['current_socket_power'] / 1000000}W")
    print(f"Power Cap: {power['power_limit'] / 1000000}W")
    
    # Memory usage
    mem_info = amdsmi.amdsmi_get_gpu_memory_usage(
        device,
        amdsmi.AmdSmiMemoryType.VRAM
    )
    used_gb = mem_info['memory_usage'] / (1024**3)
    total_gb = mem_info['memory_total'] / (1024**3)
    print(f"Memory: {used_gb:.2f}GB / {total_gb:.2f}GB")
    
    # Clock frequencies
    clocks = amdsmi.amdsmi_get_clock_info(device, amdsmi.AmdSmiClkType.GFX)
    print(f"GPU Clock: {clocks['cur_clk']} MHz")
    
except amdsmi.AmdSmiException as e:
    print(f"Error: {e}")

finally:
    amdsmi.amdsmi_shut_down()
```

### Real-Time Monitoring Loop

```python
import amdsmi
import time

def monitor_gpu(device_index=0, interval=1.0):
    """Monitor GPU metrics in real-time."""
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    device = devices[device_index]
    
    print(f"Monitoring GPU {device_index}...")
    print("Press Ctrl+C to stop")
    print("\nTime      GPU%  Temp(°C)  Power(W)  Memory(GB)  GFX Clock(MHz)")
    print("-" * 70)
    
    try:
        while True:
            # Get metrics
            util = amdsmi.amdsmi_get_gpu_activity(device)
            temp = amdsmi.amdsmi_get_temp_metric(
                device,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT
            )
            power = amdsmi.amdsmi_get_power_info(device)
            mem = amdsmi.amdsmi_get_gpu_memory_usage(
                device,
                amdsmi.AmdSmiMemoryType.VRAM
            )
            clocks = amdsmi.amdsmi_get_clock_info(
                device,
                amdsmi.AmdSmiClkType.GFX
            )
            
            # Format and print
            timestamp = time.strftime("%H:%M:%S")
            gpu_util = util['gfx_activity']
            temp_c = temp / 1000
            power_w = power['current_socket_power'] / 1000000
            mem_gb = mem['memory_usage'] / (1024**3)
            clock_mhz = clocks['cur_clk']
            
            print(f"{timestamp}  {gpu_util:4d}  {temp_c:7.1f}  "
                  f"{power_w:8.1f}  {mem_gb:10.2f}  {clock_mhz:13d}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    finally:
        amdsmi.amdsmi_shut_down()

# Run monitoring
if __name__ == "__main__":
    monitor_gpu(device_index=0, interval=1.0)
```

### Performance Control

```python
import amdsmi

# Initialize
amdsmi.amdsmi_init()
devices = amdsmi.amdsmi_get_processor_handles()
device = devices[0]

try:
    # Set performance level
    amdsmi.amdsmi_set_gpu_perf_level(
        device,
        amdsmi.AmdSmiDevPerfLevel.HIGH
    )
    print("Performance level set to HIGH")
    
    # Set power cap (300W)
    power_cap = 300 * 1000000  # Convert to microwatts
    amdsmi.amdsmi_set_power_cap(device, 0, power_cap)
    print("Power cap set to 300W")
    
    # Get current settings
    power_info = amdsmi.amdsmi_get_power_info(device)
    print(f"Current power cap: {power_info['power_limit'] / 1000000}W")
    
except amdsmi.AmdSmiException as e:
    print(f"Error: {e}")
    print("Note: Some operations require sudo/root privileges")

finally:
    amdsmi.amdsmi_shut_down()
```

### Process Information

```python
import amdsmi

# Initialize
amdsmi.amdsmi_init()
devices = amdsmi.amdsmi_get_processor_handles()

try:
    for i, device in enumerate(devices):
        print(f"\nGPU {i} Processes:")
        
        # Get processes using this GPU
        processes = amdsmi.amdsmi_get_gpu_process_list(device)
        
        for process in processes:
            pid = process['process_id']
            mem_usage = process['memory_usage']['vram_mem'] / (1024**2)  # MB
            
            print(f"  PID: {pid}")
            print(f"    Memory: {mem_usage:.2f} MB")
            print(f"    Engine: {process['engine_usage']}")

except amdsmi.AmdSmiException as e:
    print(f"Error: {e}")

finally:
    amdsmi.amdsmi_shut_down()
```

---

## Go API Usage

### Installation

```bash
go get github.com/ROCm/amdsmi/goamdsmi
```

### Basic Usage

```go
package main

import (
    "fmt"
    "github.com/ROCm/amdsmi/goamdsmi"
)

func main() {
    // Initialize AMD SMI
    ret := goamdsmi.Init()
    if ret != goamdsmi.STATUS_SUCCESS {
        fmt.Printf("Failed to initialize: %v\n", ret)
        return
    }
    defer goamdsmi.Shutdown()

    // Get number of GPUs
    numDevices, ret := goamdsmi.GetNumDevices()
    if ret != goamdsmi.STATUS_SUCCESS {
        fmt.Printf("Failed to get device count: %v\n", ret)
        return
    }
    fmt.Printf("Found %d GPU(s)\n", numDevices)

    // Iterate through devices
    for i := uint32(0); i < numDevices; i++ {
        fmt.Printf("\nGPU %d:\n", i)
        
        // Get device name
        name, ret := goamdsmi.GetDeviceName(i)
        if ret == goamdsmi.STATUS_SUCCESS {
            fmt.Printf("  Name: %s\n", name)
        }

        // Get temperature
        temp, ret := goamdsmi.GetTemp(i, goamdsmi.TEMP_TYPE_EDGE)
        if ret == goamdsmi.STATUS_SUCCESS {
            fmt.Printf("  Temperature: %.1f°C\n", float64(temp)/1000)
        }

        // Get power
        power, ret := goamdsmi.GetPower(i)
        if ret == goamdsmi.STATUS_SUCCESS {
            fmt.Printf("  Power: %.1fW\n", float64(power)/1000000)
        }

        // Get memory usage
        memUsed, memTotal, ret := goamdsmi.GetMemoryUsage(i)
        if ret == goamdsmi.STATUS_SUCCESS {
            fmt.Printf("  Memory: %.2fGB / %.2fGB\n", 
                float64(memUsed)/(1024*1024*1024),
                float64(memTotal)/(1024*1024*1024))
        }
    }
}
```

---

## C++ API Usage

### Basic Device Query

```cpp
#include <amd_smi/amdsmi.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize AMD SMI
    amdsmi_status_t ret = amdsmi_init(0);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to initialize AMD SMI\n";
        return 1;
    }

    // Get socket count
    uint32_t socket_count = 0;
    ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to get socket count\n";
        amdsmi_shut_down();
        return 1;
    }

    std::vector<amdsmi_socket_handle> sockets(socket_count);
    ret = amdsmi_get_socket_handles(&socket_count, sockets.data());

    // Get devices per socket
    for (uint32_t i = 0; i < socket_count; i++) {
        uint32_t device_count = 0;
        ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
        
        std::vector<amdsmi_processor_handle> devices(device_count);
        ret = amdsmi_get_processor_handles(sockets[i], &device_count, devices.data());

        std::cout << "Socket " << i << " has " << device_count << " device(s)\n";

        // Query each device
        for (uint32_t j = 0; j < device_count; j++) {
            amdsmi_asic_info_t asic_info;
            ret = amdsmi_get_gpu_asic_info(devices[j], &asic_info);
            if (ret == AMDSMI_STATUS_SUCCESS) {
                std::cout << "  GPU " << j << ": " << asic_info.market_name << "\n";
            }

            // Get temperature
            int64_t temp;
            ret = amdsmi_get_temp_metric(devices[j], 
                                         AMDSMI_TEMPERATURE_TYPE_EDGE,
                                         AMDSMI_TEMP_CURRENT,
                                         &temp);
            if (ret == AMDSMI_STATUS_SUCCESS) {
                std::cout << "    Temperature: " << temp / 1000.0 << "°C\n";
            }

            // Get power
            amdsmi_power_info_t power_info;
            ret = amdsmi_get_power_info(devices[j], &power_info);
            if (ret == AMDSMI_STATUS_SUCCESS) {
                std::cout << "    Power: " << power_info.current_socket_power / 1000000.0 << "W\n";
            }
        }
    }

    // Cleanup
    amdsmi_shut_down();
    return 0;
}
```

### Compile and Link

```bash
# Compile with AMD SMI
g++ -o gpu_monitor gpu_monitor.cpp \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lamd_smi \
    -Wl,-rpath,/opt/rocm/lib

# Run
./gpu_monitor
```

---

## Advanced Use Cases

### 1. Multi-GPU Load Balancing

```python
import amdsmi
import numpy as np

def get_gpu_utilization():
    """Get utilization for all GPUs."""
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    
    utilizations = []
    for device in devices:
        util = amdsmi.amdsmi_get_gpu_activity(device)
        utilizations.append(util['gfx_activity'])
    
    amdsmi.amdsmi_shut_down()
    return utilizations

def select_least_utilized_gpu():
    """Select GPU with lowest utilization."""
    utils = get_gpu_utilization()
    return np.argmin(utils)

# Use in your application
optimal_gpu = select_least_utilized_gpu()
print(f"Using GPU {optimal_gpu} for next task")
```

### 2. Power Capping for Efficiency

```python
import amdsmi

def set_efficient_power_profile(max_power_watts=300):
    """Set power caps for all GPUs."""
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    
    for i, device in enumerate(devices):
        try:
            # Set power cap
            power_cap = max_power_watts * 1000000  # Convert to µW
            amdsmi.amdsmi_set_power_cap(device, 0, power_cap)
            
            # Verify
            power_info = amdsmi.amdsmi_get_power_info(device)
            actual_cap = power_info['power_limit'] / 1000000
            print(f"GPU {i}: Power cap set to {actual_cap}W")
            
        except amdsmi.AmdSmiException as e:
            print(f"GPU {i}: Failed to set power cap - {e}")
    
    amdsmi.amdsmi_shut_down()

# Apply efficient power profile
set_efficient_power_profile(max_power_watts=300)
```

### 3. Temperature Monitoring and Throttling

```python
import amdsmi
import time

def monitor_and_throttle(temp_threshold=80.0, check_interval=5.0):
    """Monitor temperature and throttle if needed."""
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    device = devices[0]
    
    print(f"Monitoring GPU temperature (threshold: {temp_threshold}°C)")
    
    try:
        while True:
            # Get current temperature
            temp = amdsmi.amdsmi_get_temp_metric(
                device,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT
            )
            temp_c = temp / 1000
            
            print(f"Temperature: {temp_c:.1f}°C", end="")
            
            # Check if over threshold
            if temp_c > temp_threshold:
                print(" - THROTTLING")
                # Reduce performance level
                amdsmi.amdsmi_set_gpu_perf_level(
                    device,
                    amdsmi.AmdSmiDevPerfLevel.LOW
                )
            else:
                print(" - OK")
                # Restore performance
                amdsmi.amdsmi_set_gpu_perf_level(
                    device,
                    amdsmi.AmdSmiDevPerfLevel.AUTO
                )
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\nStopped monitoring")
    finally:
        amdsmi.amdsmi_shut_down()

# Run thermal management
monitor_and_throttle(temp_threshold=80.0, check_interval=5.0)
```

### 4. Process Tracking for Resource Management

```python
import amdsmi
import time
from collections import defaultdict

def track_gpu_processes(duration=60, interval=5):
    """Track GPU process usage over time."""
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    
    process_history = defaultdict(list)
    
    print(f"Tracking GPU processes for {duration} seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            for i, device in enumerate(devices):
                processes = amdsmi.amdsmi_get_gpu_process_list(device)
                
                for proc in processes:
                    pid = proc['process_id']
                    mem_mb = proc['memory_usage']['vram_mem'] / (1024**2)
                    
                    process_history[f"GPU{i}_PID{pid}"].append({
                        'timestamp': time.time(),
                        'memory_mb': mem_mb
                    })
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nTracking stopped")
    
    finally:
        amdsmi.amdsmi_shut_down()
    
    # Print summary
    print("\nProcess Summary:")
    for key, history in process_history.items():
        avg_mem = sum(h['memory_mb'] for h in history) / len(history)
        max_mem = max(h['memory_mb'] for h in history)
        print(f"{key}: Avg={avg_mem:.2f}MB, Max={max_mem:.2f}MB")

# Run process tracking
track_gpu_processes(duration=60, interval=5)
```

### 5. Automated GPU Health Check

```python
import amdsmi

def health_check():
    """Perform comprehensive GPU health check."""
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    
    print("=== GPU Health Check ===\n")
    
    for i, device in enumerate(devices):
        print(f"GPU {i}:")
        issues = []
        
        try:
            # Check temperature
            temp = amdsmi.amdsmi_get_temp_metric(
                device,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT
            )
            temp_c = temp / 1000
            print(f"  Temperature: {temp_c:.1f}°C", end="")
            if temp_c > 85:
                issues.append("High temperature")
                print(" ⚠️")
            else:
                print(" ✓")
            
            # Check power
            power_info = amdsmi.amdsmi_get_power_info(device)
            power_w = power_info['current_socket_power'] / 1000000
            cap_w = power_info['power_limit'] / 1000000
            print(f"  Power: {power_w:.1f}W / {cap_w:.1f}W", end="")
            if power_w > cap_w * 0.95:
                issues.append("Near power limit")
                print(" ⚠️")
            else:
                print(" ✓")
            
            # Check memory
            mem_info = amdsmi.amdsmi_get_gpu_memory_usage(
                device,
                amdsmi.AmdSmiMemoryType.VRAM
            )
            used_gb = mem_info['memory_usage'] / (1024**3)
            total_gb = mem_info['memory_total'] / (1024**3)
            usage_pct = (used_gb / total_gb) * 100
            print(f"  Memory: {used_gb:.2f}GB / {total_gb:.2f}GB ({usage_pct:.1f}%)", end="")
            if usage_pct > 90:
                issues.append("High memory usage")
                print(" ⚠️")
            else:
                print(" ✓")
            
            # Check for errors (if available)
            try:
                error_count = amdsmi.amdsmi_get_gpu_ecc_error_count(device)
                print(f"  ECC Errors: {error_count}", end="")
                if error_count > 0:
                    issues.append(f"{error_count} ECC errors detected")
                    print(" ⚠️")
                else:
                    print(" ✓")
            except:
                print("  ECC Errors: N/A")
            
            # Summary
            if issues:
                print(f"  Status: WARNING - {', '.join(issues)}")
            else:
                print("  Status: HEALTHY ✓")
            
        except amdsmi.AmdSmiException as e:
            print(f"  Error checking GPU: {e}")
        
        print()
    
    amdsmi.amdsmi_shut_down()

# Run health check
health_check()
```

---

## Integration Examples

### PyTorch Multi-GPU Training

```python
import torch
import amdsmi

def select_gpus_for_training(num_gpus=2, min_memory_gb=40):
    """Select best GPUs for training based on availability."""
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_processor_handles()
    
    gpu_scores = []
    for i, device in enumerate(devices):
        # Check memory
        mem_info = amdsmi.amdsmi_get_gpu_memory_usage(
            device,
            amdsmi.AmdSmiMemoryType.VRAM
        )
        total_gb = mem_info['memory_total'] / (1024**3)
        used_gb = mem_info['memory_usage'] / (1024**3)
        free_gb = total_gb - used_gb
        
        # Check utilization
        util = amdsmi.amdsmi_get_gpu_activity(device)
        gpu_util = util['gfx_activity']
        
        # Score: prioritize free memory and low utilization
        if free_gb >= min_memory_gb:
            score = free_gb * (100 - gpu_util)
            gpu_scores.append((i, score, free_gb, gpu_util))
    
    amdsmi.amdsmi_shut_down()
    
    # Sort by score and select top N
    gpu_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [gpu[0] for gpu in gpu_scores[:num_gpus]]
    
    print("Selected GPUs for training:")
    for gpu_id in selected:
        info = [g for g in gpu_scores if g[0] == gpu_id][0]
        print(f"  GPU {info[0]}: {info[2]:.1f}GB free, {info[3]}% utilized")
    
    return selected

# Use in PyTorch
selected_gpus = select_gpus_for_training(num_gpus=2, min_memory_gb=40)
os.environ['HIP_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))

# Now train your model
model = YourModel().to('cuda')
model = torch.nn.DataParallel(model)
```

### Monitoring During Training

```python
import amdsmi
import threading
import time

class GPUMonitor:
    """Background GPU monitoring during training."""
    
    def __init__(self, log_file='gpu_monitor.csv'):
        self.log_file = log_file
        self.monitoring = False
        self.thread = None
        
    def start(self, interval=10):
        """Start monitoring."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.thread.start()
        
        # Write CSV header
        with open(self.log_file, 'w') as f:
            f.write("timestamp,gpu_id,utilization,temperature,power,memory_used_gb\n")
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self, interval):
        """Monitoring loop."""
        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()
        
        while self.monitoring:
            timestamp = time.time()
            
            for i, device in enumerate(devices):
                try:
                    # Collect metrics
                    util = amdsmi.amdsmi_get_gpu_activity(device)
                    temp = amdsmi.amdsmi_get_temp_metric(
                        device,
                        amdsmi.AmdSmiTemperatureType.EDGE,
                        amdsmi.AmdSmiTemperatureMetric.CURRENT
                    )
                    power = amdsmi.amdsmi_get_power_info(device)
                    mem = amdsmi.amdsmi_get_gpu_memory_usage(
                        device,
                        amdsmi.AmdSmiMemoryType.VRAM
                    )
                    
                    # Write to log
                    with open(self.log_file, 'a') as f:
                        f.write(f"{timestamp},{i},"
                               f"{util['gfx_activity']},"
                               f"{temp/1000:.1f},"
                               f"{power['current_socket_power']/1000000:.1f},"
                               f"{mem['memory_usage']/(1024**3):.2f}\n")
                
                except Exception as e:
                    print(f"Monitoring error: {e}")
            
            time.sleep(interval)
        
        amdsmi.amdsmi_shut_down()

# Use during training
monitor = GPUMonitor(log_file='training_gpu_monitor.csv')
monitor.start(interval=10)  # Log every 10 seconds

try:
    # Your training loop
    for epoch in range(num_epochs):
        train_one_epoch(model, dataloader, optimizer)
finally:
    monitor.stop()
    print("GPU monitoring log saved to training_gpu_monitor.csv")
```

---

## Troubleshooting

### Common Issues

**1. Permission Denied Errors**

```bash
# Error: Permission denied accessing /dev/kfd
# Solution: Add user to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Log out and back in, then verify
groups | grep render
```

**2. Library Not Found**

```bash
# Error: libamd_smi.so: cannot open shared object file
# Solution: Set library path
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Make permanent
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**3. Initialization Fails**

```bash
# Error: amdsmi_init() returns error
# Check if amdgpu driver is loaded
lsmod | grep amdgpu

# If not loaded, load it
sudo modprobe amdgpu

# Check device nodes
ls -l /dev/kfd /dev/dri/render*
```

**4. No GPUs Found**

```bash
# Verify ROCm sees GPUs
rocminfo | grep "Name:" | grep gfx

# Check with lspci
lspci | grep -i amd | grep -i vga

# Verify driver
dmesg | grep amdgpu
```

**5. Python Import Error**

```python
# Error: ModuleNotFoundError: No module named 'amdsmi'
# Solution: Install with pip
pip install amdsmi

# Or from source
cd /opt/rocm/share/amd_smi/py-interface
pip install .
```

### Docker Issues

```bash
# Add required capabilities
docker run \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_MODULE \
  -v /lib/modules:/lib/modules:ro \
  your-image

# Verify inside container
amd-smi list
```

---

## Performance Considerations

### Monitoring Overhead

AMD SMI monitoring has minimal performance overhead:

- **Device queries**: < 1ms per query
- **Continuous monitoring**: ~0.1% GPU utilization at 1Hz
- **Process enumeration**: < 10ms per GPU

**Best practices**:
- Use reasonable polling intervals (1-5 seconds for monitoring)
- Batch multiple queries together
- Use Python API for production monitoring (lower overhead than CLI)

### Efficient Monitoring

```python
import amdsmi

def efficient_batch_query(device):
    """Batch multiple queries efficiently."""
    # Initialize once
    amdsmi.amdsmi_init()
    
    # Collect all needed metrics in one pass
    metrics = {}
    
    try:
        # Get all metrics together
        metrics['utilization'] = amdsmi.amdsmi_get_gpu_activity(device)
        metrics['temperature'] = amdsmi.amdsmi_get_temp_metric(
            device, 
            amdsmi.AmdSmiTemperatureType.EDGE,
            amdsmi.AmdSmiTemperatureMetric.CURRENT
        )
        metrics['power'] = amdsmi.amdsmi_get_power_info(device)
        metrics['memory'] = amdsmi.amdsmi_get_gpu_memory_usage(
            device,
            amdsmi.AmdSmiMemoryType.VRAM
        )
        
    except amdsmi.AmdSmiException as e:
        print(f"Query failed: {e}")
        return None
    
    finally:
        amdsmi.amdsmi_shut_down()
    
    return metrics
```

---

## References

### Official Documentation

- **GitHub Repository**: [https://github.com/ROCm/amdsmi](https://github.com/ROCm/amdsmi)
- **API Documentation**: [https://rocm.docs.amd.com/projects/amdsmi/en/latest](https://rocm.docs.amd.com/projects/amdsmi/en/latest)
- **ROCm Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)

### Related Tools

- **ROCm SMI Library**: Legacy monitoring library (use AMD SMI for new projects)
- **ROCProfiler**: Performance profiling tool
- **rocminfo**: System information utility

### Community

- **Issues**: [https://github.com/ROCm/amdsmi/issues](https://github.com/ROCm/amdsmi/issues)
- **ROCm GitHub**: [https://github.com/ROCm](https://github.com/ROCm)

---

## Summary

AMD SMI provides:

✓ **Comprehensive Monitoring**: Temperature, power, utilization, memory, clocks  
✓ **Multi-Language Support**: C++, Python, Go, CLI interfaces  
✓ **Performance Management**: Set power limits, clock frequencies, performance levels  
✓ **Process Tracking**: Monitor GPU usage per process  
✓ **Production Ready**: Low overhead, battle-tested in HPC environments  
✓ **Modern Design**: Successor to rocm_smi_lib with improved API  

**Quick Start**:

```bash
# Install (included with ROCm)
which amd-smi

# Monitor GPUs
amd-smi monitor

# Python usage
pip install amdsmi
python -c "import amdsmi; amdsmi.amdsmi_init(); print('AMD SMI ready')"
```

**Best for**:
- Production GPU monitoring and alerting
- Resource management in multi-GPU systems
- Integration with training/inference pipelines
- Automated health checks and diagnostics
- Performance optimization workflows

---

*For detailed API reference and advanced features, see the [official AMD SMI documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest).*

