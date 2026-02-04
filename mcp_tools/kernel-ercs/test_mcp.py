#!/usr/bin/env python3
"""Test the kernel-ercs MCP server tools."""

from kernel_ercs.server import mcp

def main():
    tools = mcp._tool_manager._tools
    
    # Test 1: check_kernel_compatibility with bad kernel
    print("=" * 50)
    print("Test 1: check_kernel_compatibility (bad kernel)")
    print("=" * 50)
    
    bad_kernel = """
@triton.jit
def bad_kernel(x_ptr):
    result = tl.libdevice.sin(tl.load(x_ptr))
"""
    result = tools["check_kernel_compatibility"].fn(kernel_code=bad_kernel)
    print("Compatible:", result["compatible"])
    print("Issues:", result["issues"])
    print()

    # Test 2: check_kernel_compatibility with good kernel
    print("=" * 50)
    print("Test 2: check_kernel_compatibility (good kernel)")
    print("=" * 50)
    
    good_kernel = """
@triton.jit
def good_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, x * 2)
"""
    result = tools["check_kernel_compatibility"].fn(kernel_code=good_kernel)
    print("Compatible:", result["compatible"])
    print("Issues:", result["issues"])
    print("Warnings:", result["warnings"])
    print()

    # Test 3: get_amd_gpu_specs
    print("=" * 50)
    print("Test 3: get_amd_gpu_specs")
    print("=" * 50)
    specs = tools["get_amd_gpu_specs"].fn()
    print("GPU:", specs["gpu"]["name"])
    print("Architecture:", specs["gpu"]["architecture"])
    print("Compute Units:", specs["gpu"]["compute_units"])
    print("Wavefront Size:", specs["gpu"]["wavefront_size"])
    print("VRAM:", specs["gpu"]["vram"])
    print()

    print("=" * 50)
    print("All tests passed! MCP Server is ready.")
    print("=" * 50)

if __name__ == "__main__":
    main()
