# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

from minisweagent.tools.bash_command import BashCommand
from minisweagent.tools.save_and_test import SaveAndTestTool
from minisweagent.tools.str_replace_editor import str_replace_editor
from minisweagent.tools.strategy_manager import StrategyManagerTool
from minisweagent.tools.submit import SubmitTool

current_dir = Path(__file__).resolve().parent
json_path = current_dir / "tools.json"
with open(json_path, encoding="utf-8") as f:
    _all_tools = json.load(f)

_TOOL_PROFILES: dict[str, set[str] | None] = {
    "full": None,  # None = register all tools (existing behavior)
    "swe": {
        "bash", "str_replace_editor", "save_and_test", "submit",
        "profile_kernel", "baseline_metrics", "strategy_manager",
    },
}


def get_tools_list(use_strategy_manager: bool = False) -> list:
    """Get filtered tools list based on settings.

    Args:
        use_strategy_manager: If True, include strategy_manager tool. If False, exclude it.
    Returns:
        List of tool definitions for the API.
    """
    excluded = set()
    if not use_strategy_manager:
        excluded.add("strategy_manager")
    return [t for t in _all_tools if t["name"] not in excluded]


# Backward compatibility
tools_list = _all_tools


class ToolRuntime:
    def __init__(
        self,
        use_strategy_manager: bool = False,
        strategy_file: str = ".optimization_strategies.md",
        on_strategy_change=None,
        patch_output_dir: str | None = None,
        tool_profile: str = "full",
    ):
        self._tool_profile = tool_profile
        self._mcp_bridges: list = []
        allowed = _TOOL_PROFILES.get(tool_profile)

        self._tool_table = {
            "bash": BashCommand(),
            "str_replace_editor": str_replace_editor(),
            "save_and_test": SaveAndTestTool(),
            "submit": SubmitTool(),
        }

        if allowed is not None:
            # Profile-restricted mode: only register explicitly allowed tools
            if "baseline_metrics" in allowed:
                from minisweagent.tools.baseline_metrics_tool import BaselineMetricsTool
                self._tool_table["baseline_metrics"] = BaselineMetricsTool()
            if use_strategy_manager and "strategy_manager" in allowed:
                self._tool_table["strategy_manager"] = StrategyManagerTool(
                    filepath=strategy_file, on_change_callback=on_strategy_change
                )
            if "profile_kernel" in allowed:
                self._register_profiler_mcp()
            self._sub_agent_tool = None
        else:
            # Full mode: register everything (existing behavior)
            if use_strategy_manager:
                self._tool_table["strategy_manager"] = StrategyManagerTool(
                    filepath=strategy_file, on_change_callback=on_strategy_change
                )

            from minisweagent.tools.baseline_metrics_tool import BaselineMetricsTool
            from minisweagent.tools.check_compat import CheckKernelCompatibilityTool
            from minisweagent.tools.resolve_kernel_url import ResolveKernelUrlTool

            self._tool_table["resolve_kernel_url"] = ResolveKernelUrlTool()
            self._tool_table["baseline_metrics"] = BaselineMetricsTool()
            self._tool_table["check_kernel_compatibility"] = CheckKernelCompatibilityTool()

            from minisweagent.tools.sub_agent_tool import SubAgentTool
            self._sub_agent_tool = SubAgentTool()
            self._tool_table["sub_agent"] = self._sub_agent_tool

            self._register_mcp_tools()

        self.use_strategy_manager = use_strategy_manager
        self._codebase_context: str | None = None

    def _register_profiler_mcp(self):
        """Register only the profiler-mcp tool."""
        try:
            from minisweagent.tools.mcp_bridge import MCPToolBridge
        except ImportError:
            return
        profiler = MCPToolBridge("profiler-mcp", timeout=600)
        self._mcp_bridges.append(profiler)
        self._tool_table["profile_kernel"] = profiler.tool("profile_kernel")

    def _register_mcp_tools(self):
        """Register all MCP server tools via MCPToolBridge.

        Each bridge wraps one MCP server process. The `.tool(name)` factory
        returns a sync callable that ToolRuntime can dispatch like a native tool.
        """
        try:
            from minisweagent.tools.mcp_bridge import MCPToolBridge
        except ImportError:
            return  # mcp_bridge not available (e.g., minimal install)

        profiler = MCPToolBridge("profiler-mcp", timeout=600)
        self._mcp_bridges.append(profiler)
        self._tool_table["profile_kernel"] = profiler.tool("profile_kernel")

        evolve = MCPToolBridge("kernel-evolve", timeout=300)
        self._mcp_bridges.append(evolve)
        self._tool_table["generate_optimization"] = evolve.tool("generate_optimization")
        self._tool_table["mutate_kernel"] = evolve.tool("mutate_kernel")
        self._tool_table["crossover_kernels"] = evolve.tool("crossover_kernels")
        self._tool_table["get_optimization_strategies"] = evolve.tool("get_optimization_strategies")

        ercs = MCPToolBridge("kernel-ercs", timeout=300)
        self._mcp_bridges.append(ercs)
        self._tool_table["evaluate_kernel_quality"] = ercs.tool("evaluate_kernel_quality")
        self._tool_table["reflect_on_kernel_result"] = ercs.tool("reflect_on_kernel_result")

        openevolve = MCPToolBridge("openevolve-mcp", timeout=7200)
        self._mcp_bridges.append(openevolve)
        self._tool_table["openevolve"] = openevolve.tool("optimize_kernel")

    def set_env(self, env: dict[str, str]) -> None:
        """Propagate environment overrides (e.g. HIP_VISIBLE_DEVICES) to tools."""
        env = dict(env)  # defensive copy to avoid shared-reference mutation
        bash = self._tool_table.get("bash")
        if bash is not None:
            bash._env_override = env
        for bridge in self._mcp_bridges:
            bridge.set_env(env)

    def set_cwd(self, cwd: str | None) -> None:
        """Propagate working directory to the bash tool so commands run in the correct worktree."""
        bash = self._tool_table.get("bash")
        if bash is not None:
            bash._cwd = cwd

    def set_codebase_context(self, context: str | None) -> None:
        """Store codebase context and propagate to SubAgentTool if present."""
        self._codebase_context = context
        if self._sub_agent_tool and context:
            self._sub_agent_tool._codebase_context = context

    def get_tools_schema(self) -> list[dict]:
        """Return JSON tool schemas for only the tools registered in _tool_table.

        This makes ToolRuntime the single source of truth: the agent can set
        model.tools = self.toolruntime.get_tools_schema() and the LLM will
        only see tools that are actually dispatchable.
        """
        return [t for t in _all_tools if t["name"] in self._tool_table]

    def get_tools_list(self) -> list:
        """Get the tools list for API based on current settings."""
        return get_tools_list(self.use_strategy_manager)

    def dispatch(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        tool_call format:
        {
            "name": "bash",
            "arguments": {...}
        }
        """
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if name not in self._tool_table:
            raise ValueError(f"Unknown tool: {name}")

        # Be robust to malformed tool calls from the LLM.
        # `bash` requires keyword-only `command`; missing it would crash the agent loop.
        if name == "bash" and "command" not in args:
            args = {**args, "command": ""}

        return self._tool_table[name](**args)


if __name__ == "__main__":
    tool_call = {
        "arguments": {
            "command": "str_replace",
            "path": "/mcp/rocPRIM_device_binary_search/rocprim/include/rocprim/device/try.hpp",
            "old_str": "#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_\n#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_",
            "new_str": "// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.\n//\n// Permission is hereby granted, free of charge, to any person obtaining a copy\n// of this software and ",
        },
        "name": "str_replace_editor",
    }
    tool_call = {
        "arguments": {
            "command": "str_replace",
            "path": "/mcp/rocPRIM_device_binary_search/rocprim/include/rocprim/device/device_binary_search.hpp",
            "old_str": '// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.\\n//\\n// Permission is hereby granted, free of charge, to any person obtaining a copy\\n// of this software and associated documentation files (the \\"Software\\"), to deal\\n// in the Software without restriction, including without limitation the rights\\n// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\\n// copies of the Software, and to permit persons to whom the Software is\\n// furnished to do so, subject to the following conditions:\\n//\\n// The above copyright notice and this permission notice shall be included in\\n// all copies or substantial portions of the Software.\\n//\\n// THE SOFTWARE IS PROVIDED \\"AS IS\\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\\n// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\\n// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\\n// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\\n// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\\n// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\\n// THE SOFTWARE.\\n\\n#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_\\n#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_\\n\\n#include <type_traits>\\n#include <iterator>\\n\\n#include \\"../config.hpp\\"\\n#include \\"../detail/various.hpp\\"\\n\\n#include \\"detail/device_binary_search.hpp\\"\\n#include \\"device_binary_search_config.hpp\\"\\n#include \\"device_transform.hpp\\"\\n\\n/// \\\\addtogroup devicemodule\\n/// @{\\n\\nBEGIN_ROCPRIM_NAMESPACE\\n\\nnamespace detail\\n{\\n\\ntemplate<\\n    class Config,\\n    class HaystackIterator,\\n    class NeedlesIterator,\\n    class OutputIterator,\\n    class SearchFunction,\\n    class CompareFunction\\n>\\ninline\\nhipError_t binary_search(void * temporary_storage,\\n                         size_t& storage_size,\\n                         HaystackIterator haystack,\\n                         NeedlesIterator needles,\\n                         OutputIterator output,\\n                         size_t haystack_size,\\n                         size_t needles_size,\\n                         SearchFunction search_op,\\n                         CompareFunction compare_op,\\n                         hipStream_t stream,\\n                         bool debug_synchronous)\\n{\\n    using value_type = typename std::iterator_traits<NeedlesIterator>::value_type;\\n\\n    if(temporary_storage == nullptr)\\n    {\\n        // Make sure user won\'t try to allocate 0 bytes memory, otherwise\\n        // user may again pass nullptr as temporary_storage\\n        storage_size = 4;\\n        return hipSuccess;\\n    }\\n\\n    return transform<Config>(\\n        needles, output,\\n        needles_size,\\n        [haystack, haystack_size, search_op, compare_op]\\n        ROCPRIM_DEVICE\\n        (const value_type& value)\\n        {\\n            return search_op(haystack, haystack_size, value, compare_op);\\n        },\\n        stream, debug_synchronous\\n    );\\n}',
            "new_str": '// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.\\n//\\n// Permission is hereby granted, free of charge, to any person obtaining a copy\\n// of this software and associated documentation files (the \\"Software\\"), to deal\\n// in the Software without restriction, including without limitation the rights\\n// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\\n// copies of the Software, and to permit persons to whom the Software is\\n// furnished to do so, subject to the following conditions:\\n//\\n// The above copyright notice and this permission notice shall be included in\\n// all copies or substantial portions of the Software.\\n//\\n// THE SOFTWARE IS PROVIDED \\"AS IS\\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\\n// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\\n// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\\n// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\\n// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\\n// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\\n// THE SOFTWARE.\\n\\n#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_\\n#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_\\n\\n#include <type_traits>\\n#include <iterator>\\n\\n#include \\"../config.hpp\\"\\n#include \\"../detail/various.hpp\\"\\n#include \\"../intrinsics.hpp\\"\\n\\n#include \\"detail/device_binary_search.hpp\\"\\n#include \\"device_binary_search_config.hpp\\"\\n#include \\"device_transform.hpp\\"\\n\\n/// \\\\addtogroup devicemodule\\n/// @{\\n\\nBEGIN_ROCPRIM_NAMESPACE\\n\\nnamespace detail\\n{\\n\\n// Optimized binary search kernel with LDS-based haystack caching\\n// This kernel caches strategic sample points from the haystack in LDS\\n// to reduce global memory accesses during the binary search\\ntemplate<\\n    unsigned int BlockSize,\\n    unsigned int ItemsPerThread,\\n    unsigned int CacheSize,\\n    class HaystackIterator,\\n    class NeedlesIterator,\\n    class OutputIterator,\\n    class SearchFunction,\\n    class CompareFunction\\n>\\nROCPRIM_DEVICE ROCPRIM_FORCE_INLINE\\nvoid binary_search_kernel_impl(\\n    HaystackIterator haystack,\\n    NeedlesIterator needles,\\n    OutputIterator output,\\n    size_t haystack_size,\\n    size_t needles_size,\\n    SearchFunction search_op,\\n    CompareFunction compare_op)\\n{\\n    using haystack_type = typename std::iterator_traits<HaystackIterator>::value_type;\\n    using needle_type = typename std::iterator_traits<NeedlesIterator>::value_type;\\n    using output_type = typename std::iterator_traits<OutputIterator>::value_type;\\n    \\n    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;\\n    \\n    // LDS cache for haystack samples and their indices\\n    __shared__ haystack_type s_cache[CacheSize];\\n    __shared__ size_t s_cache_indices[CacheSize];\\n    \\n    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();\\n    const unsigned int block_id = ::rocprim::detail::block_id<0>();\\n    const size_t block_offset = static_cast<size_t>(block_id) * items_per_block;\\n    \\n    // Collaboratively load haystack samples into shared memory cache\\n    // Each sample point is evenly distributed across the haystack\\n    unsigned int actual_cache_size = (haystack_size < CacheSize) ? static_cast<unsigned int>(haystack_size) : CacheSize;\\n    \\n    if(haystack_size > 0)\\n    {\\n        for(unsigned int i = flat_id; i < actual_cache_size; i += BlockSize)\\n        {\\n            size_t idx;\\n            if(actual_cache_size == 1)\\n            {\\n                idx = 0;\\n            }\\n            else\\n            {\\n                idx = static_cast<size_t>(i) * (haystack_size - 1) / (actual_cache_size - 1);\\n            }\\n            s_cache[i] = haystack[idx];\\n            s_cache_indices[i] = idx;\\n        }\\n    }\\n    \\n    __syncthreads();\\n    \\n    // Process needles - each thread handles ItemsPerThread needles\\n    ROCPRIM_UNROLL\\n    for(unsigned int item = 0; item < ItemsPerThread; ++item)\\n    {\\n        const size_t idx = block_offset + flat_id + static_cast<size_t>(item) * BlockSize;\\n        \\n        if(idx < needles_size)\\n        {\\n            const needle_type needle = needles[idx];\\n            \\n            // Phase 1: Binary search in cached samples to narrow down range\\n            size_t left = 0;\\n            size_t right = haystack_size;\\n            \\n            if(actual_cache_size > 1)\\n            {\\n                // Search in cache to find approximate range\\n                unsigned int cache_left = 0;\\n                unsigned int cache_right = actual_cache_size;\\n                \\n                while(cache_left < cache_right)\\n                {\\n                    unsigned int cache_mid = cache_left + (cache_right - cache_left) / 2;\\n                    if(compare_op(s_cache[cache_mid], needle))\\n                    {\\n                        cache_left = cache_mid + 1;\\n                    }\\n                    else\\n                    {\\n                        cache_right = cache_mid;\\n                    }\\n                }\\n                \\n                // Narrow the search range based on cache result\\n                if(cache_left > 0)\\n                {\\n                    left = s_cache_indices[cache_left - 1];\\n                }\\n                if(cache_left < actual_cache_size)\\n                {\\n                    right = s_cache_indices[cache_left] + 1;\\n                }\\n            }\\n            \\n            // Phase 2: Fine-grained binary search in the narrowed range\\n            output[idx] = search_op(haystack, haystack_size, left, right, needle, compare_op);\\n        }\\n    }\\n}\\n\\n// Search operations that work with pre-narrowed range\\nstruct lower_bound_range_search_op\\n{\\n    template<class HaystackIterator, class CompareOp, class Size, class T>\\n    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE\\n    Size operator()(HaystackIterator haystack, Size haystack_size, Size left, Size right, const T& value, CompareOp compare_op) const\\n    {\\n        while(left < right)\\n        {\\n            const Size mid = left + (right - left) / 2;\\n            if(compare_op(haystack[mid], value))\\n            {\\n                left = mid + 1;\\n            }\\n            else\\n            {\\n                right = mid;\\n            }\\n        }\\n        return left;\\n    }\\n};\\n\\nstruct upper_bound_range_search_op\\n{\\n    template<class HaystackIterator, class CompareOp, class Size, class T>\\n    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE\\n    Size operator()(HaystackIterator haystack, Size haystack_size, Size left, Size right, const T& value, CompareOp compare_op) const\\n    {\\n        while(left < right)\\n        {\\n            const Size mid = left + (right - left) / 2;\\n            if(compare_op(value, haystack[mid]))\\n            {\\n                right = mid;\\n            }\\n            else\\n            {\\n                left = mid + 1;\\n            }\\n        }\\n        return left;\\n    }\\n};\\n\\nstruct binary_search_range_op\\n{\\n    template<class HaystackIterator, class CompareOp, class Size, class T>\\n    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE\\n    bool operator()(HaystackIterator haystack, Size haystack_size, Size left, Size right, const T& value, CompareOp compare_op) const\\n    {\\n        while(left < right)\\n        {\\n            const Size mid = left + (right - left) / 2;\\n            if(compare_op(haystack[mid], value))\\n            {\\n                left = mid + 1;\\n            }\\n            else\\n            {\\n                right = mid;\\n            }\\n        }\\n        return left != haystack_size && !compare_op(value, haystack[left]);\\n    }\\n};\\n\\n// Kernel wrapper\\ntemplate<\\n    class Config,\\n    class HaystackIterator,\\n    class NeedlesIterator,\\n    class OutputIterator,\\n    class SearchFunction,\\n    class CompareFunction\\n>\\nROCPRIM_KERNEL\\nROCPRIM_LAUNCH_BOUNDS(device_params<Config>().kernel_config.block_size)\\nvoid binary_search_kernel(\\n    HaystackIterator haystack,\\n    NeedlesIterator needles,\\n    OutputIterator output,\\n    size_t haystack_size,\\n    size_t needles_size,\\n    SearchFunction search_op,\\n    CompareFunction compare_op)\\n{\\n    // Use 128 cache entries - good balance for most haystack sizes\\n    constexpr unsigned int cache_size = 128;\\n    binary_search_kernel_impl<\\n        device_params<Config>().kernel_config.block_size,\\n        device_params<Config>().kernel_config.items_per_thread,\\n        cache_size\\n    >(haystack, needles, output, haystack_size, needles_size, search_op, compare_op);\\n}\\n\\ntemplate<\\n    class Config,\\n    class HaystackIterator,\\n    class NeedlesIterator,\\n    class OutputIterator,\\n    class SearchFunction,\\n    class CompareFunction\\n>\\ninline\\nhipError_t binary_search(void * temporary_storage,\\n                         size_t& storage_size,\\n                         HaystackIterator haystack,\\n                         NeedlesIterator needles,\\n                         OutputIterator output,\\n                         size_t haystack_size,\\n                         size_t needles_size,\\n                         SearchFunction search_op,\\n                         CompareFunction compare_op,\\n                         hipStream_t stream,\\n                         bool debug_synchronous)\\n{\\n    using value_type = typename std::iterator_traits<NeedlesIterator>::value_type;\\n\\n    if(temporary_storage == nullptr)\\n    {\\n        // Make sure user won\'t try to allocate 0 bytes memory, otherwise\\n        // user may again pass nullptr as temporary_storage\\n        storage_size = 4;\\n        return hipSuccess;\\n    }\\n\\n    return transform<Config>(\\n        needles, output,\\n        needles_size,\\n        [haystack, haystack_size, search_op, compare_op]\\n        ROCPRIM_DEVICE\\n        (const value_type& value)\\n        {\\n            return search_op(haystack, haystack_size, value, compare_op);\\n        },\\n        stream, debug_synchronous\n    );\n}',
        },
        "name": "str_replace_editor",
    }
    tool_run = ToolRuntime()
    response = tool_run.dispatch(tool_call)
    print(response["output"])
