# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

profiler_prompt = """Analyze this AMD GPU profiler output (rocprof-compute) to identify performance optimization opportunities for kernel latency reduction.

ANALYSIS GUIDELINES:
1. Focus on high-signal sections with actual data:
   - Section 2: System Speed-of-Light (focus on "Pct of Peak" columns)
   - Section 7.2: Wavefront Runtime Stats (Dependency Wait, Issue Wait, Active Cycles)
   - Section 11: Compute Pipeline (utilization and throughput)
   - Cache sections (13-16): Hit rates and bandwidth utilization
   
2. Ignore metrics that show: 0.0, nan, or empty values - they're not active in this kernel

3. Key diagnostic ratios to calculate:
   - Kernel efficiency: Active Cycles / Total Wave Cycles
   - Memory vs Compute bound: Compare Dependency Wait Cycles vs Active Cycles
   - Cache effectiveness: Hit rates should be >60% to avoid being memory-bound

REQUIRED OUTPUT (aim for 350-450 tokens):

1. PRIMARY BOTTLENECK (2-3 sentences):
Identify the DOMINANT performance limiter using concrete metrics:
- State the specific resource/subsystem that is the bottleneck
- Provide quantitative evidence (e.g., "X% of peak", "Y cycles wasted", "Z% hit rate")
- Explain WHY this is occurring (root cause, not just the symptom)
- Severity: CRITICAL (<20% of peak), MODERATE (20-60%), or ACCEPTABLE (>60%)

Example: "Memory subsystem is CRITICALLY limiting performance. L2 cache hit rate is only 39%, causing 271-cycle fabric read latency. This results in Dependency Wait Cycles (7274) being 9x higher than Active Cycles (797), meaning the kernel spends 84% of time waiting for memory."

2. KEY METRICS (3-4 observations):
For each metric, provide the CAUSE → EFFECT → IMPACT chain:

- [Metric name]: [actual value] ([% of peak or comparison to ideal])
  ROOT CAUSE: [why this value is suboptimal]
  IMPACT: [how this limits performance - be specific]

Examples:
- VALU Utilization: 0.59 Giop/s (0.0% of 14,540 Giop/s peak)
  ROOT CAUSE: Kernel is dominated by memory operations, not compute
  IMPACT: Compute units sitting idle while waiting for data

- Wavefront Occupancy: Low (calculate from Active Cycles / Wave Cycles)
  ROOT CAUSE: High VGPR usage or small grid limiting concurrent wavefronts
  IMPACT: Cannot hide memory latency with parallel execution

3. OPTIMIZATION DIRECTIONS (2-3, ranked by expected impact):
Provide SPECIFIC, ACTIONABLE guidance (not vague suggestions):

HIGH IMPACT (>20% potential improvement):
[Specific optimization area] 
- Code-level approach: [e.g., "Increase thread block size to improve cache locality", "Use shared memory/LDS for data reuse pattern in lines X-Y"]
- Target metrics: [which metrics this will improve]
- Rationale: [why this addresses the bottleneck]

MEDIUM IMPACT (5-20% potential improvement):
[Second optimization if applicable]

AVOID vague suggestions like "improve memory access" - be specific about WHAT to change and WHY.

Hardware Context (extract from Section 1.1 if present):
- Note the GPU architecture (e.g., MI308X, gfx942) and key specs (CUs, wave_size)
- Use this to set expectations (e.g., wave_size=64 means VALU Active Threads should approach 64)

Profiler output:
{profiler_output}

Remember: Many metrics will be near-zero or empty. Focus ONLY on active metrics that reveal the bottleneck. Prioritize metrics that show actual resource consumption or performance gaps.
DO NOT use any tools call, output your THOUGHT only.
"""
