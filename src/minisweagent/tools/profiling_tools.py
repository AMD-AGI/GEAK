import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
import pandas as pd
from packaging.version import Version
import tempfile

from minisweagent.tools.prompt_for_profiling_analyzer import profiler_prompt

class ProfilingAnalyzer:
    
    def __init__(self, profiling_type: str, llm_model=None):
        self.profiling_type = profiling_type or "profiling"
        self.model = llm_model
        with tempfile.TemporaryDirectory(prefix="rocprof_") as tmpdir:
            self.output_path = Path(tmpdir).resolve()
    
    def _check_rocprof_compute(self):
        result_rocprof = subprocess.run([f'rocprof-compute --version'], capture_output=True, text=True, shell=True)
        if result_rocprof.returncode !=0:
            print("ROCProf is not installed. Starting installing.....")
            result = subprocess.run([f'sudo apt install rocprofiler-compute'], capture_output=True, text=True, shell=True)
            result = subprocess.run([f'sudo update-alternatives --install /usr/bin/rocprofiler-compute rocprof-compute /opt/rocm/bin/rocprofiler-compute 0'], capture_output=True, text=True, shell=True)
            result = subprocess.run([f'python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt'], capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                return None
            else:
                result_rocprof = subprocess.run([f'rocprof-compute --version'], capture_output=True, text=True, shell=True)
                pattern = r"rocprofiler-compute\s+version:\s*([0-9]+\.[0-9]+\.[0-9]+)"
                match = re.search(pattern, result_rocprof.stdout)
                return match.group(1) if match else None
        else:
            pattern = r"rocprofiler-compute\s+version:\s*([0-9]+\.[0-9]+\.[0-9]+)"
            match = re.search(pattern, result_rocprof.stdout)
            return match.group(1) if match else None
        

    def _extract_kernel_name(self) -> str:
        match = re.search(r'Kernel\s+\d+:\s*(.+?)\s*\.\.\.', self.content)
        return match.group(1) if match else "Unknown"
    
    def parse_roofline_rates(self) -> Dict[str, Tuple[float, float, str]]:
        rates = {}
        
        lines = self.content.split('\n')
        in_section = False
        
        for i, line in enumerate(lines):
            if '4.1 Roofline Rate Metrics' in line:
                in_section = True
                continue
            
            if in_section and '╘═' in line:
                break
                
            if in_section and '│' in line and '4.1.' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 6:
                    try:
                        metric_id = parts[1]
                        metric_name = parts[2]
                        value = float(parts[3])
                        unit = parts[4]
                        peak = float(parts[5])
                        
                        rates[metric_name] = (value, peak, unit)
                    except (ValueError, IndexError):
                        continue
        
        if not rates:
            print("Warning: Could not find 4.1 section")
        
        return rates
    
    def parse_profiling_top_kernel(self) -> Dict[str, Tuple[float, float, str]]:
        kernel_names = []
        try:
            csv_path = self.output_path / "pmc_kernel_top.csv"
            df = pd.read_csv(csv_path)
            kernels = df['Kernel_Name'].tolist()
            pct = df['Pct'].tolist()
            kernel_names = [k for k, p in zip(kernels, pct) if p > 1.0 and "amd_rocclr" not in k]
        except (ValueError, IndexError):
            print("Warning: Could not find top kernels")
        return kernel_names
    
    def parse_profiling_sys_info(self) -> Dict[str, Tuple[float, float, str]]:
        sys_info = {
            "gpu model": "gpu_model",
            "gpu architecture": "gpu_arch",
            "gpu L1": "gpu_l1",
            "gpu L2": "gpu_l2",
            "CU per gpu": "cu_per_gpu",
            "Req": "simd_per_cu",
            "Read Req": "wave_size",
            "Write Req": "workgroup_max_size",
            "Misses": "max_waves_per_cu",
            "Writeback": "lds_banks_per_cu",
            "Evict": "l2_banks",
            "Uncached Read": "total_l2_chan",
            "Uncached Write": "num_hbm_channels",
        }
        
        lines = self.content.split('\n')
        in_section = False
        
        for i, line in enumerate(lines):
            if '1. System Info' in line:
                in_section = True
                continue
            
            if in_section and '╘═' in line:
                break
                
            if in_section and '│' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 2:
                    try:
                        metric_id = parts[1]
                        info = parts[2]
                        for k, v in sys_info.items():
                            if v == metric_id:
                                sys_info[k]=info                        
                    except (ValueError, IndexError):
                        continue
        if not sys_info:
            print("Warning: Could not find sys info")
        
        return sys_info
    
    def parse_profiling_sys_speed(self) -> Dict[str, Tuple[float, float, str]]:
        
        def safe_float(x):
            try:
                return float(x)
            except:
                return None
        
        def classify_metric(metric_name):
            for cat, keywords in sys_speed.items():
                for k in keywords:
                    if k.lower() in metric_name.lower():
                        return cat
            return "other"
        
        def max_pct(metrics):
            vals = [m["pct_of_peak"] for m in metrics if m["pct_of_peak"] is not None]
            return max(vals) if vals else None


        def max_avg(metrics):
            vals = [m["avg"] for m in metrics if m["avg"] is not None]
            return max(vals) if vals else None


        def find_metric(metrics, keyword):
            for m in metrics:
                if keyword.lower() in m["metric"].lower():
                    return m["avg"]
            return None
        
        sys_speed = {
            "compute": [
                "FLOPs", "IOPs", "IPC", "VALU", "MFMA", "SALU"
            ],
            "occupancy": [
                "Active", "Occupancy", "Wavefront", "Threads"
            ],
            "memory_bw": [
                "Bandwidth", " BW"
            ],
            "cache": [
                "Cache Hit", "Hit Rate"
            ],
            "latency": [
                "Latency"
            ]
        }
        
        lines = self.content.split('\n')
        in_section = False
        rows=[]
        for i, line in enumerate(lines):
            if '2. System Speed-of-Light' in line:
                in_section = True
                continue
            
            if in_section and '╘═' in line:
                break
                
            if in_section and '│' in line and '2.1.' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 6:
                    try:
                        metric_id = parts[1]
                        metric = parts[2]
                        avg = safe_float(parts[3])
                        unit = parts[4]
                        peak = safe_float(parts[5])
                        pct_of_peak = safe_float(parts[6])
                        rows.append({
                            "metric": metric,
                            "avg": avg,
                            "unit": unit,
                            "peak": peak,
                            "pct_of_peak": pct_of_peak,
                        })
                    except (ValueError, IndexError):
                        continue
        grouped = defaultdict(list)
        for r in rows:
            cat = classify_metric(r["metric"])
            grouped[cat].append(r)

        core = {}
        # Compute bottleneck
        core["compute_util_pct"] = max_pct(grouped["compute"])
        core["ipc"] = find_metric(grouped["compute"], "IPC")
        core["valu_active_threads"] = find_metric(grouped["compute"], "Active Threads")

        # Occupancy
        core["occupancy"] = find_metric(grouped["occupancy"], "Occupancy")
        core["active_threads"] = find_metric(grouped["occupancy"], "Threads")

        # Memory bandwidth
        core["max_mem_bw_pct"] = max_pct(grouped["memory_bw"])

        # Cache efficiency
        core["l1_hit"] = find_metric(grouped["cache"], "L1")
        core["l2_hit"] = find_metric(grouped["cache"], "L2")

        # Latency
        core["max_latency_cycles"] = max_avg(grouped["latency"])
        
        if not core:
            print("Warning: Could not find sys speed")
        
        return core
    
    def parse_profiling_compute_units(self) -> Dict[str, Tuple[float, float, str]]:
        
        def safe_float(x):
            try:
                return float(x)
            except:
                return None
                
        lines = self.content.split('\n')
        in_section = False
        current_section = None
        tables = defaultdict(dict)
        for i, line in enumerate(lines):
            if '10. Compute Units - Instruction Mix' in line:
                in_section = True
                continue
            
            if in_section and '16. Vector L1 Data Cache' in line:
                break
            if in_section:
                m = re.match(r'^\s*(\d+\.\d+)\s+', line)
                if m:
                    current_section = m.group(1)
                    continue

            if in_section and '│' in line and current_section and '10.' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 6:
                    try:
                        metric = parts[2]
                        avg = safe_float(parts[3])
                        tables[current_section][metric]=avg
                    except (ValueError, IndexError):
                        continue
        
        features = {}
        # ---------- 10.1 Overall Instruction Mix ----------
        mix = tables.get("10.1", {})
        total = sum(mix.values()) if mix else 0.0

        features["total_instructions"] = total

        def ratio(x):
            return x / total if total > 0 else 0.0

        features["valu_ratio"]   = ratio(mix.get("VALU", 0.0))
        features["vmem_ratio"]   = ratio(mix.get("VMEM", 0.0))
        features["salu_ratio"]   = ratio(mix.get("SALU", 0.0))
        features["branch_ratio"] = ratio(mix.get("Branch", 0.0))

        features["compute_ratio"] = ratio(
            mix.get("VALU", 0.0) + mix.get("MFMA", 0.0)
        )
        features["memory_ratio"] = ratio(
            mix.get("VMEM", 0.0) + mix.get("SMEM", 0.0) + mix.get("LDS", 0.0)
        )

        # ---------- 10.2 VALU Arithmetic ----------
        valu = tables.get("10.2", {})
        int_ops = sum(v for k, v in valu.items() if "INT" in k)
        fp_ops  = sum(v for k, v in valu.items() if k.startswith("F"))

        total_alu = int_ops + fp_ops
        features["int_ratio"] = int_ops / total_alu if total_alu > 0 else 0.0
        features["fp_ratio"]  = fp_ops / total_alu if total_alu > 0 else 0.0

        # ---------- 10.3 VMEM Mix ----------
        vmem = tables.get("10.3", {})
        vmem_total = vmem.get("Global/Generic Instr", 0.0)

        read  = vmem.get("Global/Generic Read", 0.0)
        write = vmem.get("Global/Generic Write", 0.0)

        features["global_read_ratio"]  = read / vmem_total if vmem_total > 0 else 0.0
        features["global_write_ratio"] = write / vmem_total if vmem_total > 0 else 0.0
        features["read_write_balance"] = (
            read / write if write > 0 else float("inf")
        )

        # ---------- MFMA presence ----------
        mfma = tables.get("10.4", {})
        features["uses_mfma"] = any(v > 0 for v in mfma.values() if v is not None)
        
        if not features:
            print("Warning: Could not find compute units")
        
        return features
    
    def parse_profiling_l1_data(self) -> Dict[str, Tuple[float, float, str]]:
        
        def safe_float(x):
            try:
                return float(x)
            except:
                return None
                
        lines = self.content.split('\n')
        in_section = False
        feat = defaultdict(dict)
        reads = writes = atomics = total = 0
        for i, line in enumerate(lines):
            if '16. Vector L1 Data Cache' in line:
                in_section = True
                continue
            
            if in_section and '17. L2 Cache' in line:
                break
            # ---------- L1 Cache Effectiveness ----------
            if in_section and '│' in line and '16.1' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 4:
                    try:
                        lk = parts[2].lower()
                        if "hit rate" in lk:
                            feat["l1_cache"]["hit_rate_pct"] = safe_float(parts[3])
                        elif "bandwidth" in lk:
                            feat["l1_cache"]["bandwidth_util_pct"] = safe_float(parts[3])
                        elif lk == "utilization":
                            feat["l1_cache"]["utilization_pct"] = safe_float(parts[3])
                        elif "coalescing" in lk:
                            feat["l1_cache"]["coalescing_pct"] = safe_float(parts[3])
                    except (ValueError, IndexError):
                        continue
            
            # ---------- Stall Severity ----------
            if in_section and '│' in line and '16.2' in line:
                feat["l1_stall"].setdefault("tag_ram_stall_present", False)
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 7:
                    try:
                        lk = parts[2].lower()
                        if "stalled on l2 data" in lk:
                            feat["l1_stall"]["l2_data_stall_median_pct"] = safe_float(parts[5])
                        if "tag ram stall" in lk and max(safe_float(parts[3]),safe_float(parts[4]),safe_float(parts[5])) > 0:
                            feat["l1_stall"]["tag_ram_stall_present"] = True
                    except (ValueError, IndexError):
                        continue
            
            # ---------- Access Pattern ----------
            if in_section and '│' in line and '16.3' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 6:
                    try:
                        lk = parts[2].lower()
                        if lk == "total req":
                            total = safe_float(parts[3])
                        elif "read req" in lk:
                            reads = safe_float(parts[3])
                        elif "write req" in lk:
                            writes = safe_float(parts[3])
                        elif "atomic req" in lk:
                            atomics = safe_float(parts[3])
                        elif "cache bw" in lk:
                            feat["l1_access"]["avg_cache_bw_gbps"] = safe_float(parts[3])
                        elif "l1-l2 bw" in lk:
                            feat["l1_l2_traffic"]["l1_l2_bw_gbps"] = safe_float(parts[3])
                        
                        if "l1-l2 atomic" in lk:
                            if reads + writes > 0:
                                feat["l1_access"]["read_write_ratio"] = reads / max(writes, 1e-6)
                            feat["l1_access"]["atomic_ratio"] = atomics / max(total or 1, 1)
                            feat["l1_access"]["total_reqs"] = total
                    except (ValueError, IndexError):
                        continue                    
            
             # ---------- TLB ----------
            if in_section and '│' in line and '16.5' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 6:
                    try:
                        lk = parts[2].lower()
                        if "hit ratio" in lk:
                            feat["tlb"]["hit_rate_pct"] = safe_float(parts[3])
                        elif "translation misses" in lk:
                            feat["tlb"]["miss_rate_pct"] = safe_float(parts[3]) / max(feat["l1_access"].get("total_reqs", 1), 1)
                    except (ValueError, IndexError):
                        continue
        
        if not feat:
            print("Warning: Could not find l1 data")
        
        return feat
    
    def parse_profiling_l2_data(self) -> Dict[str, Tuple[float, float, str]]:
        
        def safe_float(x):
            try:
                return float(x)
            except:
                return None
                
        lines = self.content.split('\n')
        in_section = False
        feat = defaultdict(dict)
        for i, line in enumerate(lines):
            if '17. L2' in line:
                in_section = True
                continue
            
            if in_section and '18. L2' in line:
                break
            # ---------- L1 Cache Effectiveness ----------
            if in_section and '│' in line and '17.' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 4:
                    try:
                        lk = parts[2].lower()
                        # ---- bandwidth ----
                        if "bandwidth" in lk and "17.1" in parts[1]:
                            feat["l2_bandwidth"]["utilization"] = safe_float(parts[3])
                        elif "read bw" in lk:
                            feat["l2_bandwidth"]["read_bw_gbps"] = safe_float(parts[3])
                        elif "write and atomic bw" in lk:
                            feat["l2_bandwidth"]["write_bw_gbps"] = safe_float(parts[3])
                        elif "l2-fabric write and atomic bw" in lk:
                            feat["l2_bandwidth"]["write_bw_gbps"] = safe_float(parts[3])
                        # ---- locality ----
                        elif "hit rate" in lk:
                            feat["l2_locality"]["hit_rate"] = safe_float(parts[3])
                        elif "misses" in lk:
                            feat["l2_locality"]["miss_ratio"] = safe_float(parts[3])
                        # ---- access pattern ----
                        elif "read req" in lk:
                            feat["l2_access_pattern"]["read_ratio"] = safe_float(parts[3])
                        elif "write req" in lk:
                            feat["l2_access_pattern"]["write_ratio"] = safe_float(parts[3])
                        elif "uncached read traffic" in lk:
                            feat["l2_access_pattern"]["uncached_read_pct"] = safe_float(parts[3])
                        elif "hbm read traffic" in lk:
                            feat["l2_access_pattern"]["hbm_read_pct"] = safe_float(parts[3])
                        # ---- granularity ----
                        elif "read (128b)" in lk:
                            feat["granularity"]["read_128b"] = safe_float(parts[3])
                        elif "read (64b)" in lk:
                            feat["granularity"]["read_64b"] = safe_float(parts[3])
                        elif "write and atomic (64b)" in lk:
                            feat["granularity"]["write_64b"] = safe_float(parts[3])
                        # ---- latency ----
                        elif "read latency" in lk:
                            feat["latency"]["read_cycles"] = safe_float(parts[3])
                        elif "write and atomic latency" in lk:
                            feat["latency"]["write_cycles"] = safe_float(parts[3])                       
                        
                    except (ValueError, IndexError):
                        continue
        
        if not feat:
            print("Warning: Could not find l2 data")
        
        return feat
    
    def parse_profiling_wavefront(self) -> Dict[str, Tuple[float, float, str]]:
        
        def safe_float(x):
            try:
                return float(x)
            except:
                return None
        KEY_METRICS = [
            "VGPRs", "SGPRs", "AGPRs", "LDS Allocation", "Scratch Allocation",
            "Wavefront Occupancy", "Dependency Wait Cycles", "Issue Wait Cycles",
            "Active Cycles", "Kernel Time", "Instructions per wavefront"
        ]
        lines = self.content.split('\n')
        in_section = False
        feat = defaultdict(list)
        for i, line in enumerate(lines):
            if '7.1 Wavefront Launch Stats' in line:
                in_section = True
                continue
            
            # ---------- L1 Cache Effectiveness ----------
            if in_section and '│' in line and '7.' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 6:
                    try:
                        lk = parts[2]
                        if lk in KEY_METRICS:
                            feat[lk].append(safe_float(parts[3]))
                            feat[lk].append(parts[6])
                    except (ValueError, IndexError):
                        continue
        
        if not feat:
            print("Warning: Could not find wavefront")
        return feat

    def parse_roofline_ai(self) -> Dict[str, Tuple[float, str]]:
        ai_metrics = {}
        
        lines = self.content.split('\n')
        in_section = False
        
        for i, line in enumerate(lines):
            if '4.2 Roofline AI Plot Points' in line:
                in_section = True
                continue
            
            if in_section and '╘═' in line:
                break
                
            if in_section and '│' in line and '4.2.' in line:
                parts = [p.strip() for p in line.split('│')]
                if len(parts) >= 5:
                    try:
                        metric_id = parts[1]
                        metric_name = parts[2]
                        value = float(parts[3])
                        unit = parts[4]
                        
                        if 'HBM' in metric_name or 'Performance' in metric_name:
                            if 'Performance' in metric_name and 'Gflop' in unit:
                                value = value / 1000.0
                                unit = 'TFLOPS'
                                metric_name = 'Performance (TFLOPs)'
                            
                            ai_metrics[metric_name] = (value, unit)
                    except (ValueError, IndexError):
                        continue
        
        if not ai_metrics:
            print("Warning: Could not find 4.2 section")
        
        return ai_metrics
    
    def categorize_metrics(self, rates: Dict) -> Dict:
        categorized = {
            'bandwidth': {},
            'compute': {}
        }
        
        for metric_name, (value, peak, unit) in rates.items():
            if 'HBM' in metric_name and 'Bandwidth' in metric_name:
                categorized['bandwidth'][metric_name] = {
                    'actual': value,
                    'peak': peak,
                    'unit': unit,
                    'utilization_pct': (value / peak * 100) if peak > 0 else 0
                }
            elif 'FLOPs' in metric_name or 'IOPs' in metric_name:
                if value > 0:
                    categorized['compute'][metric_name] = {
                        'actual': value,
                        'peak': peak,
                        'unit': unit,
                        'utilization_pct': (value / peak * 100) if peak > 0 else 0
                    }
        
        return categorized
    
    def more_profiling(self):
        sys_info = self.parse_profiling_sys_info()
        more_profiler = "\nBelow are some more profiling information of this kernel, you can reference these info to analyze and generate better kernel."
        more_profiler += f"\nSection 1.0: The following are GPU system information:\n"
        for k,v in sys_info.items():
            more_profiler += f"- {k}: {v}\n"
        
        top_kernels = self.parse_profiling_top_kernel()
        more_profiler += f"\nSection 2.0: The following are the top-performing kernels that need optimization:\n"
        for k in top_kernels[:3]:
            more_profiler += f"- {k}\n"
        
        sys_speedup = self.parse_profiling_sys_speed()
        more_profiler += "\nSection 3.0: The following are Performance Utilization & Bottlenecks:"
        more_profiler += f"\n- compute_util_pct: percentage of peak compute throughput achieved. value: {sys_speedup["compute_util_pct"]}"
        more_profiler += f"\n- ipc: instructions per cycle. value: {sys_speedup["ipc"]}"
        more_profiler += f"\n- valu_active_threads: effective SIMD utilization. value: {sys_speedup["valu_active_threads"]}"
        more_profiler += f"\n- occupancy: active wavefronts relative to hardware limit. value: {sys_speedup["occupancy"]}"
        more_profiler += f"\n- active_threads: average active threads per wave. value: {sys_speedup["active_threads"]}"
        more_profiler += f"\n- max_mem_bw_pct: achieved memory bandwidth as percentage of peak. value: {sys_speedup["max_mem_bw_pct"]}"
        more_profiler += f"\n- l1_hit / l2_hit: cache hit rates indicating locality. value: {sys_speedup["l1_hit"]} / {sys_speedup["l2_hit"]}"
        more_profiler += f"\n- max_latency_cycles: observed memory or fabric latency. value: {sys_speedup["max_latency_cycles"]}"

        compute_units = self.parse_profiling_compute_units()
        more_profiler += "\nSection 4.0: The following are Compute Units - Instruction Mix:"
        more_profiler += f"\n- Integer operations account for {compute_units["int_ratio"]*100:.1f}% of all arithmetic instructions"
        more_profiler += f"\n- Floating-point operations account for {compute_units["fp_ratio"]*100:.1f}% of all arithmetic instructions"
        more_profiler += f"\n- Global memory reads account for {compute_units["global_read_ratio"]*100:.1f}% of memory operations"
        more_profiler += f"\n- Global memory writes account for {compute_units["global_write_ratio"]*100:.1f}% of memory operations"
        more_profiler += f"\n- Read/write ratio (reads divided by writes) is {compute_units["read_write_balance"]:.2f}"
        more_profiler += f"\n- MFMA instructions used: {compute_units["uses_mfma"]}"

        l1_data = self.parse_profiling_l1_data()
        more_profiler += "\nSection 5.0: The following are informantion in Vector L1 Data Cache:"
        more_profiler += f"\n- L1 cache hit rate: Indicates how much kernel data reuse is captured by L1 cache. value: {l1_data["l1_cache"]["hit_rate_pct"]}"
        more_profiler += f"\n- L1 bandwidth utilization: Measures utilization of L1 bandwidth. value: {l1_data["l1_cache"]["bandwidth_util_pct"]}"
        more_profiler += f"\n- L1 stalled on L2 data: Reflects how often L1 waits on L2 responses. value: {l1_data["l1_stall"]["l2_data_stall_median_pct"]}"
        more_profiler += f"\n- Memory access coalescing: Describes how well memory accesses are merged. value: {l1_data["l1_cache"]["coalescing_pct"]}"
        more_profiler += f"\n- L1-L2 bandwidth: Measures traffic from L1 to L2. value: {l1_data["l1_l2_traffic"]["l1_l2_bw_gbps"]}"
        more_profiler += f"\n- Read/Write Ratio:Ratio of read requests to write requests; indicates memory access pattern. value: {l1_data["l1_access"]["read_write_ratio"]}"
        more_profiler += f"\n- Atomic Ratio: Fraction of atomic accesses. value: {l1_data["l1_access"]["atomic_ratio"]}"
        more_profiler += f"\n- TLB hit rate: Indicates effectiveness of address translation cache. value: {l1_data["tlb"]["hit_rate_pct"]}"
        more_profiler += f"\n- TLB miss rate: Fraction of requests that missed the TLB. value: {l1_data["tlb"]["miss_rate_pct"] }"

        l2_data = self.parse_profiling_l2_data()
        more_profiler += "\nSection 6.0: The following are informantion in L2 Cache:"
        more_profiler += f"\n- L2 cache hit rate: {l2_data["l2_locality"]["hit_rate"]}"
        more_profiler += f"\n- l2_bw_util: {l2_data["l2_bandwidth"]["utilization"]}"
        more_profiler += f"\n- read_latency_cycles: {l2_data["latency"]["write_cycles"]}"
        more_profiler += f"\n- write_ratio: {l2_data["l2_access_pattern"]["write_ratio"]}"
        more_profiler += f"\n- uncached_read_pct: {l2_data["l2_access_pattern"]["uncached_read_pct"]}"
        more_profiler += f"\n- hbm_read_pct: {l2_data["l2_access_pattern"]["hbm_read_pct"]}"

        wavefront = self.parse_profiling_wavefront()
        total_cycles = wavefront["Dependency Wait Cycles"][0] + wavefront["Issue Wait Cycles"][0] + wavefront["Active Cycles"][0]
        more_profiler += "\nSection 7.0: The following are Threading & Allocation:"
        more_profiler += f"\n- VGPR pressure: {wavefront["VGPRs"][0]} {wavefront["VGPRs"][1]}"
        more_profiler += f"\n- SGPR pressure: {wavefront["SGPRs"][0]} {wavefront["SGPRs"][1]}"
        more_profiler += f"\n- AGPR pressure: {wavefront["AGPRs"][0]} {wavefront["AGPRs"][1]}"
        more_profiler += f"\n- Shared memory usage: {wavefront["LDS Allocation"][0]}  {wavefront["LDS Allocation"][1]}"
        more_profiler += f"\n- Scratch memory usage: {wavefront["Scratch Allocation"][0]} {wavefront["Scratch Allocation"][1]}"
        more_profiler += f"\n- Occupancy signal: {wavefront["Wavefront Occupancy"][0]} {wavefront["Wavefront Occupancy"][1]}"
        more_profiler += f"\n- Dependency wait ratio: {round(wavefront["Dependency Wait Cycles"][0]/total_cycles,2)*100}%"
        more_profiler += f"\n- Issue wait ratio: {round(wavefront["Issue Wait Cycles"][0]/total_cycles,2)*100}%"
        more_profiler += f"\n- Active execution ratio: {round(wavefront["Active Cycles"][0]/total_cycles,2)*100}%"
        more_profiler += f"\n- Instructions per wavefront: {wavefront["Instructions per wavefront"][0]} {wavefront["Instructions per wavefront"][1]}"
        return more_profiler
    
    def roofline_summary(self, categorized: Dict, ai_metrics: Dict):
        top_kernels = self.parse_profiling_top_kernel()        
        roofline = "\nBelow is the roofline information of the kernel:"
        roofline += f"\nkernel function name:\n"
        for k in top_kernels[:3]:
            roofline += f"- {k}\n"
        if categorized['bandwidth']:
            roofline += "\nHBM BANDWIDTH UTILIZATION:"
            for metric_name, data in categorized['bandwidth'].items():
                roofline += f"\n- {metric_name}: actual: {data['actual']} peak: {data['peak']} utilization_pct: {data['utilization_pct']}"
        if categorized['compute']:
            roofline += "\nCOMPUTE UTILIZATION:"
            for metric_name, data in categorized['compute'].items():
                roofline += f"\n- {metric_name}: actual: {data['actual']} peak: {data['peak']} utilization_pct: {data['utilization_pct']}"
        if ai_metrics:
            roofline += "\nARITHMETIC INTENSITY:"
            for metric_name, (value, unit) in ai_metrics.items():
                if unit:
                    roofline += f"\n- {metric_name}: value: {value} {unit}"
                else:
                    roofline += f"\n- {metric_name}: value: {value}"
        return roofline
    
    def analyze(self):
        rates = self.parse_roofline_rates()
        ai_metrics = self.parse_roofline_ai()
        categorized = self.categorize_metrics(rates)
        roofline = self.roofline_summary(categorized, ai_metrics)
        more_profiler = self.more_profiling()
        
        return {
            'roofline': roofline,
            'profiling': more_profiler
        }
    
    def __call__(self, profiling_workdir: str, profiling_cmd: str, *args, **kwds):
        kernel_name = Path(profiling_workdir).name
        results = {}
        rocprof_version = self._check_rocprof_compute()
        if not profiling_workdir or not profiling_cmd:
            return {
                "output": "No profiling_workdir and profiling_cmd arguments are provided.",
                "returncode": 1,
            }
        if rocprof_version is None:
            return {
                "output": "No ROCProf is installed. CAN NOT get profiling information.",
                "returncode": 1,
            }        
        use_profiling = Version(rocprof_version) < Version("3.3.1") and (self.profiling_type == 'roofline' or self.profiling_type == 'profiling')
        if self.profiling_type == 'profiling' or use_profiling or self.profiling_type == 'profiler_analyzer':
            make_cmd = [f"rocprof-compute profile -n {kernel_name} --path {self.output_path} -- {profiling_cmd}"]
        elif self.profiling_type == 'roofline':
            make_cmd = [f"rocprof-compute profile -n {kernel_name} --path {self.output_path} --roof-only -- {profiling_cmd}"]
        else: 
            return {
                "output": "No profiling information",
                "returncode": 1,
            }
        result = subprocess.run(make_cmd, shell=True, cwd=profiling_workdir, capture_output=True, text=True, timeout=3600*6)
        if result.returncode == 0:
            if self.profiling_type == 'profiling'  or use_profiling:
                analysis_cmd =[f"rocprof-compute analyze -p {self.output_path} -b 0 1 2 4 7 10 16 17"]
            elif self.profiling_type == 'roofline':
                analysis_cmd =[f"rocprof-compute analyze -p {self.output_path} -b 4"]
            elif self.profiling_type == 'profiler_analyzer':
                analysis_cmd =[f"rocprof-compute analyze -p {self.output_path} -b 0 1 2 4 7 10 11 16 17"]
            result = subprocess.run(analysis_cmd, shell=True, cwd=profiling_workdir, capture_output=True, text=True, timeout=3600*6)
            if result.returncode == 0:
                
                if self.profiling_type == 'profiler_analyzer' and self.model is not None:
                    msg = profiler_prompt.format(profiler_output = result.stdout)
                    prompt = [{"role": "user", "content": msg}]
                    response = self.model.query(prompt)
                    results ={
                        "output": response["content"] if response["content"] else result.stdout,
                        "returncode": 0,
                    }
                else:
                    self.content = result.stdout
                    analyzer = self.analyze()
                    if use_profiling:
                        results ={
                            "output": analyzer["profiling"],
                            "returncode": result.returncode,
                            }
                    elif self.profiling_type=='profiling':
                        results ={
                            "output": analyzer["roofline"] + analyzer["profiling"],
                            "returncode": result.returncode,
                            }
                    elif self.profiling_type=='roofline':
                        results ={
                            "output": analyzer["roofline"],
                            "returncode": result.returncode,
                        }
                    return results
        return {
            "output":  result.stdout.strip() or result.stderr.strip(),
            "returncode": 1,
        }

if __name__ == "__main__":
    repo= "/mcp/task/AIG-Eval-Internal-Tasks/fused_bucketized"
    profiling_cmd= "./applications_fused_bucketized"
    output_path= "/mcp/outputs/l1/fused_bucketized/profiling"
    kernel_profiling = ProfilingAnalyzer(profiling_type='roofline')
    response = kernel_profiling(profiling_workdir=repo, profiling_cmd=profiling_cmd)
    print(response["output"])