#!/usr/bin/env python3
"""Collect, parse, classify, and compare rocprof-compute roofline reports.

The parser and policy-facing helpers are importable and require only the Python
standard library.  The CLI adds two subcommands:

    roofline_kernel.py collect --manifest MANIFEST --phase baseline --out-dir OUT
    roofline_kernel.py compare --before BASELINE.json --after AFTER.json --out DIFF.json
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import roofline_policy


SCHEMA_VERSION = "roofline-v1"
TOP_LEVEL_STATUSES = ("ok", "partial", "skipped", "failed")
CASE_STATUSES = ("matched", "skipped", "failed")
DEFAULT_TIMEOUT_SEC = 1800.0
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
METRIC_ID_RE = re.compile(r"(?<![\d.])((?:2|4|17)\.\d+(?:\.\d+)?)\b")
KERNEL_HEADER_RES = (
    re.compile(r"^\s*(?:[|+├└─\s]*)Kernel\s+(\d+)\s*:\s*(.+?)\s*$", re.I),
    re.compile(r"^\s*(?:[|+├└─\s]*)Kernel(?:\s+Name)?\s*:\s*(.+?)\s*$", re.I),
    re.compile(r"^\s*(?:[|+├└─\s]*)Kernel\s*\[\s*(\d+)\s*\]\s*(?:=|:)\s*(.+?)\s*$", re.I),
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
MISSING_VALUES = {"", "-", "--", "n/a", "na", "nan", "none", "null"}


def strip_ansi(text):
    """Remove CSI/OSC ANSI terminal sequences from text."""
    return ANSI_RE.sub("", text or "")


def _finite_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_VALUES:
        return None
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        result = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _clean_kernel_name(value):
    value = value.strip().strip("|│").strip()
    return re.sub(r"\s+\(\s*\d+(?:\.\d+)?%\s*\)\s*$", "", value).strip()


def _kernel_header(line):
    for index, expression in enumerate(KERNEL_HEADER_RES):
        match = expression.match(line)
        if not match:
            continue
        if index == 1:
            return None, _clean_kernel_name(match.group(1))
        return int(match.group(1)), _clean_kernel_name(match.group(2))
    return None


def _table_cells(line):
    normalized = line.replace("┃", "│").replace("║", "│")
    if "│" in normalized:
        return [cell.strip() for cell in normalized.split("│") if cell.strip()]
    if "|" in normalized:
        return [cell.strip() for cell in normalized.split("|") if cell.strip()]
    return []


def _metric_row(line):
    """Parse one report row into id/name/value/unit/peak."""
    id_match = METRIC_ID_RE.search(line)
    if not id_match:
        return None
    metric_id = id_match.group(1)
    cells = _table_cells(line)
    id_index = None
    for index, cell in enumerate(cells):
        if re.search(r"(?<![\d.])%s\b" % re.escape(metric_id), cell):
            id_index = index
            break
    if id_index is not None and id_index + 1 < len(cells):
        following = cells[id_index + 1 :]
        name = following[0].strip()
        value = _finite_number(following[1]) if len(following) > 1 else None
        unit = following[2].strip() if len(following) > 2 else ""
        peak = _finite_number(following[3]) if len(following) > 3 else None
        return {
            "metric_id": metric_id,
            "metric": name,
            "value": value,
            "unit": unit,
            "peak": peak,
            "raw": line.strip(),
        }

    remainder = line[id_match.end() :].strip(" :-\t")
    parts = re.split(r"\s{2,}|\t+", remainder)
    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        return None
    name = parts[0]
    value = _finite_number(parts[1]) if len(parts) > 1 else None
    unit = parts[2] if len(parts) > 2 else ""
    peak = _finite_number(parts[3]) if len(parts) > 3 else None
    return {
        "metric_id": metric_id,
        "metric": name,
        "value": value,
        "unit": unit,
        "peak": peak,
        "raw": line.strip(),
    }


def _new_kernel(index, name):
    return {
        "kernel_index": index,
        "kernel_name": name,
        "rows": [],
        "compute_rates": [],
        "metrics": {},
        "warnings": [],
    }


def _is_compute_rate(name):
    upper = name.upper()
    return (
        ("VALU" in upper or "MFMA" in upper)
        and ("FLOP" in upper or "IOP" in upper)
        and "UTIL" not in upper
    )


def _dtype_tokens(dtypes):
    if isinstance(dtypes, str):
        dtypes = [dtypes]
    tokens = []
    for dtype in dtypes or []:
        token = re.sub(r"[^A-Z0-9]", "", str(dtype).upper())
        aliases = [token]
        if "BF16" in token:
            aliases.extend(["BF16", "B16"])
        if "FP16" in token or "F16" in token:
            aliases.extend(["FP16", "F16"])
        if "FP32" in token or "F32" in token:
            aliases.extend(["FP32", "F32"])
        if "FP64" in token or "F64" in token:
            aliases.extend(["FP64", "F64"])
        if "FP8" in token or "F8" in token:
            aliases.extend(["FP8", "F8"])
        if "FP6" in token or "F6" in token:
            aliases.extend(["FP6", "F6"])
        if "FP4" in token or "F4" in token:
            aliases.extend(["FP4", "F4"])
        if "MX" in token:
            for width in ("4", "6", "8"):
                if width in token:
                    aliases.extend(["MXFP" + width, "MXF" + width])
        if "INT8" in token or "I8" in token:
            aliases.extend(["INT8", "I8"])
        tokens.extend(aliases)
    return list(dict.fromkeys(item for item in tokens if item))


def select_dominant_compute_rate(compute_rates, dtypes=None):
    """Select a compute rate by manifest dtype, then by largest nonzero value."""
    if not compute_rates:
        return None
    dtype_tokens = _dtype_tokens(dtypes)

    def normalized_name(rate):
        return re.sub(r"[^A-Z0-9]", "", rate.get("metric", "").upper())

    matching = []
    for rate in compute_rates:
        name = normalized_name(rate)
        scores = []
        for token in dtype_tokens:
            if token in ("F16", "FP16") and "BF16" in name:
                continue
            score = len(token) if token in name else 0
            if token.startswith(("F", "FP")) and "MXFP" in name:
                score = min(score, 1)
            scores.append(score)
        score = max(scores or [0])
        if score:
            matching.append((score, rate))
    if matching:
        best_score = max(item[0] for item in matching)
        pool = [rate for score, rate in matching if score == best_score]
    else:
        pool = []
    if not pool:
        pool = list(compute_rates)
    nonzero = [
        rate for rate in pool
        if rate.get("value") is not None and rate.get("value") > 0
    ]
    if nonzero:
        return max(nonzero, key=lambda rate: rate["value"])
    with_values = [rate for rate in pool if rate.get("value") is not None]
    return max(with_values, key=lambda rate: rate["value"]) if with_values else pool[0]


def _find_row(rows, metric_id=None, names=()):
    lowered = tuple(name.lower() for name in names)
    if lowered:
        for row in rows:
            metric = row["metric"].lower()
            if any(name in metric for name in lowered):
                return row
    if metric_id:
        for row in rows:
            if row["metric_id"] == metric_id:
                return row
    return None


def _rate_by_name(rows, names):
    return _find_row(rows, names=names)


def _utilization(actual, peak):
    if actual is None or peak is None or peak <= 0:
        return None
    return 100.0 * actual / peak


def derive_metrics(rows, dtypes=None):
    """Derive normalized roofline metrics from parsed report rows."""
    rate_rows = [
        row for row in rows
        if row["metric_id"].startswith("4.1.") and _is_compute_rate(row["metric"])
    ]
    compute_rates = [
        {
            "metric_id": row["metric_id"],
            "metric": row["metric"],
            "value": row["value"],
            "unit": row["unit"],
            "empirical_peak": row["peak"],
        }
        for row in rate_rows
    ]
    dominant = select_dominant_compute_rate(compute_rates, dtypes)

    ai_hbm_row = _find_row(rows, "4.2.0", ("ai hbm", "hbm ai"))
    ai_l2_row = _find_row(rows, "4.2.1", ("ai l2", "l2 ai"))
    ai_l1_row = _find_row(rows, "4.2.2", ("ai l1", "l1 ai"))
    # Metric IDs changed when AI LDS was added (4.2.3 became AI LDS and
    # Performance moved to 4.2.4). Match the semantic name across versions.
    performance_row = _find_row(
        rows, names=("performance (gflop", "performance (giop", "performance")
    )
    hbm_rate = _rate_by_name(
        [row for row in rows if row["metric_id"].startswith("4.1.")],
        ("hbm bandwidth", "hbm bw"),
    )
    l2_rate = _rate_by_name(
        [row for row in rows if row["metric_id"].startswith("4.1.")],
        ("l2 cache bandwidth", "l2 bandwidth", "l2 bw"),
    )
    l1_rate = _rate_by_name(
        [row for row in rows if row["metric_id"].startswith("4.1.")],
        ("l1 cache bandwidth", "l1 bandwidth", "l1 bw"),
    )
    lds_rate = _rate_by_name(
        [row for row in rows if row["metric_id"].startswith("4.1.")],
        ("lds bandwidth", "lds bw"),
    )
    spec_hbm = _find_row(rows, "17.1.5")
    if spec_hbm is None:
        spec_hbm = _find_row(
            [row for row in rows if row["metric_id"].startswith("17.")],
            names=("peak hbm bandwidth", "theoretical hbm bandwidth", "hbm peak"),
        )

    compute_actual = dominant.get("value") if dominant else None
    compute_peak = dominant.get("empirical_peak") if dominant else None
    performance = performance_row["value"] if performance_row else None
    if performance is None:
        performance = compute_actual
    metrics = {
        "ai_hbm": ai_hbm_row["value"] if ai_hbm_row else None,
        "ai_l2": ai_l2_row["value"] if ai_l2_row else None,
        "ai_l1": ai_l1_row["value"] if ai_l1_row else None,
        "ai_ridge_empirical": None,
        "performance_gflops": performance,
        "compute_metric": dominant.get("metric") if dominant else None,
        "compute_actual_gflops": compute_actual,
        "compute_empirical_peak_gflops": compute_peak,
        "compute_utilization_pct": _utilization(compute_actual, compute_peak),
        "hbm_actual_gbps": hbm_rate["value"] if hbm_rate else None,
        "hbm_empirical_peak_gbps": hbm_rate["peak"] if hbm_rate else None,
        "hbm_spec_peak_gbps": spec_hbm["value"] if spec_hbm else None,
        "hbm_utilization_pct": _utilization(
            hbm_rate["value"] if hbm_rate else None,
            hbm_rate["peak"] if hbm_rate else None,
        ),
        "l2_utilization_pct": _utilization(
            l2_rate["value"] if l2_rate else None,
            l2_rate["peak"] if l2_rate else None,
        ),
        "l1_utilization_pct": _utilization(
            l1_rate["value"] if l1_rate else None,
            l1_rate["peak"] if l1_rate else None,
        ),
        "lds_utilization_pct": _utilization(
            lds_rate["value"] if lds_rate else None,
            lds_rate["peak"] if lds_rate else None,
        ),
        "roofline_efficiency_pct": None,
        "headroom_ratio": None,
        "peak_basis": "empirical",
    }
    efficiency = roofline_policy.compute_roofline_efficiency(metrics)
    metrics.update(
        {
            "ai_ridge_empirical": efficiency["ai_ridge_empirical"],
            "roofline_efficiency_pct": efficiency["roofline_efficiency_pct"],
            "headroom_ratio": efficiency["headroom_ratio"],
            "peak_basis": efficiency["peak_basis"],
            "roofline_empirical_ceiling_gflops": efficiency[
                "roofline_empirical_ceiling_gflops"
            ],
        }
    )
    no_values = [rate["value"] for rate in compute_rates if rate["value"] is not None]
    metrics["no_fp_work"] = bool(no_values) and max(no_values) == 0
    return metrics, compute_rates


class RocprofComputeParser(object):
    """Parser for old and new rocprof-compute analyze text."""

    def __init__(self, dtypes=None, saturation_pct=roofline_policy.DEFAULT_SATURATION_PCT):
        self.dtypes = dtypes or []
        self.saturation_pct = saturation_pct

    def parse(self, text):
        cleaned = strip_ansi(text)
        kernels = []
        current = None
        global_rows = []
        for line in cleaned.splitlines():
            header = _kernel_header(line)
            if header is not None:
                index, name = header
                if current is not None:
                    kernels.append(current)
                if index is None:
                    index = len(kernels)
                current = _new_kernel(index, name)
                continue
            row = _metric_row(line)
            if row is None:
                continue
            # Section 17 describes device-wide hardware/specification limits, not a
            # single dispatch. Analyze normally prints it after the final kernel,
            # so always propagate it to every parsed kernel.
            if row["metric_id"].startswith("17.") or current is None:
                global_rows.append(row)
            else:
                current["rows"].append(row)
        if current is not None:
            kernels.append(current)
        if not kernels and global_rows:
            kernels.append(_new_kernel(0, "unknown"))
            kernels[0]["rows"] = global_rows
        elif global_rows:
            for kernel in kernels:
                existing = {row["metric_id"] for row in kernel["rows"]}
                kernel["rows"].extend(
                    row for row in global_rows if row["metric_id"] not in existing
                )

        for kernel in kernels:
            metrics, compute_rates = derive_metrics(kernel["rows"], self.dtypes)
            kernel["metrics"] = metrics
            kernel["compute_rates"] = compute_rates
            kernel["classification"] = roofline_policy.build_classification(
                metrics, saturation_pct=self.saturation_pct
            )
        return kernels


def parse_rocprof_compute(
    text, dtypes=None, saturation_pct=roofline_policy.DEFAULT_SATURATION_PCT
):
    """Parse analyze text and return one normalized record per kernel."""
    return RocprofComputeParser(
        dtypes=dtypes, saturation_pct=saturation_pct
    ).parse(text)


def parse_analyze_text(
    text, dtypes=None, saturation_pct=roofline_policy.DEFAULT_SATURATION_PCT
):
    """Compatibility alias for parse_rocprof_compute."""
    return parse_rocprof_compute(
        text, dtypes=dtypes, saturation_pct=saturation_pct
    )


def _atomic_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".roofline-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_text(path, text):
    with open(path, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(text)


def locate_rocprof_compute(env=None):
    """Locate rocprof-compute using overrides, PATH, then common /opt paths."""
    environment = os.environ if env is None else env
    for variable in (
        "GEAK_ROOFLINE_COMPUTE_PATH",
        "HYPERLOOM_ROCPROF_COMPUTE_PATH",
        "ROCPROF_COMPUTE_PATH",
        "GEAK_ROCPROF_COMPUTE_PATH",
    ):
        candidate = environment.get(variable)
        if not candidate:
            continue
        candidate = os.path.expanduser(candidate)
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, "rocprof-compute")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate, variable
    candidate = shutil.which("rocprof-compute", path=environment.get("PATH"))
    if candidate:
        return candidate, "PATH"
    for candidate in (
        "/opt/rocm/bin/rocprof-compute",
        "/opt/rocm/libexec/rocprofiler-compute/rocprof-compute",
        "/opt/rocprofiler-compute/bin/rocprof-compute",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate, "opt"
    return None, None


def _text_output(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run(arguments, cwd=None, timeout_sec=DEFAULT_TIMEOUT_SEC, env=None):
    """Run an argument array and represent timeouts as exit code 124."""
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
            timeout=timeout_sec,
            env=env,
        )
        return completed.returncode, _text_output(completed.stdout), None
    except subprocess.TimeoutExpired as error:
        output = _text_output(error.stdout)
        if error.stderr:
            output += _text_output(error.stderr)
        warning = "command timed out after %s seconds" % timeout_sec
        if output and not output.endswith("\n"):
            output += "\n"
        return 124, output + warning + "\n", warning
    except OSError as error:
        warning = "command execution failed: %s" % error
        return 127, str(error) + "\n", warning


def _profile_output_option(tool, timeout_sec):
    """Choose the output option supported by the installed profiler."""
    code, output, warning = _run(
        [tool, "profile", "--help"], timeout_sec=timeout_sec
    )
    if code == 0 and "--output-directory" in output:
        return "--output-directory", warning
    return "--path", warning


def _unused_path(path):
    """Return a deterministic unused path without deleting prior artifacts."""
    if not os.path.exists(path):
        return path
    suffix = 2
    while os.path.exists("%s_%d" % (path, suffix)):
        suffix += 1
    return "%s_%d" % (path, suffix)


def _analysis_data_path(output_root):
    """Locate the actual workload/SoC directory produced by profile mode."""
    if not os.path.isdir(output_root):
        return output_root
    candidates = []
    for directory, _, filenames in os.walk(output_root):
        names = set(filenames)
        score = 0
        if "roofline.csv" in names:
            score += 4
        if "sysinfo.csv" in names:
            score += 2
        if any(name.startswith(("pmc_perf", "results_", "counter_collection"))
               for name in names):
            score += 1
        if score:
            depth = os.path.relpath(directory, output_root).count(os.sep)
            candidates.append((score, depth, directory))
    if not candidates:
        return output_root
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _literal_kernel_filters(patterns):
    """Return only patterns safe for rocprof-compute's substring filter."""
    regex_metacharacters = set(r".^$*+?{}[]\|()")
    if any(
        any(character in regex_metacharacters for character in pattern)
        for pattern in patterns
    ):
        return []
    return [pattern for pattern in patterns if pattern]


def _gpu_environment(manifest, case):
    """Constrain profile subprocesses to the manifest's selected GPU."""
    gpu_id = case.get("gpu_id")
    if gpu_id is None:
        gpu_id = manifest.get("target", {}).get("gpu_id")
    if gpu_id is None or str(gpu_id).strip() == "":
        return None
    environment = os.environ.copy()
    visible = str(gpu_id)
    environment["ROCR_VISIBLE_DEVICES"] = visible
    environment["HIP_VISIBLE_DEVICES"] = visible
    return environment


def _command_arguments(command):
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        if not command:
            raise ValueError("case command list must not be empty")
        return list(command), False
    if isinstance(command, str) and command.strip():
        return ["bash", "-lc", command], True
    raise ValueError("case command must be a non-empty list or string")


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    target = manifest.get("target", {})
    if not isinstance(target, dict) or not target.get("logical_name"):
        raise ValueError("manifest target.logical_name is required")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest cases must be a list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not case.get("case_id"):
            raise ValueError("every case requires case_id")
        if case["case_id"] in seen:
            raise ValueError("duplicate case_id: %s" % case["case_id"])
        seen.add(case["case_id"])
        _command_arguments(case.get("command"))


def _patterns(manifest, case):
    value = case.get("kernel_patterns")
    if value is None:
        value = manifest.get("target", {}).get("kernel_patterns", [])
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value or [] if str(item)]


def _select_kernel(kernels, patterns):
    if not kernels:
        return None
    for pattern in patterns:
        try:
            expression = re.compile(pattern)
        except re.error:
            expression = re.compile(re.escape(pattern))
        for kernel in kernels:
            if expression.search(kernel["kernel_name"]):
                return kernel
    return None if patterns else kernels[0]


def _tool_version(tool, timeout_sec):
    code, output, warning = _run(
        [tool, "--version"], timeout_sec=timeout_sec
    )
    line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    return {
        "exit_code": code,
        "text": line,
        "timed_out": code == 124,
        "warning": warning,
    }


def _has_valid_kernel_metrics(kernel):
    if not kernel:
        return False
    metrics = kernel.get("metrics", {})
    observed = (
        "performance_gflops",
        "compute_actual_gflops",
        "hbm_actual_gbps",
    )
    return any(_finite_number(metrics.get(name)) is not None for name in observed)


def _case_result(
    manifest,
    case,
    tool,
    phase,
    out_dir,
    timeout_sec=DEFAULT_TIMEOUT_SEC,
    saturation_pct=roofline_policy.DEFAULT_SATURATION_PCT,
):
    case_id = str(case["case_id"])
    case_dir = os.path.join(out_dir, "cases", case_id)
    os.makedirs(case_dir, exist_ok=True)
    command, wrapped = _command_arguments(case["command"])
    patterns = _patterns(manifest, case)
    profile_environment = _gpu_environment(manifest, case)
    profile_output = _unused_path(os.path.join(case_dir, "profile_data"))
    output_option, option_warning = _profile_output_option(tool, timeout_sec)
    profile_args = [
        tool, "profile", "--roof-only", "-n", case_id,
    ]
    literal_filters = _literal_kernel_filters(patterns)
    if literal_filters:
        profile_args.extend(["-k"] + literal_filters)
    profile_args.extend([output_option, profile_output, "--"])
    profile_args.extend(command)
    profile_code, profile_text, profile_warning = _run(
        profile_args,
        cwd=case.get("workdir"),
        timeout_sec=timeout_sec,
        env=profile_environment,
    )
    profile_log = os.path.join(case_dir, "profile.log")
    _write_text(profile_log, profile_text)

    analysis_path = _analysis_data_path(profile_output)
    analyze_args = [tool, "analyze", "-p", analysis_path, "-b", "2", "4", "17"]
    analyze_code, analyze_text, analyze_warning = _run(
        analyze_args,
        cwd=case.get("workdir"),
        timeout_sec=timeout_sec,
        env=profile_environment,
    )
    analyze_log = os.path.join(case_dir, "analyze.txt")
    _write_text(analyze_log, analyze_text)
    kernels = parse_rocprof_compute(
        analyze_text,
        dtypes=case.get("dtypes"),
        saturation_pct=saturation_pct,
    )
    selected = _select_kernel(kernels, patterns)
    if not _has_valid_kernel_metrics(selected):
        selected = None
    warnings = []
    if option_warning:
        warnings.append(option_warning)
    if patterns and not literal_filters:
        warnings.append(
            "regex kernel_patterns were applied after collection because profile -k "
            "uses substring matching"
        )
    if profile_code != 0:
        warnings.append(
            "profile exited %d; analyze was still attempted because artifacts may be valid"
            % profile_code
        )
    if profile_warning:
        warnings.append(profile_warning)
    if analyze_code != 0:
        warnings.append("analyze exited %d" % analyze_code)
    if analyze_warning:
        warnings.append(analyze_warning)
    if not selected:
        if patterns and kernels:
            warnings.append("no analyzed kernel matched the requested kernel_patterns")
        else:
            warnings.append("analyze output contained no valid kernel roofline metrics")

    metrics = selected["metrics"] if selected else {}
    classification = (
        roofline_policy.build_classification(
            metrics, saturation_pct=saturation_pct
        )
        if selected
        else roofline_policy.build_classification(
            {}, saturation_pct=saturation_pct
        )
    )
    status = "matched" if selected is not None else "failed"
    return {
        "case_id": case_id,
        "phase": phase,
        "status": status,
        "shape": case.get("shape"),
        "dtypes": case.get("dtypes") or [],
        "regime": case.get("regime"),
        "weight": case.get("weight", 1.0),
        "workdir": case.get("workdir"),
        "command": case.get("command"),
        "command_wrapped_with_bash_lc": wrapped,
        "kernel_patterns": patterns,
        "kernel": selected["kernel_name"] if selected else None,
        "matched_kernel_name": selected["kernel_name"] if selected else None,
        "peak_basis": metrics.get("peak_basis"),
        "compute_metric": metrics.get("compute_metric"),
        "metrics": metrics,
        "classification": classification,
        "kernels": kernels,
        "profile_exit_code": profile_code,
        "analyze_exit_code": analyze_code,
        "profile_timed_out": profile_code == 124,
        "analyze_timed_out": analyze_code == 124,
        "profile_arguments": profile_args,
        "analyze_arguments": analyze_args,
        "warnings": warnings,
        "artifacts": {
            "case_dir": case_dir,
            "profile_data": analysis_path,
            "profile_log": profile_log,
            "analyze_text": analyze_log,
        },
    }


def _summary_classification(classification):
    return {
        "theoretical_bound": classification.get(
            "theoretical_bound", "unknown"
        ),
        "observed_limit": classification.get("observed_limit", "unknown"),
        "recommended_specialties": classification.get(
            "recommended_specialties", []
        ) or [],
        "recommended_levers": classification.get(
            "recommended_levers", []
        ) or [],
        "confidence": classification.get("confidence", "low"),
        "evidence": classification.get("evidence", []) or [],
    }


def build_summary(
    cases, saturation_pct=roofline_policy.DEFAULT_SATURATION_PCT
):
    """Build routing and weighted-headroom summary without averaging AI."""
    ranked = []
    case_routes = []
    for case in cases:
        weight = _finite_number(case.get("weight"))
        if weight is None:
            weight = 1.0
        headroom = _finite_number(case.get("metrics", {}).get("headroom_ratio"))
        if case.get("status") != "matched":
            priority = 0.0
            reason = "case_not_matched"
        elif headroom is None:
            priority = 0.0
            reason = "missing_empirical_headroom"
        else:
            priority = weight * headroom
            reason = "weight_times_empirical_headroom"
        ranked.append(
            {
                "case_id": case.get("case_id"),
                "priority": priority,
                "score": priority,
                "weight": weight,
                "headroom_ratio": headroom,
                "reason": reason,
            }
        )
        case_routes.append(
            {
                "case_id": case.get("case_id"),
                "status": case.get("status"),
                "matched_kernel_name": case.get(
                    "matched_kernel_name", case.get("kernel")
                ),
                "recommended_specialties": case.get(
                    "classification", {}
                ).get("recommended_specialties", []) or [],
            }
        )
    ranked.sort(key=lambda item: (-item["priority"], str(item["case_id"])))
    specialties = []
    for rank in ranked:
        matching = next(
            (case for case in cases if case.get("case_id") == rank["case_id"]), {}
        )
        if matching.get("status") != "matched":
            continue
        for specialty in matching.get("classification", {}).get(
            "recommended_specialties", []
        ):
            if specialty not in specialties:
                specialties.append(specialty)
    statuses = {}
    for case in cases:
        status = case.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    cases_by_id = {case.get("case_id"): case for case in cases}
    dominant_case = next(
        (
            cases_by_id[rank["case_id"]]
            for rank in ranked
            if cases_by_id[rank["case_id"]].get("status") == "matched"
        ),
        None,
    )
    if dominant_case is None:
        dominant_classification = roofline_policy.build_classification(
            {}, saturation_pct=saturation_pct
        )
    else:
        dominant_classification = dominant_case.get("classification", {})
    return {
        "case_count": len(cases),
        "status_counts": statuses,
        "case_routes": case_routes,
        "priority_order": ranked,
        "recommended_specialties": specialties,
        "dominant_case_id": (
            dominant_case.get("case_id") if dominant_case is not None else None
        ),
        "dominant_classification": _summary_classification(
            dominant_classification
        ),
        "note": "Priority is weight * headroom_ratio; arithmetic intensity is not averaged.",
    }


def collect_manifest(
    manifest_path,
    phase,
    out_dir,
    timeout_sec=DEFAULT_TIMEOUT_SEC,
    saturation_pct=roofline_policy.DEFAULT_SATURATION_PCT,
):
    """Execute all manifest cases and atomically write a roofline JSON report."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_manifest(manifest)
    timeout_value = _finite_number(timeout_sec)
    if timeout_value is None or timeout_value <= 0:
        raise ValueError("timeout_sec must be a positive finite number")
    saturation_value = _finite_number(saturation_pct)
    if saturation_value is None or saturation_value < 0:
        raise ValueError("saturation_pct must be a non-negative finite number")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "%s_roofline.json" % phase)
    tool = None
    tool_source = None
    base = {
        "schema_version": SCHEMA_VERSION,
        "source": "rocprof-compute",
        "phase": phase,
        "status": "failed",
        "reason": "",
        "tool": {
            "path": tool,
            "source": tool_source,
            "version": None,
        },
        "tool_version": "",
        "policy": {
            "version": roofline_policy.POLICY_VERSION,
            "saturation_pct": saturation_value,
        },
        "policy_version": roofline_policy.POLICY_VERSION,
        "target": manifest.get("target"),
        "cases": [],
        "dominant_case_id": None,
        "summary": {},
        "json_path": json_path,
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "warnings": [],
    }
    if not manifest["cases"]:
        base["status"] = "skipped"
        base["reason"] = "no_profile_cases"
        base["summary"] = build_summary(
            [], saturation_pct=saturation_value
        )
        _atomic_json(json_path, base)
        return base

    tool, tool_source = locate_rocprof_compute()
    base["tool"]["path"] = tool
    base["tool"]["source"] = tool_source
    if tool is None:
        base["status"] = "skipped"
        base["reason"] = "rocprof_compute_unavailable"
        base["warnings"].append(
            "rocprof-compute was not found in overrides, PATH, or common /opt locations"
        )
        base["summary"] = build_summary(
            [], saturation_pct=saturation_value
        )
        _atomic_json(json_path, base)
        return base

    version = _tool_version(tool, timeout_value)
    base["tool"]["version"] = version
    base["tool_version"] = version["text"]
    if version["warning"]:
        base["warnings"].append(version["warning"])
    for case in manifest["cases"]:
        base["cases"].append(
            _case_result(
                manifest,
                case,
                tool,
                phase,
                out_dir,
                timeout_sec=timeout_value,
                saturation_pct=saturation_value,
            )
        )
    base["summary"] = build_summary(
        base["cases"], saturation_pct=saturation_value
    )
    base["dominant_case_id"] = base["summary"]["dominant_case_id"]
    statuses = [case["status"] for case in base["cases"]]
    matched = [case for case in base["cases"] if case["status"] == "matched"]
    commands_clean = all(
        case["profile_exit_code"] == 0 and case["analyze_exit_code"] == 0
        for case in matched
    )
    if statuses and all(status == "matched" for status in statuses) and commands_clean:
        base["status"] = "ok"
    elif matched:
        base["status"] = "partial"
    elif statuses and all(status == "skipped" for status in statuses):
        base["status"] = "skipped"
    else:
        base["status"] = "failed"
    _atomic_json(json_path, base)
    return base


def _load_report(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("cannot read roofline report %s: %s" % (path, error))
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
        raise ValueError("roofline report %s has no cases list" % path)
    return report


def _report_policy_identity(report):
    policy = report.get("policy")
    if not isinstance(policy, dict):
        policy = {}
    return {
        "version": policy.get("version"),
        "saturation_pct": policy.get("saturation_pct"),
    }


def _target_device_identity(report):
    target = report.get("target")
    if not isinstance(target, dict):
        target = {}
    identity = {}
    keys = (
        "device",
        "device_id",
        "device_identity",
        "device_name",
        "device_uuid",
        "gpu",
        "gpu_id",
        "gpu_identity",
        "gpu_index",
        "gpu_name",
        "gpu_uuid",
        "gpu_arch",
        "arch",
        "architecture",
        "gfx",
        "gfx_arch",
        "soc",
    )
    for key in keys:
        if key in target:
            identity[key] = target[key]
        elif key in report:
            identity[key] = report[key]
        else:
            identity[key] = None
    return identity


def compare_reports(before_path, after_path):
    """Compare complete reports by case_id with strict per-case compatibility."""
    before = _load_report(before_path)
    after = _load_report(after_path)
    before_policy = _report_policy_identity(before)
    after_policy = _report_policy_identity(after)
    if before_policy != after_policy:
        raise ValueError(
            "policy differs: before=%r after=%r"
            % (before_policy, after_policy)
        )
    before_target = _target_device_identity(before)
    after_target = _target_device_identity(after)
    if before_target != after_target:
        raise ValueError(
            "target device/GPU identity differs: before=%r after=%r"
            % (before_target, after_target)
        )
    if any(case.get("status") != "matched" for case in before["cases"]):
        raise ValueError("before report contains a case that is not matched")
    if any(case.get("status") != "matched" for case in after["cases"]):
        raise ValueError("after report contains a case that is not matched")
    before_cases = {case.get("case_id"): case for case in before["cases"]}
    after_cases = {case.get("case_id"): case for case in after["cases"]}
    if len(before_cases) != len(before["cases"]) or len(after_cases) != len(after["cases"]):
        raise ValueError("reports contain duplicate case_id values")
    if set(before_cases) != set(after_cases):
        raise ValueError(
            "case_id sets differ: before=%r after=%r"
            % (sorted(before_cases), sorted(after_cases))
        )
    comparisons = [
        roofline_policy.compare_cases(before_cases[case_id], after_cases[case_id])
        for case_id in sorted(before_cases)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "roofline-compare",
        "status": "ok",
        "before": os.path.abspath(before_path),
        "after": os.path.abspath(after_path),
        "policy": before_policy,
        "target_identity": before_target,
        "cases": comparisons,
        "improved_case_count": sum(item["improved"] is True for item in comparisons),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    collect = subparsers.add_parser("collect", help="profile manifest cases")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--phase", required=True)
    collect.add_argument("--out-dir", required=True)
    collect.add_argument(
        "--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC
    )
    collect.add_argument(
        "--saturation-pct",
        type=float,
        default=roofline_policy.DEFAULT_SATURATION_PCT,
    )
    compare = subparsers.add_parser("compare", help="compare two collection reports")
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--out", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        if arguments.subcommand == "collect":
            result = collect_manifest(
                arguments.manifest,
                arguments.phase,
                arguments.out_dir,
                timeout_sec=arguments.timeout_sec,
                saturation_pct=arguments.saturation_pct,
            )
        else:
            result = compare_reports(arguments.before, arguments.after)
            _atomic_json(arguments.out, result)
    except ValueError as error:
        print("roofline_kernel: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
