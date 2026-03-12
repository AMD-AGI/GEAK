"""Simple Rich-based dashboard for GEAK orchestrator.

Provides a live-updating terminal display showing:
- Current round and phase
- Active GPU tasks  
- Best speedup so far
- Recent log lines

Usage:
    dashboard = Dashboard(num_gpus=8, max_rounds=2)
    dashboard.start()
    dashboard.update_round(1)
    dashboard.update_task(gpu_id=0, status="running", label="topk_opt")
    dashboard.update_speedup(1.11)
    dashboard.log("Correctness passed")
    dashboard.stop()

Can also parse LLM-emitted markers:
    <!-- GEAK_DASH: {"event": "task_start", "gpu": 0, "label": "optimize"} -->
"""

import re
import time
from collections import deque
from threading import Lock

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


MARKER_RE = re.compile(r"<!--\s*GEAK_DASH:\s*(\{.*?\})\s*-->")


class Dashboard:
    def __init__(self, num_gpus: int = 8, max_rounds: int = 2, kernel_name: str = ""):
        self.num_gpus = num_gpus
        self.max_rounds = max_rounds
        self.kernel_name = kernel_name
        self._lock = Lock()

        self.current_round = 0
        self.phase = "idle"
        self.best_speedup = 1.0
        self.best_round = 0
        self.gpu_status = {}
        for i in range(num_gpus):
            self.gpu_status[i] = {"status": "idle", "label": "", "elapsed": 0}
        self.logs = deque(maxlen=12)
        self.round_results = {}
        self._start_time = time.time()
        self._live = None

    def start(self):
        self._live = Live(self._render(), refresh_per_second=2, console=Console())
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()

    def _render(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="logs", size=14),
        )
        layout["body"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="gpus", ratio=1),
        )

        elapsed = time.time() - self._start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        header_text = Text.assemble(
            ("GEAK Dashboard", "bold cyan"),
            f"  |  {self.kernel_name}" if self.kernel_name else "",
            f"  |  {elapsed_str}",
        )
        layout["header"].update(Panel(header_text, style="bold"))

        # Status box
        status_table = Table(show_header=False, expand=True, padding=(0, 1))
        status_table.add_column("Key", style="bold")
        status_table.add_column("Value")
        status_table.add_row("Round", f"{self.current_round}/{self.max_rounds}")
        status_table.add_row("Phase", self.phase)
        speedup_style = "green" if self.best_speedup > 1.0 else "white"
        status_table.add_row(
            "Best Speedup",
            Text(f"{self.best_speedup:.3f}x (round {self.best_round})", style=speedup_style),
        )
        for rnd, result in sorted(self.round_results.items()):
            spd = result.get("speedup", "?")
            status_table.add_row(f"Round {rnd}", f"{spd}x")
        layout["status"].update(Panel(status_table, title="Status"))

        # GPU box
        gpu_table = Table(expand=True, padding=(0, 1))
        gpu_table.add_column("GPU", style="bold", width=4)
        gpu_table.add_column("Status", width=10)
        gpu_table.add_column("Task")
        for gpu_id in sorted(self.gpu_status.keys()):
            info = self.gpu_status[gpu_id]
            status = info["status"]
            if status == "running":
                style = "yellow"
            elif status == "done":
                style = "green"
            elif status == "error":
                style = "red"
            else:
                style = "dim"
            gpu_table.add_row(
                str(gpu_id),
                Text(status, style=style),
                info.get("label", ""),
            )
        layout["gpus"].update(Panel(gpu_table, title="GPUs"))

        # Logs box
        log_lines = list(self.logs)
        log_text = "\n".join(log_lines) if log_lines else "(no logs yet)"
        layout["logs"].update(Panel(log_text, title="Recent Activity"))

        return layout

    def _refresh(self):
        if self._live:
            self._live.update(self._render())

    def update_round(self, round_num: int):
        with self._lock:
            self.current_round = round_num
            self.phase = "generating tasks"
            self.log(f"Starting round {round_num}")
            self._refresh()

    def update_phase(self, phase: str):
        with self._lock:
            self.phase = phase
            self._refresh()

    def update_task(self, gpu_id: int, status: str, label: str = ""):
        with self._lock:
            if gpu_id in self.gpu_status:
                self.gpu_status[gpu_id]["status"] = status
                self.gpu_status[gpu_id]["label"] = label
            self._refresh()

    def update_speedup(self, speedup: float, round_num: int = None):
        with self._lock:
            rnd = round_num or self.current_round
            self.round_results[rnd] = {"speedup": f"{speedup:.3f}"}
            if speedup > self.best_speedup:
                self.best_speedup = speedup
                self.best_round = rnd
                self.log(f"New best: {speedup:.3f}x (round {rnd})")
            self._refresh()

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if self._live:
            self._refresh()

    def parse_llm_output(self, text: str) -> str:
        """Extract GEAK_DASH markers from LLM output and update dashboard.
        Returns text with markers stripped."""
        import json

        def _handle_marker(match):
            try:
                data = json.loads(match.group(1))
                event = data.get("event", "")
                if event == "task_start":
                    self.update_task(data.get("gpu", 0), "running", data.get("label", ""))
                elif event == "task_done":
                    self.update_task(data.get("gpu", 0), "done", data.get("label", ""))
                    if "speedup" in data:
                        self.update_speedup(data["speedup"])
                elif event == "task_error":
                    self.update_task(data.get("gpu", 0), "error", data.get("label", ""))
                elif event == "round_start":
                    self.update_round(data.get("round", self.current_round + 1))
                elif event == "phase":
                    self.update_phase(data.get("phase", ""))
                elif event == "log":
                    self.log(data.get("msg", ""))
                elif event == "speedup":
                    self.update_speedup(data.get("value", 1.0), data.get("round"))
            except Exception:
                pass
            return ""

        return MARKER_RE.sub(_handle_marker, text)

    def parse_log_line(self, line: str):
        """Parse orchestrator log lines for status updates."""
        line_lower = line.lower()
        if "round" in line_lower and ("starting" in line_lower or "begin" in line_lower):
            m = re.search(r"round\s*(\d+)", line_lower)
            if m:
                self.update_round(int(m.group(1)))
        elif "dispatch" in line_lower:
            self.update_phase("dispatching")
        elif "speedup" in line_lower:
            m = re.search(r"(\d+\.?\d*)\s*x", line)
            if m:
                self.update_speedup(float(m.group(1)))
        elif "correctness" in line_lower and "pass" in line_lower:
            self.log("Correctness passed")
        elif "error" in line_lower or "fail" in line_lower:
            self.log(f"⚠ {line.strip()[:60]}")
        elif "finalize" in line_lower:
            self.update_phase("finalizing")
            self.log("Finalizing...")


def demo():
    """Quick demo of the dashboard."""
    import time as _time

    dash = Dashboard(num_gpus=8, max_rounds=2, kernel_name="topk")
    dash.start()
    try:
        _time.sleep(1)
        dash.update_round(1)
        _time.sleep(0.5)
        dash.update_phase("dispatching")
        for i in range(8):
            dash.update_task(i, "running", "kernel_optimization")
            _time.sleep(0.2)
        _time.sleep(1)
        for i in range(8):
            dash.update_task(i, "done", "kernel_optimization")
            _time.sleep(0.1)
        dash.update_phase("evaluating")
        dash.update_speedup(1.08, 1)
        _time.sleep(1)
        dash.update_round(2)
        dash.update_phase("dispatching")
        for i in range(8):
            dash.update_task(i, "running", "kernel_optimization")
        _time.sleep(2)
        for i in range(8):
            dash.update_task(i, "done")
        dash.update_speedup(1.11, 2)
        dash.update_phase("done")
        dash.log("Optimization complete!")
        _time.sleep(2)
    finally:
        dash.stop()


if __name__ == "__main__":
    demo()
