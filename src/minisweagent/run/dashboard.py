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

import logging
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


def _truncate_text(text: str, max_len: int = 56) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


class _DashboardLive(Live):
    """Live subclass that re-renders from the Dashboard on every tick."""

    def __init__(self, dashboard, **kwargs):
        self._dashboard = dashboard
        super().__init__("", **kwargs)

    def get_renderable(self):
        return self._dashboard._render()


class Dashboard:
    def __init__(
        self,
        num_gpus: int = 8,
        max_rounds: int = 2,
        kernel_name: str = "",
        gpu_ids: list[int] | None = None,
    ):
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else list(range(num_gpus))
        self.num_gpus = len(self.gpu_ids)
        self.max_rounds = max_rounds
        self.kernel_name = kernel_name
        self._lock = Lock()

        self.current_round = 0
        self.phase = "idle"
        self.best_speedup = 1.0
        self.best_round = 0
        self.checklist_order = [
            "resolve",
            "discovery",
            "harness",
            "profile",
            "baseline",
            "commandment",
            "round",
            "dispatch",
            "evaluation",
            "finalize",
        ]
        self.checklist = {
            "resolve": {"label": "URL resolved", "status": "pending", "detail": ""},
            "discovery": {"label": "Generating tests", "status": "pending", "detail": ""},
            "harness": {"label": "Test harness", "status": "pending", "detail": ""},
            "profile": {"label": "Profiling context", "status": "pending", "detail": ""},
            "baseline": {"label": "Baseline metrics", "status": "pending", "detail": ""},
            "commandment": {"label": "Commandment", "status": "pending", "detail": ""},
            "round": {"label": "Round loop", "status": "pending", "detail": ""},
            "dispatch": {"label": "Dispatching workers", "status": "pending", "detail": ""},
            "evaluation": {"label": "Evaluation", "status": "pending", "detail": ""},
            "finalize": {"label": "Final report", "status": "pending", "detail": ""},
        }
        self.kernel_signals: list[str] = []
        self.method_plan: list[str] = []
        self.tried_methods = deque(maxlen=6)
        self.gpu_status = {}
        for gpu_id in self.gpu_ids:
            self.gpu_status[gpu_id] = {
                "status": "idle",
                "label": "",
                "elapsed": 0,
                "stage": "",
                "step": None,
                "tool": "",
                "intent": "",
                "result": "",
                "history": deque(maxlen=3),
            }
        self.logs = deque(maxlen=12)
        self.round_results = {}
        self._start_time = time.time()
        self._live = None

    def start(self):
        self._live = _DashboardLive(self, refresh_per_second=2, console=Console())
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
            Layout(name="left", ratio=1),
            Layout(name="workers", ratio=2),
        )
        layout["left"].split_column(
            Layout(name="status", size=8),
            Layout(name="checklist", size=13),
            Layout(name="insights"),
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

        # Checklist
        checklist_table = Table(show_header=False, expand=True, padding=(0, 1))
        checklist_table.add_column("State", width=5, style="bold")
        checklist_table.add_column("Step", style="bold")
        checklist_table.add_column("Detail")
        state_map = {
            "pending": Text("...", style="dim"),
            "active": Text("NOW", style="yellow"),
            "done": Text("OK", style="green"),
            "error": Text("ERR", style="red"),
        }
        for key in self.checklist_order:
            item = self.checklist[key]
            state = state_map.get(item["status"], Text(item["status"], style="white"))
            detail = item.get("detail", "")
            checklist_table.add_row(state, item["label"], _truncate_text(detail, max_len=32) if detail else "")
        layout["checklist"].update(Panel(checklist_table, title="Checklist"))

        insights_table = Table.grid(expand=True, padding=(0, 1))
        insights_table.add_column("Key", style="bold", width=8)
        insights_table.add_column("Value")
        signal_lines = self.kernel_signals or ["No bottleneck summary yet"]
        current_methods = []
        for gpu_id in sorted(self.gpu_status.keys()):
            info = self.gpu_status[gpu_id]
            if info.get("status") == "running":
                trying = info.get("intent") or info.get("label") or ""
                if trying and trying not in current_methods:
                    current_methods.append(trying)
        insights_table.add_row("Signals", "\n".join(f"- {_truncate_text(line, 44)}" for line in signal_lines[:4]))
        insights_table.add_row(
            "Current",
            "\n".join(f"- {_truncate_text(line, 44)}" for line in current_methods[:3]) if current_methods else "- none",
        )
        insights_table.add_row(
            "Tried",
            "\n".join(f"- {_truncate_text(line, 44)}" for line in list(self.tried_methods)[:4]) if self.tried_methods else "- none",
        )
        insights_table.add_row(
            "List",
            "\n".join(f"- {_truncate_text(line, 44)}" for line in self.method_plan[:4]) if self.method_plan else "- none",
        )
        layout["insights"].update(Panel(insights_table, title="Kernel Signals"))

        # Worker cards
        worker_cards = []
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

            worker_table = Table.grid(expand=True, padding=(0, 1))
            worker_table.add_column("Key", style="bold", width=7)
            worker_table.add_column("Value")
            worker_table.add_row("Status", Text(status, style=style))
            if info.get("stage"):
                worker_table.add_row("Stage", info["stage"])
            trying_now = info.get("intent") or info.get("label") or "-"
            worker_table.add_row("Trying", _truncate_text(trying_now, max_len=54))
            if info.get("step") is not None:
                worker_table.add_row("Step", str(info["step"]))
            if info.get("tool"):
                worker_table.add_row("Tool", _truncate_text(info["tool"], max_len=22))
            if info.get("result"):
                worker_table.add_row("Last", _truncate_text(info["result"], max_len=54))
            history = list(info.get("history") or [])
            if history:
                worker_table.add_row("Recent", "\n".join(history[-2:]))

            worker_cards.append(
                Panel(
                    worker_table,
                    title=f"GPU {gpu_id}",
                    border_style=style,
                )
            )

        worker_grid = Table.grid(expand=True)
        worker_grid.add_column(ratio=1)
        worker_grid.add_column(ratio=1)
        if worker_cards:
            for idx in range(0, len(worker_cards), 2):
                left_card = worker_cards[idx]
                right_card = worker_cards[idx + 1] if idx + 1 < len(worker_cards) else Text("")
                worker_grid.add_row(left_card, right_card)
        layout["workers"].update(Panel(worker_grid if worker_cards else "(no workers)", title="Workers"))

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
            self._set_checklist_item("round", "active", f"round {round_num}/{self.max_rounds}")
            self._set_checklist_item("dispatch", "pending", "")
            self._set_checklist_item("evaluation", "pending", "")
            for gpu_id in self.gpu_status:
                self.gpu_status[gpu_id]["status"] = "idle"
                self.gpu_status[gpu_id]["label"] = ""
                self.gpu_status[gpu_id]["stage"] = ""
                self.gpu_status[gpu_id]["step"] = None
                self.gpu_status[gpu_id]["tool"] = ""
                self.gpu_status[gpu_id]["intent"] = ""
                self.gpu_status[gpu_id]["result"] = ""
                self.gpu_status[gpu_id]["history"].clear()
            self.log(f"Starting round {round_num}", _already_locked=True)

    def update_phase(self, phase: str):
        with self._lock:
            self.phase = phase
            self._refresh()

    def _set_checklist_item(self, key: str, status: str, detail: str = ""):
        if key not in self.checklist:
            return
        item = self.checklist[key]
        item["status"] = status
        item["detail"] = detail

    def update_kernel_signals(self, signals: list[str]):
        with self._lock:
            self.kernel_signals = [s for s in signals if s][:6]
            self._refresh()

    def set_method_plan(self, methods: list[str]):
        with self._lock:
            self.method_plan = [m for m in methods if m][:12]
            self._refresh()

    def record_method_attempt(self, method: str, outcome: str = ""):
        with self._lock:
            if not method:
                return
            entry = method if not outcome else f"{method} -> {outcome}"
            entry = _truncate_text(entry, max_len=54)
            if entry in self.tried_methods:
                return
            self.tried_methods.appendleft(entry)
            self._refresh()

    def update_checklist_item(self, key: str, status: str, detail: str = ""):
        with self._lock:
            self._set_checklist_item(key, status, detail)
            self._refresh()

    def update_task(self, gpu_id: int, status: str, label: str = ""):
        with self._lock:
            if gpu_id in self.gpu_status:
                info = self.gpu_status[gpu_id]
                info["status"] = status
                info["label"] = label
                if status == "running":
                    info["stage"] = "starting"
                    info["step"] = None
                    info["tool"] = ""
                    info["intent"] = ""
                    info["result"] = ""
                    info["history"].clear()
                elif status == "done":
                    info["stage"] = "completed"
                elif status == "error":
                    info["stage"] = "failed"
            self._refresh()

    def update_worker_message(
        self,
        gpu_id: int,
        *,
        stage: str | None = None,
        step: int | None = None,
        tool_name: str = "",
        intent: str | None = None,
        result: str | None = None,
        history_line: str | None = None,
    ):
        with self._lock:
            if gpu_id not in self.gpu_status:
                return
            info = self.gpu_status[gpu_id]
            if stage:
                info["stage"] = stage
            if step is not None:
                if info.get("step") != step:
                    info["result"] = ""
                info["step"] = step
            if tool_name:
                info["tool"] = tool_name
            if intent:
                info["intent"] = intent
            if result:
                info["result"] = result
            if history_line:
                line = _truncate_text(history_line, max_len=42)
                history = info["history"]
                if not history or history[-1] != line:
                    history.append(line)
            self._refresh()

    def update_speedup(self, speedup: float, round_num: int = None):
        with self._lock:
            rnd = round_num if round_num is not None else self.current_round
            self.round_results[rnd] = {"speedup": f"{speedup:.3f}"}
            if speedup > self.best_speedup:
                self.best_speedup = speedup
                self.best_round = rnd
                self.log(f"New best: {speedup:.3f}x (round {rnd})", _already_locked=True)
            else:
                self._refresh()

    def log(self, msg: str, _already_locked: bool = False):
        ts = time.strftime("%H:%M:%S")
        if _already_locked:
            formatted = f"[{ts}] {msg}"
            if not self.logs or self.logs[-1] != formatted:
                self.logs.append(formatted)
            self._refresh()
            return
        with self._lock:
            formatted = f"[{ts}] {msg}"
            if not self.logs or self.logs[-1] != formatted:
                self.logs.append(formatted)
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
            except Exception as exc:
                logging.getLogger(__name__).debug("GEAK_DASH marker parse error: %s", exc)
            return ""

        return MARKER_RE.sub(_handle_marker, text)

    def parse_log_line(self, line: str):
        """Parse orchestrator log lines for status updates.

        Also checks for GEAK_DASH markers embedded in the line.
        """
        marker_match = MARKER_RE.search(line)
        if marker_match:
            self.parse_llm_output(line)
            return

        line_stripped = re.sub(r"\[/?[a-z_ ]+\]", "", line).strip()
        line_lower = line_stripped.lower()

        # Round start: "--- Homogeneous round 1/2 ---" or "round N"
        m = re.search(r"round\s+(\d+)\s*/\s*(\d+)", line_lower)
        if m:
            self.update_round(int(m.group(1)))
            return

        # Dispatching
        if "dispatching" in line_lower or "run_task_batch" in line_lower:
            self.update_phase("dispatching")
            self.log(line_stripped[:70])
            return

        # Round best result: "Round 1 best: kernel_optimization (1.08x)"
        m = re.search(r"round\s+(\d+)\s+best:.*?(\d+\.?\d*)\s*x", line_lower)
        if m:
            self.update_speedup(float(m.group(2)), int(m.group(1)))
            self.update_phase("evaluated")
            self.log(line_stripped[:70])
            return

        # FULL_BENCHMARK
        if "full_benchmark" in line_lower:
            if "running" in line_lower:
                self.update_phase("full benchmark")
            self.log(line_stripped[:70])
            return

        # PROFILE
        if "profile" in line_lower and ("running" in line_lower or "comparison" in line_lower):
            self.update_phase("profiling")
            self.log(line_stripped[:70])
            return

        # Early stopping
        if "early stop" in line_lower:
            self.update_phase("early stopped")
            self.log(line_stripped[:70])
            return

        # Finalize
        if "finalize" in line_lower or "auto-finalize" in line_lower:
            self.update_phase("finalizing")
            self.log(line_stripped[:70])
            return

        # Task file
        if "task file:" in line_lower:
            self.log(line_stripped[:70])
            return

        # Dispatch / build / apply failures
        if "dispatch failed" in line_lower or "build failed" in line_lower or "patch apply failed" in line_lower:
            self.log(f"⚠ {line_stripped[:65]}")
            return

        # Generic speedup mention
        m = re.search(r"(\d+\.\d+)\s*x", line)
        if m and "speedup" in line_lower:
            self.update_speedup(float(m.group(1)))
            return


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
