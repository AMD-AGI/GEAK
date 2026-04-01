#!/usr/bin/env python3
"""GEAK CLI — GPU kernel optimization via REST API.

Usage:
    python geak.py config   --model_name NAME --api_base URL --api_key KEY [--model_class CLASS]
    python geak.py optimize FILE [FILE2 ...] [--prompt TEXT] [--step_limit N] [--gpu_count N]
    python geak.py optimize-repo REPO_URL [--branch BRANCH] [--prompt TEXT] [--step_limit N]
    python geak.py status   TASK_ID
    python geak.py results  TASK_ID [--output_dir DIR]
    python geak.py list     [--status STATUS]
    python geak.py cancel   TASK_ID

Environment variables:
    GEAK_API_URL   GEAK server URL  (required)
    GEAK_API_KEY   SaFE API key     (required)
"""

import argparse
import json
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

# ── Configuration ─────────────────────────────────────────────────────

API_URL = os.environ.get("GEAK_API_URL", "").rstrip("/")
API_KEY = os.environ.get("GEAK_API_KEY", "")


def _check_env():
    if not API_URL:
        print("Error: GEAK_API_URL not set. Export it or add to .env file.")
        sys.exit(1)
    if not API_KEY:
        print("Error: GEAK_API_KEY not set. Export it or add to .env file.")
        sys.exit(1)


def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _api(method, path, **kwargs):
    url = f"{API_URL}/api/v1{path}"
    resp = requests.request(method, url, headers=_headers(), **kwargs)
    if resp.status_code == 204:
        return {"success": True}
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    if resp.status_code >= 400:
        print(f"Error ({resp.status_code}): {data.get('detail', data)}")
        sys.exit(1)
    return data


def _print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ── Commands ──────────────────────────────────────────────────────────

def cmd_config(args):
    """Configure LLM model (saved per-user, one-time setup)."""
    payload = {
        "model_class": args.model_class,
        "model_name": args.model_name,
        "api_key": args.api_key,
        "model_kwargs": {
            "api_base": args.api_base,
            "temperature": 0.0,
            "max_tokens": 16000,
        },
    }
    result = _api("PUT", "/config/model", json=payload)
    print("Model configured:")
    _print_json(result)


def cmd_optimize(args):
    """Optimize HIP kernel file(s) (create + submit + poll + results)."""
    # Read all files
    files = []
    for filepath in args.files:
        if not os.path.isfile(filepath):
            print(f"Error: file not found: {filepath}")
            sys.exit(1)
        filename = os.path.basename(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        files.append({"filename": filename, "content": content})

    names = ", ".join(f["filename"] for f in files)
    print(f"Creating task for {names} ...")

    # Create task
    payload = {
        "input_type": "file",
        "files": files,
    }
    if args.prompt:
        payload["prompt"] = args.prompt
    if args.step_limit:
        payload["config"] = {"agent": {"step_limit": args.step_limit}}
    if args.gpu_count:
        payload["runtime"] = {"gpu_count": args.gpu_count}

    task = _api("POST", "/tasks", json=payload)
    task_id = task["id"]
    print(f"Task created: {task_id}")

    # Submit
    _api("POST", f"/tasks/{task_id}/submit")
    print(f"Task submitted. Running on GPU ...")

    # Poll
    _poll_until_done(task_id)

    # Results
    _download_results(task_id, args.output_dir)


def cmd_optimize_repo(args):
    """Optimize HIP kernels in a git repository."""
    print(f"Creating task for repo: {args.repo_url} ...")

    payload = {
        "input_type": "repo",
        "repo": {"url": args.repo_url},
    }
    if args.branch:
        payload["repo"]["branch"] = args.branch
    if args.prompt:
        payload["prompt"] = args.prompt
    if args.step_limit:
        payload["config"] = {"agent": {"step_limit": args.step_limit}}

    task = _api("POST", "/tasks", json=payload)
    task_id = task["id"]
    print(f"Task created: {task_id}")

    # Submit
    _api("POST", f"/tasks/{task_id}/submit")
    print(f"Task submitted. Running on GPU ...")

    # Poll
    _poll_until_done(task_id)

    # Results
    _download_results(task_id, args.output_dir)


def cmd_status(args):
    """Check task status."""
    task = _api("GET", f"/tasks/{args.task_id}")
    status = task.get("status", "unknown")
    created = task.get("created_at", "")
    updated = task.get("updated_at", "")
    input_type = task.get("input_type", "")

    print(f"Task:    {args.task_id}")
    print(f"Status:  {status}")
    print(f"Type:    {input_type}")
    print(f"Created: {created}")
    print(f"Updated: {updated}")

    if task.get("safe_workload_id"):
        print(f"Workload: {task['safe_workload_id']}")
    if task.get("error_message"):
        print(f"Error:   {task['error_message']}")


def cmd_results(args):
    """Download results from a completed task."""
    task = _api("GET", f"/tasks/{args.task_id}")
    if task.get("status") != "completed":
        print(f"Task status is '{task.get('status')}', not 'completed'.")
        if task.get("status") == "running":
            print("Task is still running. Use 'status' to check progress.")
        return

    _download_results(args.task_id, args.output_dir)


def cmd_list(args):
    """List tasks."""
    params = {"limit": 20}
    if args.status:
        params["status"] = args.status

    data = _api("GET", "/tasks", params=params)
    tasks = data.get("tasks", [])

    if not tasks:
        print("No tasks found.")
        return

    # Table header
    print(f"{'Task ID':<38} {'Status':<12} {'Type':<6} {'Created':<20}")
    print("-" * 80)
    for t in tasks:
        tid = t.get("id", "")
        st = t.get("status", "")
        tp = t.get("input_type", "")
        cr = t.get("created_at", "")[:19]
        print(f"{tid:<38} {st:<12} {tp:<6} {cr:<20}")


def cmd_cancel(args):
    """Cancel a running task."""
    result = _api("POST", f"/tasks/{args.task_id}/cancel")
    print(f"Task {args.task_id} cancelled.")


# ── Helpers ───────────────────────────────────────────────────────────

def _poll_until_done(task_id, interval=30, max_wait=3600):
    """Poll task status until completed or failed."""
    elapsed = 0
    while elapsed < max_wait:
        task = _api("GET", f"/tasks/{task_id}")
        status = task.get("status", "unknown")

        if status == "completed":
            print(f"\nTask completed! (elapsed: {elapsed}s)")
            return
        elif status in ("failed", "cancelled"):
            print(f"\nTask {status}.")
            if task.get("error_message"):
                print(f"Error: {task['error_message']}")
            sys.exit(1)

        print(f"  [{elapsed}s] status: {status} ...", flush=True)
        time.sleep(interval)
        elapsed += interval

    print(f"\nTimeout after {max_wait}s. Task may still be running.")
    print(f"Check later with: python geak.py status {task_id}")
    sys.exit(1)


def _download_results(task_id, output_dir=None):
    """Download all output files from a task."""
    outputs = _api("GET", f"/tasks/{task_id}/outputs")
    files = outputs.get("files", [])

    if not files:
        print("No output files found.")
        return

    out_dir = output_dir or f"geak_output_{task_id[:8]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nDownloading {len(files)} file(s) to {out_dir}/")

    for f in files:
        fpath = f["path"]
        size = f.get("size", 0)
        url = f"{API_URL}/api/v1/tasks/{task_id}/download"
        resp = requests.get(url, headers=_headers(), params={"path": fpath}, stream=True)

        local_path = os.path.join(out_dir, fpath)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        with open(local_path, "wb") as fp:
            for chunk in resp.iter_content(chunk_size=8192):
                fp.write(chunk)

        print(f"  {fpath} ({size:,} bytes)")

    print(f"\nDone. Results saved to {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GEAK — GPU kernel optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # config
    p = sub.add_parser("config", help="Configure LLM model")
    p.add_argument("--model_name", required=True, help="Model name, e.g. openai/claude-opus-4.5")
    p.add_argument("--api_base", required=True, help="LLM API base URL")
    p.add_argument("--api_key", required=True, help="LLM API key (sk-xxx)")
    p.add_argument("--model_class", default="litellm", help="Model class (default: litellm)")

    # optimize
    p = sub.add_parser("optimize", help="Optimize HIP kernel file(s)")
    p.add_argument("files", nargs="+", help="Path to .hip file(s), e.g. silu.hip or silu.hip Makefile utils.h")
    p.add_argument("--prompt", help="Optimization instructions")
    p.add_argument("--step_limit", type=int, help="Max agent steps (default: 10)")
    p.add_argument("--gpu_count", type=int, help="Number of GPUs (default: 1)")
    p.add_argument("--output_dir", help="Output directory (default: geak_output_<id>)")

    # optimize-repo
    p = sub.add_parser("optimize-repo", help="Optimize HIP kernels in a git repo")
    p.add_argument("repo_url", help="Git repository URL")
    p.add_argument("--branch", help="Git branch (default: main)")
    p.add_argument("--prompt", help="Optimization instructions")
    p.add_argument("--step_limit", type=int, help="Max agent steps (default: 10)")
    p.add_argument("--output_dir", help="Output directory")

    # status
    p = sub.add_parser("status", help="Check task status")
    p.add_argument("task_id", help="Task ID")

    # results
    p = sub.add_parser("results", help="Download task results")
    p.add_argument("task_id", help="Task ID")
    p.add_argument("--output_dir", help="Output directory")

    # list
    p = sub.add_parser("list", help="List tasks")
    p.add_argument("--status", help="Filter by status: pending, running, completed, failed, cancelled")

    # cancel
    p = sub.add_parser("cancel", help="Cancel a running task")
    p.add_argument("task_id", help="Task ID")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    _check_env()

    commands = {
        "config": cmd_config,
        "optimize": cmd_optimize,
        "optimize-repo": cmd_optimize_repo,
        "status": cmd_status,
        "results": cmd_results,
        "list": cmd_list,
        "cancel": cmd_cancel,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
