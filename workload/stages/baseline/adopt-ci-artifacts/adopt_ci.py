#!/usr/bin/env python3
"""Official InferenceX CI run -> one baseline per config (search-space point). See SKILL.md.

For every point of the run that passes the `select` filter this reads the point's own CI
artifacts -- `bmk_agentic_*` (identity, official throughput, completed tokens, measured
window) and `server_logs_*` (the scheduler log and the client's records) -- and writes:

  points/<pid>/baseline.json         the single-point baseline schema, reference = this CI point
  points/<pid>/observed_inputs.json  exactly what observed_workload.py needs for this point
  points/<pid>/observed_workload.json  (validation run of observed_workload.py on it)
  baseline.json                      the first representative point's baseline, plus
                                     `points`, `groups` and `representative`

Nothing is re-measured. The numbers are the CI's own; `measured` is true because the CI
measured them, and `provenance.measured_on` says where.

Stdlib only. Artifacts are untrusted input: archives are unpacked with path checks and
only the members this skill reads are extracted.
"""

import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.environ.get("WORKLOAD_ROOT", ""), "scripts"))
from emit_result import emit, read_json  # noqa: E402

API = "https://api.github.com/repos/%s/actions/runs/%s/artifacts?per_page=100&page=%d"
KINDS = ("server_logs_", "bmk_agentic_", "agentic_", "power_audit_", "job_event_")
THROUGHPUT_KEY = "request_metrics.throughput.output.tokens_per_second"
TOKENS_KEYS = ("server_metrics.tokens.generation_total",)
INTERVAL_RE = re.compile(r"decode_log_interval['\"]?\s*[:=]\s*(\d+)")
LOG_TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d) \d\d:\d\d:\d\d")
PHASE_RE = re.compile(r"^(\d\d:\d\d:\d\d(?:\.\d+)?)\s.*Phase profiling \(profiling\) (started|complete)")
# members of the server_logs tarball this skill reads; nothing else is extracted
WANT_MEMBERS = (re.compile(r"(^|/)logs/[^/]*_agg_w\d+\.out$"),
                re.compile(r"(^|/)logs/[^/]*_(prefill|decode)_w\d+\.out$"),
                re.compile(r"(^|/)logs/agentic/benchmark\.log$"),
                re.compile(r"(^|/)logs/agentic/aiperf_artifacts/profile_export_aiperf\.json$"),
                re.compile(r"(^|/)logs/config\.yaml$"))


def resolve(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# --------------------------------------------------------------------------- sources
def http_json(url, token):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "geak-workload"})
    if token:
        req.add_unredirected_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def list_remote(repo, run_id, token):
    arts, page = [], 1
    while True:
        d = http_json(API % (repo, run_id, page), token)
        arts += d.get("artifacts") or []
        if len(arts) >= d.get("total_count", 0) or not d.get("artifacts"):
            return arts
        page += 1


def download_zip(url, token, dest):
    # The archive URL answers with a redirect to signed blob storage. The token goes only
    # to api.github.com: add_unredirected_header keeps it off the redirected request.
    req = urllib.request.Request(url, headers={"User-Agent": "geak-workload"})
    req.add_unredirected_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():
            safe_member(m.filename, dest)
        z.extractall(dest)


def safe_member(name, dest):
    target = os.path.realpath(os.path.join(dest, name))
    if not (target == os.path.realpath(dest) or target.startswith(os.path.realpath(dest) + os.sep)):
        raise SystemExit("unsafe archive member %r" % name)


def local_artifacts(dirs):
    """Pre-downloaded artifacts: directories named after the artifact, at most 2 levels
    below each given dir (the layout a manual download leaves)."""
    found = {}
    for root in dirs:
        for depth in ("*", "*/*"):
            for p in glob.glob(os.path.join(root, depth)):
                name = os.path.basename(p.rstrip("/"))
                if os.path.isdir(p) and name.startswith(KINDS):
                    found.setdefault(name, os.path.abspath(p))
    return found


def point_key(name):
    for k in KINDS:
        if name.startswith(k):
            return k[:-1], name[len(k):]
    return None, None


# --------------------------------------------------------------------------- per point
def extract_server_logs(art_dir, dest):
    """server_logs artifact -> the members WANT_MEMBERS names, under dest."""
    tars = glob.glob(os.path.join(art_dir, "*.tar.gz")) + glob.glob(os.path.join(art_dir, "*", "*.tar.gz"))
    got = []
    for t in tars:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not any(r.search(m.name) for r in WANT_MEMBERS):
                    continue
                safe_member(m.name, dest)
                tf.extract(m, dest, filter="data")
                got.append(os.path.normpath(os.path.join(dest, m.name)))
    # some uploads carry the logs unpacked
    for r in WANT_MEMBERS:
        for p in glob.glob(os.path.join(art_dir, "**", "*"), recursive=True):
            if os.path.isfile(p) and r.search(p.replace(os.sep, "/")):
                got.append(p)
    return sorted(set(got))


def window_of(bmk, bench_log, server_log, tz_offset):
    """Measured window (server-log local time) from the power audit's unix window, else
    from the client's Phase profiling started/complete lines dated by the server log."""
    pa = bmk.get("power_audit") or {}
    s, e = num(pa.get("window_start_unix")), num(pa.get("window_end_unix"))
    tz = dt.timezone(dt.timedelta(minutes=tz_offset))
    if s and e and e > s:
        f = lambda x: dt.datetime.fromtimestamp(x, tz).strftime("%Y-%m-%d %H:%M:%S.%f")  # noqa: E731
        return f(s), f(e), "bmk power_audit window"
    if not (bench_log and server_log):
        return None, None, "no window source"
    day = None
    with open(server_log, errors="replace") as fh:
        for line in fh:
            m = LOG_TS_RE.match(line)
            if m:
                day = m.group(1)
                break
    start = end = None
    with open(bench_log, errors="replace") as fh:
        for line in fh:
            m = PHASE_RE.match(line)
            if m:
                if m.group(2) == "started":
                    start = m.group(1)
                else:
                    end = m.group(1)
    if not (day and start and end):
        return None, None, "Phase profiling markers not found"
    s = dt.datetime.strptime("%s %s" % (day, start.split(".")[0]), "%Y-%m-%d %H:%M:%S")
    e = dt.datetime.strptime("%s %s" % (day, end.split(".")[0]), "%Y-%m-%d %H:%M:%S")
    if e <= s:          # the leg crossed midnight
        e += dt.timedelta(days=1)
    return s.strftime("%Y-%m-%d %H:%M:%S"), e.strftime("%Y-%m-%d %H:%M:%S"), "benchmark.log phase markers"


def tz_minutes(art_dirs):
    for d in art_dirs:
        for p in glob.glob(os.path.join(d, "**", "agentic_power_timezone_offset.txt"), recursive=True):
            m = re.match(r"\s*([+-])(\d\d):?(\d\d)", open(p).read())
            if m:
                return (1 if m.group(1) == "+" else -1) * (int(m.group(2)) * 60 + int(m.group(3)))
    return 0


def pid_of(bmk):
    kv = bmk.get("kv_offloading") or "none"
    be = (bmk.get("kv_offload_backend") or {}).get("name")
    spec = bmk.get("spec_decoding") or "none"
    s = "tp%s-ep%s-c%s-kv%s%s-spec%s" % (bmk.get("tp"), bmk.get("ep"), bmk.get("conc"), kv,
                                          ("-" + be) if be else "", spec)
    s = re.sub(r"[^A-Za-z0-9-]+", "-", s)
    return re.sub(r"-{2,}", "-", s)


def matches(bmk, sel):
    for k, want in (sel or {}).items():
        have = bmk.get("infmax_model_prefix") if k == "model_prefix" else bmk.get(k)
        if isinstance(have, dict):
            have = have.get("name")
        want = want if isinstance(want, list) else [want]
        if have not in want and str(have) not in [str(w) for w in want]:
            return False
    return True


# Launch settings that select kernel paths. Configs that differ in any of them are not one
# deployment for kernel purposes, so they never share a group (and never one UT): a UT runs
# in one environment and would time every config's cases on one config's kernels.
SPLIT_ON = ["attention-backend", "kv-cache-dtype", "page-size", "speculative-algorithm",
            "speculative-num-draft-tokens", "speculative-draft-attention-backend",
            "enable-dp-attention", "dp-size", "moe-runner-backend", "quantization",
            "env:SGLANG_USE_AITER", "env:SGLANG_USE_AITER_UNIFIED_ATTN", "env:MUSE_MOE_BACKEND"]


def launch_of(config_yaml):
    """CI recipe (logs/config.yaml) -> {"args": {...}, "env": {...}} of the serving role."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        doc = yaml.safe_load(open(config_yaml))
    except Exception:  # noqa: BLE001
        return None
    roles = (doc or {}).get("roles") or {}
    if set(roles) != {"agg"}:
        return {"unsupported_roles": sorted(roles)}
    agg = roles["agg"] or {}
    return {"args": {str(k): v for k, v in (agg.get("args") or {}).items()},
            "env": {str(k): str(v) for k, v in (agg.get("env") or {}).items()},
            "image": ((doc.get("model") or {}).get("container") or "").strip(),
            "recipe": config_yaml}


def launch_value(launch, key):
    if not launch:
        return None
    if key.startswith("env:"):
        return (launch.get("env") or {}).get(key[4:])
    return (launch.get("args") or {}).get(key)


def serving_env(launch, path):
    """The CI launch as an env file for bench_e2e.sh: the role env, and every server arg
    except the parallel sizes (bench_e2e passes TP itself) as EXTRA_SERVER_ARGS."""
    import shlex
    args = []
    for k, v in (launch.get("args") or {}).items():
        if k in ("tensor-parallel-size", "tp-size", "served-model-name"):
            continue
        if v is True:
            args.append("--" + k)
        elif v is False or v is None:
            continue
        else:
            args += ["--" + k, str(v)]
    with open(path, "w") as fh:
        fh.write("# generated by adopt-ci-artifacts from %s\n" % launch.get("recipe"))
        for k, v in sorted((launch.get("env") or {}).items()):
            fh.write("%s=%s\n" % (k, shlex.quote(v)))
        fh.write("EXTRA_SERVER_ARGS=%s\n" % shlex.quote(" ".join(shlex.quote(a) for a in args)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- main
def main():
    out = os.environ["OUT"]
    params = read_json(os.environ["PARAMS"])
    ci = params.get("ci") or {}
    repo = ci.get("repo", "SemiAnalysisAI/InferenceX")
    run_id = str(ci.get("run_id") or "")
    token = os.environ.get(ci.get("token_env", "GITHUB_TOKEN")) or None
    local = local_artifacts(ci.get("artifacts_dirs") or [])
    noise_band = float(params.get("noise_band", 1.045))
    group_by = params.get("group_by", ["ep"])
    split_on = params.get("split_on", SPLIT_ON)
    require_power = params.get("require_power_valid", True)
    observed_script = os.path.join(os.environ.get("WORKLOAD_ROOT", ""), "stages", "profile",
                                   "kernel-extract", "scripts", "observed_workload.py")

    # 1. which artifacts belong to which point
    names = dict(local)
    remote = {}
    if run_id:
        try:
            for a in list_remote(repo, run_id, token):
                if not a.get("expired"):
                    remote[a["name"]] = a
        except Exception as e:  # noqa: BLE001
            if not local:
                return emit("blocked", notes="cannot list artifacts of run %s (%s) and no "
                            "ci.artifacts_dirs given" % (run_id, e))
    by_point = {}
    for name in set(names) | set(remote):
        kind, key = point_key(name)
        if kind:
            by_point.setdefault(key, {})[kind] = name
    if not by_point:
        return emit("blocked", notes="no InferenceX point artifacts found (run %r, dirs %s)"
                    % (run_id, ci.get("artifacts_dirs")))

    def fetch(name, dest):
        if name in local:
            return local[name]
        if name not in remote:
            return None
        if not token:
            raise LookupError("artifact %s is not available locally and %s is unset (GitHub "
                              "requires a token to download artifacts, even of a public repo)"
                              % (name, ci.get("token_env", "GITHUB_TOKEN")))
        d = os.path.join(dest, name)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            download_zip(remote[name]["archive_download_url"], token, d)
        return d

    points, problems = [], []
    for key, kinds in sorted(by_point.items()):
        if "bmk_agentic" not in kinds:
            continue
        cache = os.path.join(out, "artifacts")
        try:
            bdir = fetch(kinds["bmk_agentic"], cache)
        except LookupError as e:
            return emit("blocked", notes=str(e))
        bmk_files = glob.glob(os.path.join(bdir, "**", "*.json"), recursive=True) if bdir else []
        if not bmk_files:
            problems.append("%s: bmk_agentic has no json" % key)
            continue
        bmk = read_json(bmk_files[0])
        if not matches(bmk, params.get("select")):
            continue
        pid = pid_of(bmk)
        pdir = os.path.join(out, "points", pid)
        os.makedirs(pdir, exist_ok=True)
        try:
            sdir = fetch(kinds.get("server_logs", ""), cache) if kinds.get("server_logs") else None
        except LookupError as e:
            return emit("blocked", notes=str(e))
        if not sdir:
            problems.append("%s: no server_logs artifact" % pid)
            continue
        files = extract_server_logs(sdir, os.path.join(pdir, "server_logs"))
        server = [f for f in files if re.search(r"_agg_w\d+\.out$", f)]
        disagg = [f for f in files if re.search(r"_(prefill|decode)_w\d+\.out$", f)]
        bench = next((f for f in files if f.endswith("benchmark.log")), None)
        if not server:
            problems.append("%s: no aggregated server log in server_logs (%s)%s"
                            % (pid, [os.path.basename(f) for f in files],
                               "; disaggregated serving is not supported yet" if disagg else ""))
            continue
        interval = None
        with open(server[0], errors="replace") as fh:
            for line in fh:
                m = INTERVAL_RE.search(line)
                if m:
                    interval = int(m.group(1))
                    break
        tz = tz_minutes([sdir, os.path.dirname(bdir)])
        ws, we, wsrc = window_of(bmk, bench, server[0], tz)
        tput = num(resolve(bmk, THROUGHPUT_KEY))
        tokens = next((int(v) for v in (num(resolve(bmk, k)) for k in TOKENS_KEYS) if v), None)
        why = []
        if not tput or tput <= 0:
            why.append("no %s" % THROUGHPUT_KEY)
        if not interval:
            why.append("decode_log_interval not in the server log")
        if not ws:
            why.append("no measured window (%s)" % wsrc)
        if require_power and bmk.get("power_valid") in (0, False):
            why.append("power_valid=false: %s" % bmk.get("power_invalid_reasons"))
        if why:
            problems.append("%s: %s" % (pid, "; ".join(why)))
            continue
        inputs = {"log": server, "rank": 0, "decode_log_interval": interval,
                  "window_start": ws, "window_end": we, "window_source": wsrc,
                  "completed_output_tokens": tokens, "bench_log": bench}
        json.dump(inputs, open(os.path.join(pdir, "observed_inputs.json"), "w"), indent=1)
        check = None
        if os.path.exists(observed_script):
            cmd = [sys.executable, "-I", observed_script, "--log", *server, "--rank", "0",
                   "--decode-log-interval", str(interval), "--window-start", ws, "--window-end", we,
                   "--out", os.path.join(pdir, "observed_workload.json")]
            if tokens:
                cmd += ["--completed-output-tokens", str(tokens)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                problems.append("%s: observed_workload.py failed: %s" % (pid, (r.stderr or r.stdout)[-400:]))
                continue
            ow = read_json(os.path.join(pdir, "observed_workload.json"))
            check = (ow.get("decode") or {}).get("crosscheck") or {}
            d = check.get("disagreement_pct")
            if d is not None and d > float(params.get("max_decode_disagreement_pct", 5.0)):
                problems.append("%s: decode expansion disagrees with completed tokens by %.2f%%"
                                % (pid, d))
                continue
        config = {"tp": bmk.get("tp"), "ep": bmk.get("ep"), "gpus": bmk.get("num_gpus"),
                  "concurrency": bmk.get("conc"), "kv_offloading": bmk.get("kv_offloading"),
                  "kv_offload_backend": (bmk.get("kv_offload_backend") or {}).get("name"),
                  "spec_decoding": bmk.get("spec_decoding"), "dp_attention": bmk.get("dp_attention"),
                  "pp": bmk.get("pp")}
        base = {
            "metric": "output_token_throughput", "unit": "tok/s", "throughput": tput,
            "higher_is_better": True, "noise_band": noise_band, "config": config,
            "reference": {"name": "ci", "value": tput, "ratio": 1.0},
            "measured": True,
            "point_id": pid,
            "provenance": {"method": "adopt-ci-artifacts", "measured_on": "inferencex_ci",
                           "repo": repo, "run_id": run_id, "artifact_key": key,
                           "hw": bmk.get("hw"), "image": bmk.get("image"), "model": bmk.get("model"),
                           "recipe_fingerprint": bmk.get("recipe_fingerprint"),
                           "bmk": bmk_files[0], "bmk_sha256": sha256(bmk_files[0]),
                           "server_logs": server, "throughput_key": THROUGHPUT_KEY,
                           "power_valid": bmk.get("power_valid"),
                           "decode_crosscheck": check},
        }
        recipe = next((f for f in files if f.endswith("config.yaml")), None)
        launch = launch_of(recipe) if recipe else None
        if launch and launch.get("unsupported_roles"):
            problems.append("%s: CI recipe roles %s; only aggregated serving is supported"
                            % (pid, launch["unsupported_roles"]))
            continue
        if launch:
            json.dump(launch, open(os.path.join(pdir, "launch.json"), "w"), indent=1)
            serving_env(launch, os.path.join(pdir, "serving_env.sh"))
            base["launch"] = os.path.join("points", pid, "launch.json")
            base["serving_env"] = os.path.join("points", pid, "serving_env.sh")
            base["_launch"] = launch
        else:
            problems.append("%s: no readable CI recipe (logs/config.yaml) for the launch settings" % pid)
            continue
        json.dump({k: v for k, v in base.items() if k != "_launch"},
                  open(os.path.join(pdir, "baseline.json"), "w"), indent=1)
        points.append(base)

    if not points:
        return emit("blocked", notes="no usable point: " + " | ".join(problems or ["select matched nothing"]))
    if problems and not params.get("allow_partial", False):
        return emit("blocked", notes="%d of %d selected points unusable (set allow_partial to "
                    "proceed without them): %s" % (len(problems), len(points) + len(problems),
                                                   " | ".join(problems)))

    # 2. groups and representative points (lowest and highest concurrency per group)
    groups = {}
    for b in points:
        g = "-".join("%s%s" % (k, b["config"].get(k)) for k in group_by)
        path = tuple((k, launch_value(b["_launch"], k)) for k in split_on)
        g = (g, path)
        groups.setdefault(g, []).append(b)
    rep = params.get("representative")
    if not rep:
        rep = []
        for g, bs in sorted(groups.items()):
            bs = sorted(bs, key=lambda b: b["config"]["concurrency"])
            rep += sorted({bs[0]["point_id"], bs[-1]["point_id"]},
                          key=lambda p: [b["point_id"] for b in bs].index(p))
    known = {b["point_id"] for b in points}
    if set(rep) - known:
        return emit("blocked", notes="representative points %s are not among %s"
                    % (sorted(set(rep) - known), sorted(known)))
    images = sorted({b["provenance"]["image"] for b in points})
    models = sorted({b["provenance"]["model"] for b in points})
    head = next(b for b in points if b["point_id"] == rep[0])
    # name the groups; a group split only by a path key gets that key in its name
    names, launch_diff = {}, {}
    by_base = {}
    for (gname, path) in groups:
        by_base.setdefault(gname, []).append(path)
    for (gname, path), bs in groups.items():
        alts = by_base[gname]
        name = gname
        if len(alts) > 1:
            differ = [k for i, (k, v) in enumerate(path) if len({a[i][1] for a in alts}) > 1]
            name += "-" + "-".join("%s=%s" % (k.replace("env:", ""), v) for k, v in path if k in differ)
            name = re.sub(r"[^A-Za-z0-9=.-]+", "-", name)
        names[(gname, path)] = name
        keys = sorted({k for b in bs for k in list(b["_launch"]["args"]) + ["env:" + e for e in b["_launch"]["env"]]})
        diff = {k: {b["point_id"]: launch_value(b["_launch"], k) for b in bs} for k in keys
                if len({json.dumps(launch_value(b["_launch"], k)) for b in bs}) > 1}
        if diff:
            launch_diff[name] = diff
    groups = {names[k]: v for k, v in groups.items()}
    for b in points:
        b.pop("_launch", None)
    top = dict(head, multi_config=True,
               points=[{"point_id": b["point_id"], "config": b["config"], "throughput": b["throughput"],
                        "baseline": os.path.join("points", b["point_id"], "baseline.json"),
                        "observed_inputs": os.path.join("points", b["point_id"], "observed_inputs.json")}
                       for b in points],
               groups={g: [b["point_id"] for b in sorted(bs, key=lambda b: b["config"]["concurrency"])]
                       for g, bs in groups.items()},
               group_by=group_by, split_on=split_on, launch_diff=launch_diff,
               representative=rep,
               note="top-level throughput/config are the first representative point's; every "
                    "point's own baseline is under points/<pid>/baseline.json")
    json.dump(top, open(os.path.join(out, "baseline.json"), "w"), indent=1)
    if len(images) == 1 and len(models) == 1:
        json.dump({"active_stack": {"model": models[0], "image": images[0]}},
                  open(os.path.join(out, "stack_identity.json"), "w"), indent=1)
    return emit("ok", artifacts={"baseline": "baseline.json"},
                metrics={"points": len(points), "groups": len(groups),
                         "representative": len(rep), "throughput": head["throughput"]},
                notes="CI run %s: %d points in %d groups (%s); representative %s%s"
                      % (run_id or "<local>", len(points), len(groups), ", ".join(sorted(groups)),
                         rep, ("; skipped: " + " | ".join(problems)) if problems else ""))


if __name__ == "__main__":
    sys.exit(main())
