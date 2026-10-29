#!/usr/bin/env python3
"""Refuse to start a CI reproduction that cannot possibly reproduce.

Every check here costs milliseconds; the run it guards costs an hour on eight GPUs and
is the reference every later comparison is measured against. A baseline that silently
measured something adjacent to the CI point is worse than no baseline, because every
downstream delta inherits the error with a plausible-looking sign.

The six things that must be pinned, and what goes wrong when they are not:

1. **Image digest, not tag.** A tag is a moving pointer. `v0.5.19-rocm720-mi35x-20260913`
   can be repushed, and the only evidence afterwards is a throughput that drifted for no
   reason anybody can name. The digest is the identity.
2. **Data and seed.** Same corpus file, same entry count, same seed, same per-lane warmup.
   The trace corpus decides the prompt-length mix, which decides the prefill/decode ratio,
   which is most of the throughput.
3. **`benchmark_id`.** The AgentX scenario builds its cache-bust marker out of AIPerf's
   per-invocation benchmark id, so the id *is* the KV keyspace. Reuse one and the second
   run starts on a warmed prefix — it will look faster, and nothing in its output says so.
   Ids are therefore recorded in a ledger and never reissued.
4. **Cache-bust target.** The scenario locks `first_turn_prefix`. Any other value changes
   how much of each prompt can be served from cache, which changes the number being
   compared.
5. **CPU binding on the real topology.** One logical CPU per physical core, taken from the
   machine's own `lscpu -p` and only from cores this process was actually granted. Binding
   to logical siblings puts two ranks on one core's execution resources; binding to cores
   outside the cpuset makes sglang's own affinity pass (`SGLANG_SET_CPU_AFFINITY=1`)
   compute per-rank masks the cgroup then refuses, which is how a prior run on this
   cluster ended up having to set it to 0 in both arms.
6. **A fresh container and empty compile caches.** A warm Triton/XDG cache can carry
   kernels compiled from *another campaign's overlay* — this exact stack was once profiled
   with glm52_optimized kernels still in the Triton cache, and the only reason anybody
   noticed was the MLIR source locations inside the cache entries. The cache must be new,
   and (verified after the run, not here) must have filled up, which is the only proof the
   environment variable actually reached the server process.

Separately, a KV-offload point must hand the client every KV_OFFLOAD* variable it reads:
the AgentX client checks some at start and the rest only after the measurement window
(see check_kv_offload).

Model weights are deliberately **not** in the cold set: re-downloading them changes
nothing about the measurement and can change the revision underneath you. Pin the
revision instead.

Usage: preflight.py --params params.json --out <dir> [--ledger <path>]
Exit 0 with `alignment.json` written, or non-zero having listed every blocker at once.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# The compile caches, which must be cold. Nothing here holds model weights.
COLD_CACHE_VARS = ("TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR",
                   "TRITON_HOME")
# Variables that must never be pointed at a fresh empty directory: emptying them either
# re-downloads tens of gigabytes or silently changes which weights are served.
NEVER_COLD = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "MODEL_PATH")
# What the AgentX client reads to describe a KV-offload point; see check_kv_offload().
KV_OFFLOAD_VARS = ("KV_OFFLOADING", "KV_OFFLOAD_BACKEND", "KV_OFFLOAD_BACKEND_METADATA",
                   "TOTAL_CPU_DRAM_GB")


def sh(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


# ---------------------------------------------------------------- cpu topology

def physical_cores(granted=None):
    """One logical CPU per physical core, restricted to what we were granted.

    `lscpu -p=CPU,CORE,SOCKET,NODE` is the machine's own account of its topology; guessing
    that siblings are `cpu` and `cpu + ncores/2` is right on this node and wrong on the
    next one.
    """
    rc, out, err = sh(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    if rc != 0:
        return None, "lscpu -p failed: %s" % (err or rc)
    if granted is None:
        granted = os.sched_getaffinity(0)
    seen, plan, skipped = {}, [], []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        cpu, core, sock = int(parts[0]), parts[1], parts[2]
        key = (sock, core)
        if cpu not in granted:
            continue
        if key in seen:
            skipped.append(cpu)          # an SMT sibling of a core already taken
            continue
        seen[key] = cpu
        plan.append(cpu)
    return {"cpus": sorted(plan), "physical_cores": len(plan),
            "smt_siblings_dropped": len(skipped),
            "granted_logical_cpus": len(granted)}, None


# ---------------------------------------------------------------- checks

def check_pins(p, problems):
    pins = p.get("pins") or {}

    digest = (pins.get("image_digest") or "").strip()
    image = (pins.get("image") or "").strip()
    if not DIGEST_RE.match(digest):
        problems.append(
            "pins.image_digest is %r. A tag is a moving pointer and cannot identify what "
            "was measured; pin the digest (docker image inspect %s --format "
            "'{{index .RepoDigests 0}}')." % (digest or None, image or "<image>"))
    elif image:
        rc, out, _ = sh(["docker", "image", "inspect", image,
                         "--format", "{{json .RepoDigests}}"])
        if rc != 0:
            problems.append("the pinned image %s is not present locally; pull it by "
                            "digest (%s@%s) before measuring." % (image, image.split(":")[0],
                                                                  digest))
        else:
            try:
                have = json.loads(out)
            except ValueError:
                have = []
            if not any(d.endswith(digest) for d in have):
                problems.append(
                    "the local image tagged %s has digest(s) %s, which do not include the "
                    "pinned %s. The tag has moved; pull the digest explicitly." %
                    (image, have, digest))

    for key, why in (
            ("model", "which weights were served"),
            ("model_revision", "which revision of those weights -- a repo can move under "
                               "a fixed name"),
            ("seed", "the sampling/scheduling seed; an unpinned seed changes the trace "
                     "replay order"),
            ("benchmark_id", "the KV keyspace: the AgentX cache-bust marker is built from "
                             "it"),
            ("cache_bust", "how much of each prompt can be served from cache"),
            ("duration_s", "the measurement window; a short window is a screening "
                           "diagnostic, not a baseline")):
        if pins.get(key) in (None, ""):
            problems.append("pins.%s is not pinned, and it decides %s." % (key, why))

    if pins.get("cache_bust") not in (None, "", "first_turn_prefix"):
        problems.append(
            "pins.cache_bust is %r. The AgentX scenario plugin locks first_turn_prefix; "
            "any other target changes the cache hit rate and therefore the number."
            % pins["cache_bust"])

    dur = pins.get("duration_s")
    if isinstance(dur, (int, float)) and dur < 900:
        problems.append("pins.duration_s is %s. Runs under 900 s are screening "
                        "diagnostics on this workload, not a baseline." % dur)

    # The corpus may be a local file or a named public-dataset loader -- the AgentX
    # harness takes the latter (`--public-dataset semianalysis_cc_traces_weka_062126`),
    # and demanding a path would make an honest config impossible. Either way it must be
    # named here, because the loader name selects the corpus *generation*: the 062126 and
    # 061526 corpora differ in prompt-length mix, hence in prefill/decode ratio, hence in
    # throughput.
    corpus = pins.get("trace_corpus") or {}
    path, loader = corpus.get("path"), corpus.get("loader")
    if not path and not loader:
        problems.append("pins.trace_corpus names neither a `path` nor a public-dataset "
                        "`loader`; the corpus decides the prompt-length mix and with it "
                        "the prefill/decode ratio.")
    elif path and not os.path.exists(path):
        problems.append("pins.trace_corpus.path %s does not exist on this machine." % path)
    elif path and corpus.get("entries"):
        try:
            n = sum(1 for line in open(path, "rb") if line.strip())
            if n != corpus["entries"]:
                problems.append("the corpus at %s holds %d entries; the CI run used %d. "
                                "This is a different dataset." % (path, n, corpus["entries"]))
        except OSError as e:
            problems.append("cannot read the corpus at %s: %s" % (path, e))
    return pins


def check_benchmark_id(pins, ledger_path, problems):
    """An id is spent the moment it is used: its cache-bust marker has warmed a keyspace."""
    bid = str(pins.get("benchmark_id") or "")
    if not bid or bid == "auto":
        return []
    used = []
    if ledger_path and os.path.exists(ledger_path):
        try:
            used = json.load(open(ledger_path)).get("used") or []
        except (OSError, ValueError):
            used = []
    if bid in [u.get("benchmark_id") for u in used if isinstance(u, dict)]:
        problems.append(
            "benchmark_id %r has already been used on this machine (%s). The AgentX "
            "cache-bust marker embeds it, so a second run under the same id inherits the "
            "first run's warmed KV prefixes and measures high. Choose a new id." %
            (bid, ledger_path))
    return used


def cache_plan(iso, out_dir):
    """Where each compile cache will live.

    Two shapes, because passing `-e TRITON_CACHE_DIR=...` is not enough on a harness that
    exports the variable itself. The AgentX harness's `environment.sh` sets
    `TRITON_CACHE_DIR=$IX_RUN/cache/triton` unconditionally, so a `-e` pin loses and the
    run compiles into whatever that directory already held.

      * `isolation.cache_dirs` -- an explicit {VAR: dest} map, where dest is either a path
        (used on both sides) or {"container": "<path the harness exports>"}. A fresh empty
        host directory is created under `cache_root` and bind-mounted onto that container
        path, so the harness's own export now lands in a cold directory instead of in
        whatever the shared run tree already held. The host side stays private to this
        run -- the harness's real cache tree is never emptied, only shadowed.
      * otherwise -- one directory per variable under `cache_root`, exported only.

    Either way the proof is the same and is taken after the run: the host directory
    filled up. Returns (cache_root, {VAR: host_dir}, {host_dir: container_dir}).
    """
    cache_root = iso.get("cache_root") or os.path.join(out_dir, "caches")
    explicit = iso.get("cache_dirs") or {}
    if not explicit:
        vars_ = list(iso.get("cache_vars") or COLD_CACHE_VARS)
        return cache_root, dict((v, os.path.join(cache_root, v.lower())) for v in vars_), {}
    dirs, mounts = {}, {}
    for var, dest in explicit.items():
        container = dest.get("container") if isinstance(dest, dict) else dest
        host = (dest.get("host") if isinstance(dest, dict) else None) \
            or os.path.join(cache_root, var.lower())
        dirs[var] = host
        mounts[host] = container
    return cache_root, dirs, mounts


def check_isolation(p, out_dir, problems):
    iso = p.get("isolation") or {}
    cache_root, plan, mounts = cache_plan(iso, out_dir)
    for v in plan:
        if v in NEVER_COLD:
            problems.append(
                "isolation lists %s among the compile caches. That is a weights cache: "
                "emptying it re-downloads the model and can change the revision "
                "underneath the run. Pin pins.model_revision instead." % v)
    for v, d in sorted(plan.items()):
        if os.path.isdir(d) and os.listdir(d):
            problems.append(
                "%s would point at %s, which already holds %d entries. A warm Triton/XDG "
                "cache can carry kernels compiled from another campaign's overlay -- that "
                "has happened on this stack -- so the baseline would not be measuring the "
                "pristine image. Use a fresh cache_root." % (v, d, len(os.listdir(d))))
    if iso.get("fresh_container") is False:
        problems.append("isolation.fresh_container is false. Reusing a container carries "
                        "its page cache, its compiled kernels and whatever a previous "
                        "session installed in place; none of that is in the image.")
    name = iso.get("container_name")
    if name:
        rc, out, _ = sh(["docker", "ps", "-aq", "--filter", "name=^%s$" % name])
        if rc == 0 and out.strip():
            problems.append("a container named %s already exists (%s). This skill will "
                            "not reuse it and will not silently remove it; remove it "
                            "yourself or choose another name." % (name, out.strip()))
    return cache_root, plan, mounts


def check_cpu(p, problems):
    cpu = p.get("cpu") or {}
    mode = cpu.get("bind", "physical-cores")
    if mode == "inherit":
        return {"mode": "inherit", "cpus": sorted(os.sched_getaffinity(0)),
                "note": "binding inherited; recorded so the candidate arm can match it"}
    if mode == "explicit":
        cpus = cpu.get("cpus") or []
        if not cpus:
            problems.append("cpu.bind is explicit but cpu.cpus is empty.")
        return {"mode": "explicit", "cpus": sorted(cpus)}

    plan, err = physical_cores()
    if err:
        problems.append("cannot read the CPU topology (%s), so physical-core binding "
                        "cannot be planned. Set cpu.bind to inherit and record why." % err)
        return {"mode": "physical-cores", "cpus": []}
    plan["mode"] = "physical-cores"
    if plan["physical_cores"] < 8:
        problems.append(
            "only %d physical cores were granted to this process. An 8-rank TP8 server "
            "cannot be bound one rank per core, and sglang's own affinity pass will "
            "compute masks the cgroup refuses." % plan["physical_cores"])
    want = cpu.get("sglang_set_cpu_affinity")
    if want in (1, "1", True) and plan["physical_cores"] < (p.get("ranks") or 8):
        problems.append(
            "cpu.sglang_set_cpu_affinity=1 with %d granted physical cores for %s ranks: "
            "sglang derives per-rank masks from the full topology, so under a restricted "
            "cpuset it binds to CPUs the cgroup denies. Either grant the whole node or "
            "set it to 0 -- and then set it to 0 in every arm it is compared against."
            % (plan["physical_cores"], p.get("ranks") or 8))
    return plan


# ---------------------------------------------------------------- kv offload

ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def read_env_file(path):
    """KEY=VALUE assignments from a file the harness sources with `set -a`. Read, never run."""
    env = {}
    for line in open(path):
        m = ASSIGN_RE.match(line.strip())
        if not m:
            continue
        try:
            words = shlex.split(m.group(2), comments=True)
        except ValueError:
            words = [m.group(2)]
        env[m.group(1)] = " ".join(words)
    return env


def harness_env(p, problems):
    """The environment the client will see, as far as it can be read without running it.

    `params.env` reaches the container with `-e`; each `harness.env_files` entry (relative
    to `harness.dir`) is sourced by the harness itself and so wins over it. When
    `env_files` is not given, the reference harness's `pins.env` is read if it exists.
    """
    h = p.get("harness") or {}
    env, sources = dict((k, str(v)) for k, v in (p.get("env") or {}).items()), ["params.env"]
    files = h.get("env_files")
    if files is None:
        files = ["pins.env"] if h.get("dir") and \
            os.path.isfile(os.path.join(h["dir"], "pins.env")) else []
    for f in files:
        path = f if os.path.isabs(f) else os.path.join(h.get("dir") or "", f)
        try:
            env.update(read_env_file(path))
            sources.append(path)
        except OSError as e:
            problems.append("harness.env_files names %s, which cannot be read (%s)." % (path, e))
    return env, sources


def check_kv_offload(p, problems):
    """The AgentX client checks the KV-offload variables twice, and the second time is late.

    `infx/bench/agentic/run.py` refuses to start when KV_OFFLOADING=dram has no
    KV_OFFLOAD_BACKEND or TOTAL_CPU_DRAM_GB -- after the server has already spent its
    startup. `infx/results/agentic` then refuses to aggregate unless
    KV_OFFLOAD_BACKEND_METADATA also parses and names the same backend -- after the whole
    measurement window, so the run completes, leaves no throughput, and the hour is gone.
    Both happened on one hicache baseline (c64, 2026-10-28). The stricter of the two rules
    is checked here, before any GPU time.
    """
    env, sources = harness_env(p, problems)
    declared = (p.get("config") or {}).get("kv_offloading")
    seen = env.get("KV_OFFLOADING")
    where = ", ".join(sources)
    if declared and seen and seen != declared:
        problems.append("config.kv_offloading is %r but the client would see KV_OFFLOADING=%r "
                        "(from %s)." % (declared, seen, where))
    mode = declared or seen
    doc = {"mode": mode, "sources": sources,
           "env": dict((k, env[k]) for k in KV_OFFLOAD_VARS if k in env)}
    if not mode:
        return doc
    backend = env.get("KV_OFFLOAD_BACKEND", "")
    raw = env.get("KV_OFFLOAD_BACKEND_METADATA")
    if mode == "none":
        if backend or raw not in (None, "", "null"):
            problems.append("KV offload is off but KV_OFFLOAD_BACKEND%s is set (from %s); the "
                            "AgentX aggregation refuses that combination after the run."
                            % ("" if backend else "_METADATA", where))
        return doc
    missing = [v for v in KV_OFFLOAD_VARS if env.get(v) in (None, "")]
    if missing:
        problems.append(
            "KV offload is %r, and the AgentX client needs %s, which nothing it is given sets "
            "(looked in %s). run.py stops at client start without KV_OFFLOAD_BACKEND or "
            "TOTAL_CPU_DRAM_GB; the aggregation stops only after the full window without "
            "KV_OFFLOAD_BACKEND_METADATA. Copy them from the CI job's benchmark env into "
            "params.env or a harness.env_files entry." % (mode, ", ".join(missing), where))
    if backend == "none":
        problems.append("KV_OFFLOAD_BACKEND is 'none' while KV offload is %r." % mode)
    if raw not in (None, ""):
        try:
            meta = json.loads(raw)
        except ValueError:
            meta = None
        ok = isinstance(meta, dict) and set(meta) in ({"name"}, {"name", "version"}) and \
            all(isinstance(v, str) and v for v in meta.values())
        if not ok:
            problems.append("KV_OFFLOAD_BACKEND_METADATA is %r; the aggregation wants a JSON "
                            "object with a non-empty 'name' and optional 'version'." % raw)
        elif backend and meta["name"] != backend:
            problems.append("KV_OFFLOAD_BACKEND_METADATA names %r but KV_OFFLOAD_BACKEND is %r; "
                            "the aggregation requires them to match." % (meta["name"], backend))
    dram = env.get("TOTAL_CPU_DRAM_GB", "")
    if dram and not (dram.isdigit() and int(dram) > 0):
        problems.append("TOTAL_CPU_DRAM_GB is %r; run.py wants a positive integer." % dram)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=None)
    a = ap.parse_args()

    p = json.load(open(a.params))
    problems = []

    pins = check_pins(p, problems)
    used = check_benchmark_id(pins, a.ledger, problems)
    cache_root, cache_dirs, cache_mounts = check_isolation(p, a.out, problems)
    cpu_plan = check_cpu(p, problems)
    kv_offload = check_kv_offload(p, problems)

    ci = p.get("ci") or {}
    if not ci.get("output_throughput"):
        problems.append("ci.output_throughput is not set: there is nothing to align the "
                        "reproduction against, and 'it ran' is not 'it reproduced'.")

    doc = {
        "pins": pins,
        "ci": ci,
        "cpu_plan": cpu_plan,
        "kv_offload": kv_offload,
        "cache_root": cache_root,
        "cache_vars": cache_dirs,
        "cache_mounts": cache_mounts,
        "benchmark_ids_already_used": len(used),
        "problems": problems,
        "verdict": "blocked" if problems else "ready",
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "alignment.json"), "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    if problems:
        print("preflight refused the run; every blocker is listed, not just the first:",
              file=sys.stderr)
        for x in problems:
            print("  - %s" % x, file=sys.stderr)
        return 1
    print("preflight ok: digest %s, seed %s, benchmark_id %s, cache_bust %s, %d physical "
          "cores, cold caches under %s"
          % (pins.get("image_digest"), pins.get("seed"), pins.get("benchmark_id"),
             pins.get("cache_bust"), len(cpu_plan.get("cpus") or []), cache_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
