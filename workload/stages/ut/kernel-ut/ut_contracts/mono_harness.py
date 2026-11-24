"""Multi-rank harness of a mono_layer task (UT contract family `mono_layer`): the generic part.

scaffold_ut.py copies this file into the task as rank_harness.py; it is frozen task source,
outside the candidate overlay, and a task never edits it. Everything model-specific -- weights,
KV pools, page tables, how a step's launches are made, which buffers the runner writes -- is the
task's rank_model.py (class RankModel, templates/rank_model.py.tmpl; worked example
references/examples/muse_spark_target_rank_model.py). This file owns what every monokernel
harness needs and what is easy to get subtly wrong (DESIGN 1.5 / 6):

  * topology: rank 0 is the calling process (the UT's leg worker: the driver's profiler,
    isolation and graph-capture gates see its launches); ranks 1..WORLD-1 are helper processes
    this module starts on the other leased devices (``python3 rank_harness.py --helper <rank>
    <port>``, one device each). A gloo group carries the runner's peer-buffer handshake and the
    per-call commands; all ranks run the same launches in the same order;
  * one step = the model's LAUNCHES launches on a path (RankModel.PATHS[0] the deployed one,
    the others the oracle paths cases.ORACLES runs), then end_step();
  * poisoning: before each launch the KV slots it writes (fp8 0x7F = e4m3 NaN), the routing ids
    (-7), the float buffers it produces (NaN; the inputs it reads from a previous launch only
    before launch 0); between calls every 32-bit word of the shared scratch / peer data regions
    (0x7F7F7F7F) except armed sentinel words and flag / trace / queue regions. Every cache is
    saved before the step and restored after it, so every call sees the same caches;
  * outputs, the family's contract order (every rank's, stacked by the caller): stream_out,
    attn_out, layer_out[launch], stream_in, kv_rows[layer, pool, K/V, row] (the step's rows of
    the first LAUNCHES layers' caches, decoded to float32; index l of the `launches` axis is the
    l-th layer's cache whichever launch writes it, so the paths line up), kv_stray[layer, pool]
    (bytes changed outside the step's target slots; layers past LAUNCHES, which nothing may
    write, are counted at [LAUNCHES - 1, their pool]), ids[launch] (routed_moe), and report-only
    diag_*;
  * graph replay (Replay, one static step a width) and the interleave sequence (graphs of two
    widths and eager steps back to back, each against an eager reference computed first);
  * rank_chain timing (Chain): a chain of distinct layers through the deployed path, with a
    parity check against call()'s path on the same case;
  * the JIT cache keyed by the bytes of the package this leg resolved (leg_env).
"""

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time

if __name__ == "__main__":
    # run as a script (the helper ranks): the task dir holds a unittest.py that shadows the
    # stdlib module torch imports -- drop it from the path first
    _TASK = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _TASK]

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_model():
    """The task's rank_model.py, by path (frozen task source; it imports the subject through the
    ambient overlay)."""
    path = os.path.join(_HERE, "rank_model.py")
    mod = sys.modules.get("rank_model")
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location("rank_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rank_model"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_model()
Model = M.RankModel
NP = int(Model.WORLD)
LAUNCHES = int(Model.LAUNCHES)
PATHS = tuple(Model.PATHS)
DEPLOYED = PATHS[0]
KV_POISON = 0x7F             # e4m3fn NaN
IDS_POISON = -7
# shared poison word: 0x7F7F7F7F is a finite fp32 / bf16-pair 3.4e38 (a stale partial a launch
# really uses blows the output up; one it weights by 0 stays 0) and a mailbox tag that is never
# current. Not NaN: a merge that multiplies the partial rows of partitions a short request does
# not have by 0 would turn 0 x NaN into a false failure (Muse Spark K-attn, v3-129 11-11).
SHARED_POISON = int(os.environ.get("MONO_UT_SHARED_POISON") or os.environ.get("MUSE_UT_SHARED_POISON")
                    or "0x7f7f7f7f", 0)   # < 0: none (diagnosis)
SHARED_POISON_SKIP = os.environ.get("MONO_UT_SHARED_POISON_SKIP") or os.environ.get("MUSE_UT_SHARED_POISON_SKIP")
SENT = -1                    # an armed sentinel word: never overwritten
ZERO_SENT = 0                # an armed zero-sentinel word: never overwritten either
# scratch / peer regions never poisoned: hand-off flags, stage traces, queue words
_KEEP = re.compile(getattr(Model, "KEEP_REGIONS", r"flag|trace|ugq"))


def poison_words(w, word):
    """Set the int32 words `w` to `word` in place, except armed sentinel words: all ones (a SENT
    region) and 0 (a zero-sentinel region). Either region's reader takes any other word as
    pushed, so poisoning an armed word hands it a partial nobody wrote (Muse Spark R9's k1_part /
    unp: G2 11-11)."""
    w.masked_fill_((w != SENT) & (w != ZERO_SENT), word)


def _floats(ts):
    return [t for t in ts if t is not None and t.is_floating_point()]


class _Capture:
    """Device buffers one step's outputs are copied into, stream-ordered (graph-safe)."""

    def __init__(self, model, rows, dev):
        bufs = model.buffers()
        self.stream_in = torch.empty(rows, Model.HIDDEN, device=dev, dtype=Model.STREAM_DTYPE)
        self.layer_out = torch.empty(LAUNCHES, rows, Model.HIDDEN, device=dev, dtype=Model.STREAM_DTYPE)
        self.attn_out = torch.empty(rows, Model.HIDDEN, device=dev, dtype=Model.INPUT_DTYPE)
        self.ids = (torch.full((LAUNCHES, rows, Model.TOP_K), IDS_POISON, device=dev, dtype=torch.int32)
                    if Model.TOP_K and bufs.get("ids") is not None else None)
        self.diag = {}
        for name, buf in (bufs.get("diag") or {}).items():
            if buf is None:
                continue
            if isinstance(buf, (tuple, list)):
                width = sum(t.shape[1] for t in buf)
                self.diag[name] = torch.empty(LAUNCHES, rows, width, device=dev, dtype=buf[0].dtype)
            else:
                self.diag[name] = torch.empty(LAUNCHES, rows, buf.shape[1], device=dev, dtype=buf.dtype)


class Rank:
    """One rank's model; eager steps on any path, static (graph) steps."""

    def __init__(self, rank, dev, group=None):
        import torch.distributed as dist

        self.rank, self.dev = rank, dev
        self.model = Model(rank, dev, group or dist.group.WORLD)
        self.caches = self.model.caches
        if len(self.caches) < LAUNCHES:
            raise RuntimeError("RankModel has %d layers' caches, fewer than LAUNCHES=%d" % (len(self.caches), LAUNCHES))
        self.static = None
        self._shared = {}

    # ------------------------------------------------------------------ poisoning
    def shared_regions(self, rows):
        """(poisoned, kept): the scratch / peer regions filled between calls, and the ones left
        alone with the reason. Layouts come from RankModel.shared_layouts (read from the
        subject's own modules); a buffer without one stays unpoisoned (reported)."""
        if rows in self._shared:
            return self._shared[rows]
        specs, kept = self.model.shared_layouts(rows)
        kept = list(kept)
        poisoned = []
        for name, buf, layout_fn in specs:
            if buf is None:
                kept.append({"region": name, "why": "the runner has no such buffer"})
                continue
            try:
                layout = layout_fn()
            except Exception as e:  # noqa: BLE001 - a candidate may move the layouts; report it
                kept.append({"region": name, "why": "layout raised %r" % e})
                continue
            end = 0
            for rname, (off, n) in sorted(layout.items(), key=lambda kv: kv[1][0]):
                end = max(end, off + n)
                if _KEEP.search(rname):
                    kept.append({"region": "%s.%s" % (name, rname), "why": "flag / trace / queue word"})
                elif off + n > buf.numel():
                    kept.append({"region": "%s.%s" % (name, rname), "why": "past the buffer (%d > %d)" % (off + n, buf.numel())})
                else:
                    poisoned.append((buf, off, n, "%s.%s" % (name, rname)))
            if end < buf.numel():
                kept.append({"region": "%s[%d:%d]" % (name, end, buf.numel()), "why": "outside the layout"})
        self._shared[rows] = (poisoned, kept)
        return poisoned, kept

    def poison_shared(self, rows):
        """Fill the peer buffer's data regions and the scratch (not their flags) between calls: a
        launch that reads a partial it did not wait for reads the poison, not the previous call's."""
        if SHARED_POISON < 0:
            return
        word = SHARED_POISON - (1 << 32) if SHARED_POISON >= 1 << 31 else SHARED_POISON
        for buf, off, n, name in self.shared_regions(rows)[0]:
            if SHARED_POISON_SKIP and re.search(SHARED_POISON_SKIP, name):
                continue
            poison_words(buf[off:off + n - n % 4].view(torch.int32), word)

    def _poison_launch(self, i, path, m, rows):
        """Before launch i: its cache targets (0x7F), the ids (-7) and the float buffers it
        produces (NaN); the buffers a launch reads from a previous one only before launch 0."""
        targets = self.model.kv_targets(m)
        for layer in self.model.writes(path, i):
            idx = targets[self.model.pool_of(layer)]
            for t in self.caches[layer]:
                t.view(torch.uint8).index_fill_(0, idx, KV_POISON)
        bufs = self.model.buffers()
        if bufs.get("ids") is not None:
            bufs["ids"][:rows].fill_(IDS_POISON)
        for t in _floats((bufs.get("produced") or {}).values()):
            t[:rows].fill_(float("nan"))
        if i == 0:
            for t in _floats((bufs.get("inputs") or {}).values()):
                t[:rows].fill_(float("nan"))

    # ------------------------------------------------------------------ one step (graph-safe)
    def _launches(self, rows, path, m, fb, stream, x, cap):
        mdl = self.model
        cap.stream_in.copy_(stream)
        for i in range(LAUNCHES):
            self._poison_launch(i, path, m, rows)
            stream, x = mdl.launch(i, rows, m, fb, stream, x)
            cap.layer_out[i].copy_(stream)
            bufs = mdl.buffers()
            if cap.ids is not None:
                cap.ids[i].copy_(bufs["ids"][:rows])
            for name, buf in (bufs.get("diag") or {}).items():
                if name not in cap.diag:
                    continue
                if isinstance(buf, (tuple, list)):
                    torch.cat([t[:rows] for t in buf], dim=1, out=cap.diag[name][i])
                else:
                    cap.diag[name][i].copy_(buf[:rows])
        mdl.end_step()
        cap.attn_out.copy_(x)

    def _snapshot(self):
        return [tuple(t.clone() for t in kv) for kv in self.caches]

    def _restore(self, snap):
        for kv, kvs in zip(self.caches, snap):
            for t, s in zip(kv, kvs):
                t.copy_(s)

    def _collect(self, rows, m, cap, snap):
        """The step's outputs on the CPU (call after a synchronize)."""
        kvw = Model.KV_WIDTH
        pools = int(getattr(Model, "POOLS", 2))
        targets = self.model.kv_targets(m)
        kv_rows = torch.zeros(LAUNCHES, pools, 2, rows, kvw, dtype=torch.float32)
        kv_stray = torch.zeros(LAUNCHES, pools, dtype=torch.int64)
        for layer, (kv, kvs) in enumerate(zip(self.caches, snap)):
            p = self.model.pool_of(layer)
            idx = targets[p]
            written = layer < LAUNCHES
            for j, (t, s) in enumerate(zip(kv, kvs)):
                changed = t.view(torch.uint8) != s.view(torch.uint8)
                if written:
                    kv_rows[layer, p, j] = t[idx].float().reshape(rows, kvw).cpu()
                    changed[idx] = False
                kv_stray[min(layer, LAUNCHES - 1), p] += int(changed.sum())
        out = {"stream_out": cap.layer_out[LAUNCHES - 1].cpu(), "attn_out": cap.attn_out.cpu(),
               "layer_out": cap.layer_out.cpu(), "stream_in": cap.stream_in.cpu(),
               "kv_rows": kv_rows, "kv_stray": kv_stray}
        if Model.TOP_K:
            out["ids"] = (cap.ids.cpu() if cap.ids is not None
                          else torch.full((LAUNCHES, rows, Model.TOP_K), IDS_POISON, dtype=torch.int32))
        for name, t in cap.diag.items():
            out["diag_" + name] = t.cpu()
        return out

    # ------------------------------------------------------------------ eager step
    def step(self, rows, context, stream, attn_input, path=DEPLOYED):
        """One step on `path`, every rank in lockstep (the barrier follows the poisoning)."""
        import torch.distributed as dist

        if path not in PATHS:
            raise ValueError("path %r is not one of RankModel.PATHS %s" % (path, PATHS))
        torch.cuda.synchronize(self.dev)
        self.poison_shared(rows)
        mdl = self.model
        mdl.set_path(path)
        m = mdl.step_meta(rows, context)
        fb = mdl.bind(m)
        cap = _Capture(mdl, rows, self.dev)
        snap = self._snapshot()
        st = stream.to(self.dev).to(Model.STREAM_DTYPE).contiguous().clone()
        x = attn_input.to(self.dev).contiguous().clone()
        torch.cuda.synchronize(self.dev)
        dist.barrier()
        try:
            self._launches(rows, path, m, fb, st, x, cap)
            torch.cuda.synchronize(self.dev)
            out = self._collect(rows, m, cap, snap)
        finally:
            mdl.set_path(DEPLOYED)
            self._restore(snap)
            torch.cuda.synchronize(self.dev)
        return out

    # ------------------------------------------------------------------ graph replay
    def fill(self, rows, context, stream, attn_input, width, static=None):
        """Copy a case into the static buffers (allocated once, at the first fill), then a
        barrier: no rank may replay into a peer buffer another rank is still poisoning."""
        import torch.distributed as dist

        s = static if static is not None else self.static
        if s is None:
            s = self.static = _Static(self, rows, width)
        s.fill(rows, context, stream, attn_input)
        torch.cuda.synchronize(self.dev)
        dist.barrier()
        return s

    def run_static(self, static=None):
        (static or self.static).run()

    def read_static(self, static=None):
        return (static or self.static).read()

    # ------------------------------------------------------------------ interleave
    def interleave(self, plan, width_of):
        """DESIGN 1.5: one runner, no rebuild, graphs of two widths and eager steps in
        alternation. Three phases, so that the sequence itself is what gets tested:
          1. references: every step eagerly on the deployed path, in plan order, before any
             graph exists (each step restores the caches it wrote);
          2. capture: one graph per width (a warm launch, then the capture), caches restored;
          3. the sequence, back to back: 'g' replays the width's graph, 'e' runs eagerly,
             's' runs PATHS[1] (diagnosis). No reference step runs in between, so a width's
             graph runs right after the other width's, and an 'e' step runs on whatever
             runner / scratch / flag state the graph before it left.
        A state bug at a width switch then shows as the sequence step disagreeing with its
        reference, which was computed on state no graph had touched. Only the caches are put
        back between sequence steps (every case has its own random page table, so one case's
        KV writes would otherwise land in the next case's context). Returns
        [(reference, result)] of this rank, CPU tensors."""
        cases = []
        for st in plan:
            rows, context = int(st["rows"]), int(st["context"])
            stream, x = make_inputs(rows, int(st["seed"]))
            cases.append((rows, context, stream, x, st["mode"]))
        refs = [self.step(rows, context, stream, x) for rows, context, stream, x, _ in cases]
        statics = {}
        for rows, context, stream, x, mode in cases:
            if mode == "g" and rows not in statics:
                s = statics[rows] = _Static(self, rows, width_of[rows])
                self.fill(rows, context, stream, x, None, static=s)
                s.capture()
                self._restore_static(s)
        got = []
        try:
            for rows, context, stream, x, mode in cases:
                if mode == "g":
                    s = statics[rows]
                    self.fill(rows, context, stream, x, None, static=s)
                    s.graph.replay()
                    got.append(s.read())
                elif mode == "s":
                    got.append(self.step(rows, context, stream, x, PATHS[1]))
                else:
                    got.append(self.step(rows, context, stream, x))
        finally:
            for s in statics.values():
                s.release()
        return list(zip(refs, got))

    def _restore_static(self, s):
        """Put the caches back to the static's last fill (after a capture's warm launch)."""
        torch.cuda.synchronize(self.dev)
        self._restore(s.snap)
        torch.cuda.synchronize(self.dev)


class _Static:
    """A width's static step: meta, capture buffers, the cache snapshot of the current fill,
    and (once captured) its graph. Always the deployed path."""

    def __init__(self, rank, rows, width):
        self.R, self.rows, self.width = rank, rows, width
        self.m = None
        self.cap = _Capture(rank.model, rows, rank.dev)
        self.stream = torch.empty(rows, Model.HIDDEN, device=rank.dev, dtype=Model.STREAM_DTYPE)
        self.x = torch.empty(rows, Model.HIDDEN, device=rank.dev, dtype=Model.INPUT_DTYPE)
        self.snap = None
        self.graph = None

    def fill(self, rows, context, stream, attn_input):
        R = self.R
        torch.cuda.synchronize(R.dev)
        if self.snap is not None:
            R._restore(self.snap)
        R.poison_shared(rows)
        R.model.set_path(DEPLOYED)
        m = R.model.step_meta(rows, context, self.width)
        if self.m is None:
            self.m = {k: v.clone() for k, v in m.items()}
        shapes = {k: tuple(v.shape) for k, v in m.items()}
        want = {k: tuple(v.shape) for k, v in self.m.items()}
        if rows != self.rows or shapes != want:
            raise RuntimeError("replay case rows %d meta %s vs static %d %s: step_meta(width=...) must give "
                               "every case of a width the same shapes" % (rows, shapes, self.rows, want))
        for k in self.m:
            self.m[k].copy_(m[k])
        self.fb = R.model.bind(self.m)
        self.snap = R._snapshot()
        self.stream.copy_(stream.to(R.dev))
        self.x.copy_(attn_input.to(R.dev))

    def run(self):
        # kv_targets is evaluated in the step on the static meta, so a graph replay indexes the
        # slots of whatever case was filled last
        self.R._launches(self.rows, DEPLOYED, self.m, self.fb, self.stream, self.x, self.cap)

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.run()                       # warm (the kernels are built by the eager reference)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize(self.R.dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.run()

    def read(self):
        """The replayed step's outputs; the caches go back to their snapshot at once, so an
        eager step between two replays never reads the rows a replay wrote (the page tables of
        two cases share pages, a small SWA pool most of all: v3-129 smoke 11-11)."""
        torch.cuda.synchronize(self.R.dev)
        out = self.R._collect(self.rows, self.m, self.cap, self.snap)
        self.R._restore(self.snap)
        torch.cuda.synchronize(self.R.dev)
        return out

    def release(self):
        if self.snap is not None:
            self.R._restore(self.snap)
            torch.cuda.synchronize(self.R.dev)
            self.snap = None


def make_inputs(rows, seed):
    """An interleave step's own stream and attention input (CPU), from its seed."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    stream = torch.randn(rows, Model.HIDDEN, generator=g).to(Model.STREAM_DTYPE)
    x = torch.randn(rows, Model.HIDDEN, generator=g).to(Model.INPUT_DTYPE)
    return stream, x


# ----------------------------------------------------------------------------- the group
_GROUP = None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _package_root():
    """The bound package directory this process resolves, without importing it when it is not
    imported yet: leg_env() runs before the process_state snapshot, and the candidate package's
    import-time code must run after it, where the gate sees what it does."""
    pkg = sys.modules.get(Model.PACKAGE)
    if pkg is not None:
        return list(pkg.__path__)[0]
    spec = importlib.util.find_spec(Model.PACKAGE)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("%s does not resolve to a package on this leg" % Model.PACKAGE)
    return list(spec.submodule_search_locations)[0]


def source_digest():
    """sha256 over the .py bytes of the bound package this process resolved (the installed one
    in the baseline leg, kernel_src/<package> in the candidate's)."""
    import hashlib

    root = _package_root()
    h = hashlib.sha256()
    for d, dirs, files in sorted(os.walk(root)):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, root).encode())
                with open(p, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


CACHE_ENV = getattr(Model, "JIT_CACHE_ENV", "FLYDSL_RUNTIME_CACHE_DIR")


def _source_cache_dir():
    """A JIT disk-cache key that covers a launcher's own source but not the stage functions
    (or common helpers) it reaches through its launch context would load a stale binary for an
    edited stage. Key the cache directory by the bytes of the whole package this process
    resolved (the bind is the whole package, DESIGN 6.0)."""
    base = (os.environ.get("MONO_UT_JIT_CACHE_ROOT") or os.environ.get("MUSE_UT_FLYDSL_CACHE_ROOT")
            or os.path.join(_HERE, ".flydsl_cache"))
    return os.path.join(base, source_digest()[:20])


def leg_env():
    """meta.leg_env (shared/leg_runner.apply_leg_env): the leg's JIT cache directory.

    The JIT reads CACHE_ENV from os.environ at every build and has no other way to set it. Its
    value differs per leg by design (the digest of the source the leg resolved), so setting it
    while the leg runs reads as the candidate changing process state (G1b process_state FAIL).
    leg_runner calls this first, before the snapshot, so the value is part of both of the leg's
    snapshots; the helpers and rank_timing's workers inherit it."""
    if not CACHE_ENV:
        return {}
    return {CACHE_ENV: _source_cache_dir()}


def use_source_cache_dir():
    """Make sure the JIT builds into this leg's source-keyed cache; returns the directory.

    Under leg_runner meta.leg_env has already set the same value and this only creates the
    directory. A process started some other way (a smoke script) gets it set here, before its
    first build."""
    if not CACHE_ENV:
        return None
    cache = _source_cache_dir()
    os.makedirs(cache, exist_ok=True)
    if os.environ.get(CACHE_ENV) != cache:
        os.environ[CACHE_ENV] = cache
    return cache


def report(event, **fields):
    """One JSON line to run/harness_report.jsonl (rank 0 only): what the harness could not
    poison, the model's facts, the interleave results."""
    row = {"event": event, "pid": os.getpid(), "time": time.time(), **fields}
    try:
        os.makedirs(os.path.join(_HERE, "run"), exist_ok=True)
        with open(os.path.join(_HERE, "run", "harness_report.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except OSError:
        pass


def _visible():
    """The leased devices' logical indices, rank 0's first."""
    pinned = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    me = int(pinned.split(",")[0]) if pinned and "," not in pinned else 0
    n = int(os.environ.get("MONO_UT_WORLD_DEVICES") or os.environ.get("MUSE_UT_WORLD_DEVICES") or NP)
    return [me] + [d for d in range(n) if d != me][: NP - 1]


class Group:
    """Rank 0 here, ranks 1..NP-1 as helpers; a gloo group for the handshake and commands."""

    def __init__(self):
        import torch.distributed as dist

        use_source_cache_dir()   # before any build; the helpers inherit it (normally set by meta.leg_env)
        devs = _visible()
        if len(devs) < NP:
            raise RuntimeError("this task needs %d devices, sees %s" % (NP, devs))
        timeout = __import__("datetime").timedelta(seconds=1800)
        # The rendezvous store is created HERE on port 0: the OS picks a free port and the store
        # keeps it bound, so no other process can take it between choosing and binding (a
        # _free_port() probe that closes its socket before gloo binds lost that race once in a
        # cold verification: EADDRINUSE in one negative-control leg, the control read as uncaught,
        # 11-12 geak_c6 draft rep1). The helpers connect to the port the store reports.
        store = dist.TCPStore("127.0.0.1", 0, NP, True, timeout=timeout, wait_for_workers=False)
        port = int(store.port)
        env = {k: v for k, v in os.environ.items()}
        self.procs = []
        for r in range(1, NP):
            e = dict(env, HIP_VISIBLE_DEVICES=str(devs[r]), CUDA_VISIBLE_DEVICES=str(devs[r]))
            self.procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--helper", str(r), str(port)],
                                               env=e, stdout=sys.stderr, stderr=sys.stderr))
        dist.init_process_group("gloo", store=store, rank=0, world_size=NP, timeout=timeout)
        self._store = store   # keep the listening store alive for the group's lifetime
        self.dist = dist
        self.rank = Rank(0, torch.device("cuda", torch.cuda.current_device()))
        self._sync()
        poisoned, kept = self.rank.shared_regions(8)
        report("group", source_root=_package_root(), source_digest=source_digest()[:20],
               model=self.rank.model.facts(), unpoisoned=kept,
               poisoned=[name for _, _, _, name in poisoned], notes=list(getattr(Model, "NOTES", [])))

    def _sync(self):
        torch.cuda.synchronize()
        self.dist.barrier()

    def cmd(self, *msg):
        obj = [msg]
        self.dist.broadcast_object_list(obj, src=0)

    def gather(self, mine):
        out = [None] * NP
        self.dist.gather_object(mine, out, dst=0)
        return out


def group():
    global _GROUP
    if _GROUP is None:
        _GROUP = Group()
    return _GROUP


def _stack(outs):
    """Every rank's dict -> one dict of [world, ...] tensors."""
    return {k: torch.stack([o[k] for o in outs]) for k in outs[0]}


def call_ranks(rows, context, stream, attn_input, path=DEPLOYED):
    """One step on all ranks on `path`; every rank's outputs, stacked, on the operands' device."""
    G = group()
    sc, xc = stream.detach().cpu(), attn_input.detach().cpu()
    G.cmd("step", rows, context, sc, xc, path)
    mine = G.rank.step(rows, context, sc, xc, path)
    outs = G.gather(None)
    outs[0] = mine
    dev = stream.device
    return {k: v.to(dev) for k, v in _stack(outs).items()}


class Replay:
    """The replay bundle's rank side: rank 0 fills / runs / reads its own static buffers in
    the driver's process; the helpers mirror every fill and, once they hold a graph, replay
    it right after the fill (the driver then replays rank 0's graph; the in-kernel
    exchanges pair the ranks)."""

    def __init__(self, width):
        self.G = group()
        self.width = width
        self.captured = False

    def fill(self, rows, context, stream, attn_input):
        sc, xc = stream.detach().cpu(), attn_input.detach().cpu()
        self.G.cmd("fill", rows, context, sc, xc, self.width, self.captured)
        self.G.rank.fill(rows, context, sc, xc, self.width)

    def run(self):
        if torch.cuda.is_current_stream_capturing():
            self.G.cmd("capture")
            self.G.rank.run_static()
            self.captured = True
        else:
            self.G.cmd("run")
            self.G.rank.run_static()

    def read(self):
        self.G.cmd("read")
        mine = self.G.rank.read_static()
        outs = self.G.gather(None)
        outs[0] = mine
        return _stack(outs)


# ----------------------------------------------------------------------------- interleave
GATED = ("stream_out", "attn_out", "layer_out", "stream_in", "kv_rows", "kv_stray", "ids")


def _close(name, ref, got):
    """Graph / eager agreement of one gated output: floats within 1/128 (kv rows: one e4m3
    code), both sides finite; kv_stray 0 on both; ids exact."""
    if name == "kv_stray":
        return {"ok": bool((ref == 0).all() and (got == 0).all()), "max": int(max(ref.max(), got.max()))}
    if name == "ids":
        return {"ok": bool(torch.equal(ref, got)), "mismatches": int((ref != got).sum())}
    r, g = ref.double(), got.double()
    finite = bool(torch.isfinite(r).all() and torch.isfinite(g).all())
    atol, rtol = (1 / 64, 1 / 8) if name == "kv_rows" else (1 / 128, 1 / 128)
    err = (r - g).abs()
    lim = atol + rtol * r.abs()
    return {"ok": finite and bool((err <= lim).all()), "finite": finite,
            "max_abs": float(err[torch.isfinite(err)].max()) if bool(torch.isfinite(err).any()) else None}


def _brief(v):
    """One failed output's numbers for the gate's why."""
    return "(" + ", ".join("%s=%s" % (k, ("%.4g" % x) if isinstance(x, float) else x)
                           for k, x in v.items() if k != "ok") + ")"


def interleave_plan(small_contexts, large_contexts, small_rows=8, large_rows=16, seed=4242):
    """small graph -> large graph -> small eager -> large graph -> small graph (5 steps, every
    one with its own inputs and context), run back to back after every step's reference
    (Rank.interleave). small / large are the two served widths (rows)."""
    modes = [(small_rows, "g"), (large_rows, "g"), (small_rows, "e"), (large_rows, "g"), (small_rows, "g")]
    ctx = {small_rows: list(small_contexts), large_rows: list(large_contexts)}
    used = {small_rows: 0, large_rows: 0}
    plan = []
    for k, (rows, mode) in enumerate(modes):
        c = ctx[rows][used[rows] % len(ctx[rows])]
        used[rows] += 1
        plan.append({"rows": rows, "context": int(c), "mode": mode, "seed": seed + 31 * k})
    return plan


def run_interleave(plan):
    """Run `plan` on every rank (one runner each, no rebuild) and compare each step's result
    with its eager reference (computed before the sequence) on every rank. Returns the replay
    gate's extra case: {"ok", "steps", "order", and "why" naming the failed steps / outputs}."""
    G = group()
    width_of = {}
    for st in plan:
        w = Model.page_width(int(st["context"])) if hasattr(Model, "page_width") else (int(st["context"]) + 15) // 16
        width_of[st["rows"]] = max(width_of.get(st["rows"], 0), w)
    G.cmd("interleave", plan, width_of)
    mine = G.rank.interleave(plan, width_of)
    outs = G.gather(None)
    outs[0] = mine
    steps, ok = [], True
    for k, st in enumerate(plan):
        ref = _stack([o[k][0] for o in outs])
        got = _stack([o[k][1] for o in outs])
        res = {}
        for name in GATED:
            if name not in ref:
                continue
            if name in ("stream_out", "layer_out"):
                d_ref = ref[name] - (ref["stream_in"] if name == "stream_out" else ref["stream_in"][:, None])
                d_got = got[name] - (got["stream_in"] if name == "stream_out" else got["stream_in"][:, None])
                res[name] = _close(name, d_ref, d_got)
            else:
                res[name] = _close(name, ref[name], got[name])
        step_ok = all(v["ok"] for v in res.values())
        ok = ok and step_ok
        steps.append({"step": k, **st, "ok": step_ok, "outputs": res})
    out = {"ok": ok, "steps": steps,
           "order": "references (eager, before any graph): %s; sequence (back to back): %s"
                    % (" ".join("e%d" % s["rows"] for s in plan), " ".join("%s%d" % (s["mode"], s["rows"]) for s in plan))}
    if not ok:
        out["why"] = "; ".join(
            "step %d (%s%d ctx %d): %s" % (s["step"], s["mode"], s["rows"], s["context"],
                                           ", ".join("%s %s" % (n, _brief(v)) for n, v in s["outputs"].items() if not v["ok"]))
            for s in steps if not s["ok"])
    report("interleave", **out)
    return out


# ----------------------------------------------------------------------------- rank_chain
PARITY_TOL = 1.0 / 128


class Chain:
    """rank_chain timing (meta.timing.kind rank_chain, shared/rank_timing.py) of one rank: a
    chain of distinct layers through the deployed path at the case's rows and context.

    length = chain_length + 1 launches, short = 1, so (T[step(length)] - T[step(short)]) /
    chain_length removes the bootstrap and the first launch and leaves chain_length launches,
    each on its own layer's weights and KV cache. The model is built with n_layers = length + 1
    so that every timed launch has a next layer (RankModel decides what that layer is). step()
    only enqueues launches.

    check() (DESIGN 6.1): the first call also runs the call() path once -- a Rank on the same
    case and operands, whose first layers RankModel builds identical to the chain's -- and
    compares the chain's step(1) stream with that path's layer_out[0], within 1/128 of the
    step's change. A chain that times something other than what call() checks fails here."""

    def __init__(self, rank, world, group, case, args, chain_length):
        # the JIT cache must be keyed by the source this leg resolved; meta.leg_env set it when
        # the leg started and this worker inherited it (this only creates the directory)
        use_source_cache_dir()
        if int(world) != NP:
            raise RuntimeError("rank_chain world %s != RankModel.WORLD %d" % (world, NP))
        self.rank, self.group = rank, group
        self.length = chain_length + 1
        self.short = 1
        axes = case.get("axes") or {}
        self.rows, self.context = int(axes["rows"]), int(axes["context"])
        self.sig = case.get("sig")
        dev = self.dev = torch.device("cuda", torch.cuda.current_device())
        self.model = Model(rank, dev, group, n_layers=self.length + 1)
        self.model.set_path(DEPLOYED)
        self.meta = self.model.step_meta(self.rows, self.context)
        self.fb = self.model.bind(self.meta)
        # the case's operands (cases.make_args): the same ones call() gets for this case
        self.stream0 = args["stream"].to(dev).to(Model.STREAM_DTYPE).contiguous()
        self.x0 = args["attn_input"].to(dev).contiguous()
        self.stream = torch.empty_like(self.stream0)
        self.x = torch.empty_like(self.x0)
        self.parity = None

    def step(self, n):
        self.stream.copy_(self.stream0)          # the launch updates the stream in place
        self.x.copy_(self.x0)
        st, x = self.stream, self.x
        for i in range(n):
            st, x = self.model.launch(i, self.rows, self.meta, self.fb, st, x)
        self.model.end_step()

    def _guard_words(self):
        why = self.model.device_errors()
        if why:
            raise RuntimeError("rank_chain %s rank %d: device wait failed: %s" % (self.sig, self.rank, why))

    def _parity(self):
        """call()'s path once on this case (every rank, collective) against the chain's step(1)."""
        ref = Rank(self.rank, self.dev, self.group)
        try:
            out = ref.step(self.rows, self.context, self.stream0.cpu(), self.x0.cpu(), DEPLOYED)
        finally:
            torch.cuda.synchronize(self.dev)
            ref.model.close()
            del ref
            torch.cuda.empty_cache()
        self.step(1)
        torch.cuda.synchronize(self.dev)
        self._guard_words()
        s_in = self.stream0.double().cpu()
        got = self.stream.double().cpu() - s_in
        want = out["layer_out"][0].double() - s_in
        err = (got - want).abs()
        lim = PARITY_TOL + PARITY_TOL * want.abs()
        ok = bool(torch.isfinite(got).all() and torch.isfinite(want).all() and (err <= lim).all())
        self.parity = {"ok": ok, "max_abs": float(err.max()), "bitwise": bool(torch.equal(got, want))}
        print("[rank_chain] %s rank %d parity with call(): %s" % (self.sig, self.rank, self.parity),
              file=sys.stderr, flush=True)
        if not ok:
            raise RuntimeError("rank_chain %s rank %d: step(1) differs from call()'s layer_out[0] "
                               "(max |d| %.3g, tol 1/128 of the change): the chain does not time what call() checks"
                               % (self.sig, self.rank, self.parity["max_abs"]))

    def check(self):
        self._guard_words()
        if self.parity is None:
            self._parity()

    def close(self):
        self.model.close()


# ----------------------------------------------------------------------------- helpers
def _helper(rank, port):
    import datetime

    import torch.distributed as dist

    torch.cuda.set_device(0)
    timeout = datetime.timedelta(seconds=1800)
    # a client of rank 0's store (Group.__init__), the same store-based path rank 0 takes
    store = dist.TCPStore("127.0.0.1", port, NP, False, timeout=timeout)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=NP, timeout=timeout)
    me = Rank(rank, torch.device("cuda", 0))
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    while True:
        obj = [None]
        dist.broadcast_object_list(obj, src=0)
        msg = obj[0]
        op = msg[0]
        if op == "step":
            out = me.step(*msg[1:])
            dist.gather_object(out, None, dst=0)
        elif op == "fill":
            rows, context, sc, xc, width, replay = msg[1:]
            me.fill(rows, context, sc, xc, width)
            if replay and graph is not None:
                graph.replay()
        elif op == "run":
            me.run_static()
        elif op == "capture":
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                me.run_static()
        elif op == "read":
            out = me.read_static()
            dist.gather_object(out, None, dst=0)
        elif op == "interleave":
            out = me.interleave(*msg[1:])
            dist.gather_object(out, None, dst=0)
        elif op == "exit":
            break


def _smoke(argv):
    """python3 rank_harness.py <rows> <context> [interleave <small ctx,..> <large ctx,..> | @plan.json]
    (rank 0 here, helpers spawned): every path once (the deployed one twice), per-output
    finiteness / repeatability / deployed-vs-oracle, the model's device error words."""
    rows, context = int(argv[1]), int(argv[2])
    torch.cuda.set_device(0)
    t0 = time.time()
    G = group()
    print("group+build", round(time.time() - t0, 1), flush=True)
    st, x = make_inputs(rows, 1)
    res = {}
    for path in (DEPLOYED,) + PATHS:
        t = time.time()
        res.setdefault(path, []).append(call_ranks(rows, context, st, x, path))
        print("step", path, round(time.time() - t, 2), flush=True)
    a, a2 = res[DEPLOYED][0], res[DEPLOYED][1]
    for other in PATHS[1:]:
        s = res[other][0]
        for name in GATED:
            if name not in a:
                continue
            fa = a[name].double() if a[name].is_floating_point() else a[name]
            fs = s[name].double() if s[name].is_floating_point() else s[name]
            line = {"vs": other,
                    "finite": bool(torch.isfinite(fa).all()) if fa.is_floating_point() else None,
                    "rep_equal": bool(torch.equal(a[name], a2[name])) if not a[name].is_floating_point()
                    else bool(((a[name] == a2[name]) | (a[name].isnan() & a2[name].isnan())).all()),
                    "max_diff": float((fa - fs).abs().max()) if fa.is_floating_point() else int((fa != fs).sum())}
            if name == "kv_stray":
                line[DEPLOYED], line[other] = a[name].sum().item(), s[name].sum().item()
            if name == "ids":
                line["poison_left"] = int((a[name] == IDS_POISON).sum())
            print(name, json.dumps(line), flush=True)
    print("device_errors", G.rank.model.device_errors(), "facts", json.dumps(G.rank.model.facts(), default=str), flush=True)
    if len(argv) > 3 and (argv[3] == "interleave" or argv[3].startswith("@")):
        if argv[3] == "interleave":
            small = [int(c) for c in argv[4].split(",")]
            large = [int(c) for c in argv[5].split(",")]
            plan = interleave_plan(small, large)
        else:
            plan = [{"seed": 9000 + 7 * k, **s_} for k, s_ in enumerate(json.load(open(argv[3][1:])))]
        r = run_interleave(plan)
        print("interleave", r["ok"], json.dumps([{k: v for k, v in s_.items() if k != "outputs"} for s_ in r["steps"]]),
              flush=True)
        for s_ in r["steps"]:
            if not s_["ok"]:
                print("  step", s_["step"], json.dumps(s_["outputs"]), flush=True)
    G.cmd("exit")
    print("total", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--helper":
        try:
            _helper(int(sys.argv[2]), int(sys.argv[3]))
        finally:
            os._exit(0)
    try:
        _smoke(sys.argv)
    finally:
        os._exit(0)
