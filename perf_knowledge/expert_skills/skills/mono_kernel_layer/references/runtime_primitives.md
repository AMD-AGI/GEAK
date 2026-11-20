# Runtime primitives for persistent kernels (`mk_runtime`, FlyDSL on ROCm)

The synchronization and memory primitives a mono kernel is built on: a device-side launch epoch, bounded
waits, a sticky error word shared across ranks, mailboxes, grid and cross-rank barriers, the in-kernel TP
all-reduce, and the host-side peer group that creates the IPC-shared buffers.

Every mono kernel synchronizes through these primitives. Each one is bounded by a deadline, shares a
sticky error word across ranks, and derives its tags from the launch epoch, which keeps graph replay
correct.

**Where the library comes from.** `mk_runtime` is a separate runtime library; it is not vendored in this
repository. A task that uses it carries a copy as `kernel_src/mk_runtime/`, and that copy stays
unmodified ([acceptance_gates.md](acceptance_gates.md#hard-gates), gate 4). Cross-GPU and cross-workgroup
data and waits go through these primitives only
([stage_authoring.md](stage_authoring.md#shape-of-the-code-hard-gates), rule 5).

## Library layout

| module | runs on | what |
|---|---|---|
| `abi.py` | pure Python | Control-region layout and offsets, error codes, `grid_target`, `pack`/`unpack`, `decode_error`. |
| `device.py` | FlyDSL IR (inside `@flyc.kernel`) | Atomics, waits, mailboxes, barriers, all-reduce. |
| `host.py` | torch + HIP (ctypes) | `PeerGroup`: control regions and IPC-shared peer buffers for one TP group. |
| `selftest_kernels.py` | FlyDSL | Reference kernels (`build_chain`, `build_orphan_wait`, `build_rank_barrier`, `build_allreduce`). Copy their structure; the launcher pattern below follows `build_allreduce`. |

## Control region (`abi.py`)

One per rank, allocated **uncached** (`hipExtMallocWithFlags(..., uncached)`) and IPC-exported to the
peers. Every multi-writer slot sits on its own 128-byte line. The region holds: header, sticky
`error`, `launches` (= the launch epoch), `MAX_RANKS` rank slots, `MAX_XCDS` release words, `grid`,
`grid_done`, `xrank_done`, then `n_mailboxes` 8-byte tagged mailboxes.

## Device API (`device.py`; all addresses are raw i64)

- **Scopes.** `SYSTEM` (cross-GPU hand-offs, the default) and `AGENT` (between workgroups of one GPU).
  `store_release_u64`, `load_acquire_u64`, `load_relaxed_u64`, `atomic_add_u64`, `atomic_or_u64`,
  `fence_release`, `fence_acquire`.
- **Time.** `realtime()` (`s_memrealtime`, 100 MHz), `deadline_after(ticks)`, and `ticks_for_us(us)`
  (host side, pass the result in as an `Int64` argument).
- **Launch epoch.**
  - `seq = launch_begin(region)` at kernel entry, identical in every workgroup.
  - `launch_end(region, seq, n_blocks, tid)` at exit: the last workgroup publishes `launches = seq`.
  - Do not also call `launch_count` (it is only for kernels without an epoch).
  - Every tag below is derived from `seq`, so a replay with fixed arguments gets fresh tags.
- **Waits.**
  - `wait_until(addr, accept, *, deadline, region)` returns `(word, ok)`. It stops on accept, on the
    deadline, or once the local error word is non-zero.
  - `wait_seq(addr, seq, ...)` waits until the value is `>= seq`.
  - If `ok` is false, skip the dependent work; `abort_launch()` traps instead, if a hard failure is
    preferred.
- **Errors.**
  - `report_unless(ok, *, region, peer_regions, rank, site, cause=ERR_TIMEOUT)` records the first error
    in the local region and in every peer's, so all ranks stop.
  - `site` is a small integer that is unique per wait site; it is decoded on the host.
  - `error_code`, `raise_error`, `error_word`, `healthy(region)`.
- **Mailboxes.**
  - `mailbox_put(region_or_peer, index, seq, payload)`: one 64-bit release store of the tag (`seq`) and
    a 32-bit payload. The producer writes its data first.
  - `mailbox_get(region, index, seq, *, deadline)` returns `(payload_i32, ok)`. It polls only that
    mailbox and sees the data once `ok` is set.
  - Mailboxes are per destination region: to hand off to a peer, `put` into the peer's region.
- **Grid barrier.**
  - `grid_barrier(region, seq, *, phase, per_launch, n_blocks, tid, deadline, rank, peer_regions,
    site, scope=AGENT)` runs over all `n_blocks` workgroups. They must all be co-resident, i.e. at most
    one persistent workgroup per CU.
  - `phase` numbers this launch's barriers, `0..per_launch-1`.
  - Use `scope=SYSTEM` when peers will read what was written before the barrier.
- **Cross-rank barrier.**
  - `xrank_barrier(region, regions_dev, seq, *, index, per_launch, n_ranks, rank, tid, bid, deadline,
    site)` runs after a `grid_barrier(scope=SYSTEM)`. Workgroup 0 meets the peers (`rank_barrier`)
    and the other workgroups wait on `xrank_done`.
  - `regions_dev` is the device array of every rank's region address (`PeerGroup.region_ptrs_dev`).
- **TP all-reduce.**
  - `allreduce_sum(*, region, regions_dev, partials_dev, out, n_elems, dtype, seq, n_ranks, rank, tid,
    bid, threads, n_blocks, grid_phase, grid_per_launch, xrank_index, xrank_per_launch, deadline,
    site, slot_elems=0)`. It uses one grid barrier and one cross-rank barrier, then sums the partials of
    all ranks in f32, in rank order. Every rank gets **bit-identical** output, equal to RCCL's for TP2
    in the tests.
  - Before calling it, write this rank's partial into
    `partial_slot(partials[rank], seq, slot_elems or n_elems, dtype)`, a double-buffered half chosen by
    `seq & 1`.
  - **`slot_elems` is the half's capacity.** Every width that shares one partial buffer must pass the
    same `slot_elems` (normally max width × hidden), or the halves of different widths overlap. Size the
    buffer as `2 * slot_elems * elem_bytes`.
- **Data.** `load_elem`, `store_elem`, `to_f32`, `from_f32` (bf16/f16/f32), `addr(base, off)`,
  `grid_stride(start, stop, step, body)`, `when(cond)`, `not_`, `c_i64`, `c_i32`, `as_i64`, `as_i32`.

## Host API (`host.py`)

```python
g = host.PeerGroup(rank=r, world_size=tp, device=dev, n_mailboxes=256,
                   buffers={"partial": 2 * slot_elems * 2}, group=None)  # all_gather over `group`
g.region                         # this rank's control region (i64)
g.region_ptrs_dev                # int64 device tensor: every rank's region (pass .data_ptr())
g.buffers["partial"]             # this rank's buffer (torch uint8)
g.buffer_ptrs_dev["partial"]     # int64 device tensor: every rank's buffer address
g.error(); g.raise_if_error()    # decode the sticky error word (host sync; never per step)
g.launches()                     # launch epoch / counter (host sync)
g.reset(); g.close()             # tests/teardown only
```

- Create it **outside any graph capture**, on the first eligible call, collectively on every rank of the
  TP group.
- The IPC export uses `hipMemGetAddressRange` to find the allocation base, because the torch caching
  allocator sub-allocates. The peer address is `opened_base + offset`. This was a real bug: wrong
  results with no error reported. Do not "simplify" it.
- `PeerGroup` uses `dist.all_gather_object`. Pass the framework's TP CPU group as `group` when its
  default group is not the TP group.

## Launcher pattern

```python
@flyc.kernel(known_block_size=[THREADS, 1, 1])
def mk_layer(rank: Int32, region: Int64, regions: Int64, ..., timeout_ticks: Int64):
    tid, bid = fx.thread_idx.x, fx.block_idx.x
    seq = d.launch_begin(region)
    deadline = d.deadline_after(timeout_ticks)
    ...  # stages; hand-offs via mailbox_put/get; barriers with distinct `site`s
    d.launch_end(region, seq, N_BLOCKS, tid)

@flyc.jit
def run(rank: Int32, region: Int64, ..., stream: Stream = Stream(None)):
    mk_layer(rank, region, ...).launch(grid=(N_BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)
run.func.__name__ = f"mk_<kind>_w{width}_r{n_ranks}_..."   # unique per configuration
```

Call it with `stream=torch.cuda.current_stream()`, and pass addresses as `Int64(t.data_ptr())`.

## Pitfalls seen so far

- **RCCL under graph capture.** The NCCL watchdog queries events of collectives issued before the
  capture, and HIP rejects that even in `thread_local` mode. The oracle harness therefore settles for
  0.3 s after warmup (a per-task setting) and agrees on capture success across ranks. Native-path
  collectives in your adapter need the same treatment.
- **Divergent fallbacks.** If one rank takes the native path while another launches the mono kernel,
  both ranks hang until the deadline. Base the decision only on data every rank shares (shape, dtype,
  flags).
- **The FlyDSL compile cache** does not see changes in helper modules (e.g. editing `mk_runtime`). Bump
  the launcher name or clear the cache after such an edit.
- **Loop-carried values in FlyDSL ≥ 0.4.** A Python `for` / `if` inside a `@flyc.kernel` body becomes
  `scf.for` / `scf.if`, and values carried through it come back as DSL wrappers (`Int64`, ...) without
  `.type`. Pass them through `d.as_i64` / `d.as_i32` before handing them to raw `arith.*Op` builders;
  the `device.py` helpers already unwrap their arguments.
