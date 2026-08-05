# Entry modes: which gates apply, and which harness shape

A deep-dig run starts in one of three states. The gates are **not** the same in all three, and
applying the port's gates to an in-place run — or the in-place defaults to a port — is a known way to
lose a run before the first real round. Decide the mode first; everything below follows from it.

| | **A. Port** | **B. In-place** | **C. Re-entry** |
| --- | --- | --- | --- |
| source at entry | tuned plain Triton | already explicit-tile, already measured | a checkpointed variant |
| comparator | `champion_ms` from the bundle | the **incumbent**, measured and asserted | whatever the interrupted run used |
| transcription debt | **YES, by construction** | none | already paid, or carried unpaid |
| G1 `champion_gate.py` | REQUIRED, on `plain_champion.json` | REQUIRED, on an incumbent bundle | RE-ASSERT on resume |
| G2 `parity_gate.py` | REQUIRED before the first climb | **DOES NOT APPLY** as a gate | only if the debt was carried unpaid |
| G3 `probe.py measure` | every anchor and every staging change | unchanged | unchanged |
| G4 `ab_bench.py` | unchanged | unchanged | unchanged |
| harness loop shape | **PORT** | **ORDINARY optimize** | as the interrupted run |
| round 1 outcome | expected **below** the comparator | expected at-or-above | continues the trajectory |

---

## A. Port — plain Triton to an explicit-tile DSL

The mode `champion-handoff.md` is written for. The two things that decide it:

**The debt is real and it is not optional to attribute it.** A faithful anchor lands below the
comparator by construction; `parity_gate.py` splits the gap across `lost_pipeline` / `lost_layout` /
`lost_RA` from the compiled artifacts and exits 2 while it is unpaid. Climbing from an unpaid anchor
caps the port: the best lever found gets quoted against a broken starting point, and the run closes
below the champion while reporting a healthy-looking gain against its own anchor. Both numbers
(`vs_anchor`, `vs_champion`) travel with every result, and the champion one decides.

**Before accepting "the plain tier is finished", spot-check the pin at ±1 grid step on each swept
axis.** A sweep's own report that its winner survived is not evidence about points it never tested.
This is cheap, it is not what `champion_gate.py`'s SAMPLING check can see (that check reads the
bundle's own `partially_sampled` claim), and it has caught a **6.1%** plain win sitting one grid point
outside the swept range on a kernel whose tier log claimed a completed re-sweep. A port that starts
one grid point short of the real champion inherits that error as a fake Gluon gain.

**Harness shape.** Any harness with loop control needs telling this is a port, because its defaults
are tuned for ordinary optimization where a candidate under the baseline is worthless and a
non-improving round really is a stall. On a port those same defaults delete the recovery phase: the
transcription round produces no candidate, so no patch is saved, nothing is verified, the round has no
winner, and the loop stops two rounds into a port that is working exactly as designed. What a port
needs instead:

- **candidate floor below 1.0** — low enough for *your measured anchor*, not for a guess. Naive
  anchors between 0.5× and 0.7× are ordinary and one at **0.51×** has been observed, which is a bad
  window away from falling out of a 0.5 floor.
- **progress delta negative** — an experiment that costs ground is information on a port.
- **no-improve tolerance ≥ the longest non-improving streak you expect.** Measured on `pa_decode`:
  wins at rounds 1, 8, 8, 9, 10 with **five consecutive** non-improving rounds between them. A
  tolerance of 4 ends that run one round before the payoff.
- **budget counted in the right unit.** If the harness counts directions rather than rounds, and a
  deep direction costs more than one, the round count is budget/cost — not budget.

---

## B. In-place — the source is already explicit-tile

Nothing is transcribed, so there is no anchor and no debt. Three consequences, and the second is the
one that bites.

**`parity_gate.py` must not be run as a gate.** There is no `anchor_ms`. Passing the incumbent as
both sides yields ratio 1.00 → CLEARED, which is *true and vacuous*: it records a gate as satisfied
that was never applicable, which is worse than not running it. Use the **tool** freely as a
diagnostic — `--champion-ms <incumbent> --anchor-ms <a variant that regressed>` is exactly the right
way to read a regression you introduced mid-run, because `lost_layout` / `lost_RA` /
`lost_pipeline` are the right vocabulary for it either way — but say which you are doing, gate or
diagnostic, in the round log.

**The two-comparator discipline collapses to one, and that removes a safety net.** On a port,
carrying `vs_anchor` beside `vs_champion` is what stops you from selling a recovery as a win. In-place
there is only `vs_incumbent`, so nothing structurally reminds you that the denominator has to be
honest. Therefore the incumbent must be a **measured, asserted** number — not "the file I started
from", and not a figure inherited from another box or another container. Assert it with G1 against an
incumbent bundle (below), and re-measure it in your own first window.

**The harness must be in ORDINARY optimize shape.** Setting the port knobs here is the mirror-image
mistake: a candidate floor below 1.0 and a negative progress delta keep a genuinely stalled search
alive, burning the budget that the port shape exists to protect.

### The incumbent bundle: same schema, different meaning per field

`champion_gate.py` keys on `schema: plain_champion` and every check still means something with the
explicit-tile source as `kernel` — but two fields change meaning and one is where an in-place run is
most likely to be fooling itself:

| field | in-place reading |
| --- | --- |
| `SOURCE` / `LIVE` | unchanged, and still the pair that fails silently most often — an incumbent edited since it was measured is not the thing that was measured |
| `CONFIG` | still "the dump came from this source at the pinned launch config". Check it against the **explicit-tile** dump, not a plain one |
| `COMPARATOR` | `incumbent_ms <= default_ms`. If the kernel has no meaningful default, say so — the "not a strawman" claim is then unprovable, not proven |
| **`SAMPLING`** | **the highest-value check in this mode.** An explicit-tile kernel's tile is usually *inherited from the plain champion and never re-swept in the explicit DSL*, where the tile is coupled to the layout family rather than a free knob. A bundle that reports a completed sweep may be reporting the *plain* sweep. Say which |
| `LOCUS` / `TOOLCHAIN` | unchanged, and load-bearing: cross-GPU and cross-container comparators have drifted 25% on measured hardware |

### What is identical in both modes

G3 and G4 do not care how you got here: probe **both** occupancy limiters the moment anything
compiles, and time with the interleaved instrument — per-arm kernel objects and cache dirs, oracle
before timing (checking `isfinite` first, since a tolerance comparison cannot fail on NaN), a delta
under the spread reported as `NOT RESOLVED`, and a flat result set across three or more arms treated
as a **cache-collision suspect** until the arm order has been permuted. The reversed-intuition traps
and "the round count is the denominator" apply unchanged.

---

## C. Re-entry — resuming a checkpointed run

Reload the checkpoint and the decision log as baseline and history, then continue the same loop.
Two things are not optional:

- **Re-assert G1.** The bundle may have been regenerated while you were away, and a resumed run that
  silently changed denominators produces a trajectory nobody can read.
- **If the interrupted run carried its debt unpaid** (`parity_unreached`, or a G2 exit 2 it never
  closed), the debt is still owed. Re-run G2 before the first new climb, not after.

A coupled bundle is always continued whole: resuming half of a memory-path → layout → pipeline change
measures neither half.
