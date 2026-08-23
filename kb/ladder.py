"""Publishing one measurement down a ladder of canonical ids — shared by both lanes' writers.

`publish(store, recs, files, score_of)` writes every rung in `recs` to ONE store, all-or-none, and
returns `(written_cids, promoted_cids, error)`.

All rungs or none. A partially-filled ladder is the one outcome worth avoiding: the coarse page
would hold whichever runs happened to succeed twice and would rank them as if that were the whole
history — and because the scheme has no search, no reader could ever tell that page was thin.
Stopping at the first failure leaves fewer records than intended but never a page that lies about
its own completeness, since the exact rung is written first.

`score_of(rec)` returns the scalar the champion pointer ranks on for that rung. The kernel lane
ranks every rung on `speedup`; the e2e lane ranks the exact rung on throughput and the coarser
rungs on speedup, and opens a DIFFERENT store per rung to do it — so it calls this once per rung
with a one-element `recs`, and drives all-or-none at its own loop level (breaking on a non-empty
`error`). The kernel lane, one metric for all rungs, passes the whole ladder in one call.

`promote=False` records the measurement but leaves the champion pointer where it is. Promotion is
purely a score comparison — it does not consult `validated` — so a number the WRITER already knows
is provisional (a run salvaged from disk artifacts, with no final Validate behind it) would
otherwise outrank a properly-gated champion on nothing but arithmetic. Such a record is still worth
storing: `candidates()` returns every session on the page, so a reader finds it either way, and the
champion keeps meaning "the best number somebody actually validated".
"""


def publish(store, recs, files, score_of, promote=True):
    """Write every rec to `store`, all-or-none. Returns (written, promoted, error)."""
    written, promoted_cids = [], []
    for rec in recs:
        try:
            store.write(rec["canonical_id"], rec["session_id"], rec["knowledge"], files)
            written.append(rec["canonical_id"])
            if promote and store.maybe_promote(
                    rec["canonical_id"], rec["session_id"], score_of(rec)):
                promoted_cids.append(rec["canonical_id"])
        except Exception as e:
            return written, promoted_cids, "%s: %s" % (type(e).__name__, str(e)[:160])
    return written, promoted_cids, ""
