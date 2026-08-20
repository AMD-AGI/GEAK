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
"""


def publish(store, recs, files, score_of):
    """Write every rec to `store`, all-or-none. Returns (written, promoted, error)."""
    written, promoted = [], []
    for rec in recs:
        try:
            store.write(rec["canonical_id"], rec["session_id"], rec["knowledge"], files)
            written.append(rec["canonical_id"])
            if store.maybe_promote(rec["canonical_id"], rec["session_id"], score_of(rec)):
                promoted.append(rec["canonical_id"])
        except Exception as e:
            return written, promoted, "%s: %s" % (type(e).__name__, str(e)[:160])
    return written, promoted, ""
