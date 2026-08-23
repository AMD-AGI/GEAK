# Patches

Empty until the experiment produces something. One file per change, and each one needs to be
reviewable on its own:

- Name it for what it does, not for when you made it: `moe_dispatch_fuse.patch`, not `patch3.patch`.
- Head each file with a comment block giving the base it applies to (commit, or the file's pristine
  copy if there is no git metadata), the exact command to apply it, and the measurement it produced:
  local baseline tok/s → patched tok/s, plus the accuracy result.
- If a change only made sense together with another, say so in both headers. A stack that is only
  reproducible in one order is worth documenting as such.
- Keep patches that lost, in a `rejected/` subdirectory with their numbers. Knowing what was measured
  and discarded is most of the value of this directory.

If the framework in this container ships without source (installed as a plain package rather than a
checkout), keep a pristine copy of every file you touch so a diff can be produced at all — there is no
base commit to diff against.
