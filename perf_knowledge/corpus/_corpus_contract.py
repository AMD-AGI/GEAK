#!/usr/bin/env python3
"""Resolve the knowledge repository this workflow reads, and check the contract with it.

The corpus itself -- evidence, decision cards, performance axes, the pages rendered from them --
lives in a separate knowledge repository. This repo holds only the logic that *consumes* it:
picking which cards apply to a problem, resolving a citation for a prompt, scoring an outcome
and writing the result back. Nothing here is a knowledge artifact, and nothing here can verify
one; `--check` on a rendered page is a self-consistency property of the repo that stores it.

So there are exactly two things to get right at the seam, and this module is both of them.

**Where the corpus is.** `KERNEL_KNOWLEDGE_DIR` points at a checkout of the knowledge repo. That
is the same variable `kernel_workflow/kernel_lane.js` already resolves, so a lane and a script
invoked by hand agree on which corpus they are talking about. There is no bundled default: a
silent fallback to a stale vendored copy is how two sources of truth get created, and the whole
point of the split was to have one.

**Whether we agree on its shape.** `catalog.yaml` declares `schema_version` and the vocabulary
and modules a consumer may use. This module asserts the version is one we understand and that
`kb.AUTHORING_LANGUAGES` still equals the catalog's `authoring_languages` -- the join key
between a corpus record and a learned card. Two copies of that list in two repos is precisely
the drift the declaration was moved there to prevent, so it is checked on every resolve rather
than documented and hoped for.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSION = 1
ENV_VAR = "KERNEL_KNOWLEDGE_DIR"


class CorpusUnavailable(RuntimeError):
    """No knowledge repository is configured, or the one configured is not readable."""


class CorpusContractError(RuntimeError):
    """A knowledge repository was found, but it does not match what this consumer supports."""


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def knowledge_dir(explicit: str | Path | None = None) -> Path:
    """The configured knowledge repository, or raise with the one thing the caller must do."""
    raw = str(explicit) if explicit else os.environ.get(ENV_VAR, "")
    if not raw:
        raise CorpusUnavailable(
            f"{ENV_VAR} is not set. The performance corpus lives in a separate knowledge "
            "repository; point this at a checkout of it (the directory containing "
            "`corpus/catalog.yaml`).",
        )
    root = Path(raw).expanduser().resolve()
    if not (root / "corpus" / "catalog.yaml").is_file():
        raise CorpusUnavailable(f"{root} has no corpus/catalog.yaml; not a knowledge repository")
    return root


def catalog_path(explicit: str | Path | None = None) -> Path:
    return knowledge_dir(explicit) / "corpus" / "catalog.yaml"


def load_catalog(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Read the catalog and check the two things that make it usable: version and vocabulary."""
    resolved = Path(path) if path else catalog_path()
    catalog = _load_yaml(resolved)

    version = catalog.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise CorpusContractError(
            f"{resolved}: corpus schema_version {version!r}, this consumer supports "
            f"{SUPPORTED_SCHEMA_VERSION}. Either update the consumer or check out a knowledge "
            "repository at the matching schema.",
        )

    declared = catalog.get("authoring_languages")
    if not declared:
        raise CorpusContractError(f"{resolved}: declares no authoring_languages")
    _check_language_vocabulary(resolved, declared)

    return resolved, catalog


def _check_language_vocabulary(resolved: Path, declared: list[str]) -> None:
    """`kb.AUTHORING_LANGUAGES` must equal what the corpus declares.

    The corpus keys every source record on a language and the learned cards key every card on
    one; joining the two is the reason the field exists. The list is declared in the knowledge
    repo and conformed to here, so a language added there without being added here fails loudly
    on the next resolve instead of silently dropping records from the join.
    """
    kb_path = Path(__file__).resolve().parents[2] / "kernel_workflow" / "scripts" / "kb.py"
    spec = importlib.util.spec_from_file_location("_corpus_contract_kb", kb_path)
    if not spec or not spec.loader:  # pragma: no cover - a broken checkout, not a contract break
        raise CorpusContractError(f"cannot load the card parser at {kb_path}")
    kb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kb)

    ours, theirs = set(kb.AUTHORING_LANGUAGES), set(declared)
    if ours != theirs:
        raise CorpusContractError(
            f"authoring language vocabulary disagrees with {resolved}:\n"
            f"  only in kb.py:       {sorted(ours - theirs)}\n"
            f"  only in catalog:     {sorted(theirs - ours)}\n"
            "The corpus declares this list; update kb.AUTHORING_LANGUAGES to match it.",
        )


def family(catalog: dict[str, Any], requested: str) -> dict[str, Any]:
    for entry in catalog.get("families") or []:
        if entry.get("id") == requested or requested in (entry.get("patterns") or []):
            return entry
    known = sorted(str(e.get("id")) for e in catalog.get("families") or [])
    raise CorpusContractError(f"no corpus family for {requested!r}; catalogued: {known}")


def load_declared_module(catalog_file: Path, entry: dict[str, Any], key: str):
    """Load a module the catalog names, by path, under a private name.

    The catalog names the modules a consumer may use (`performance_normalizer`,
    `source_extractor`). Resolving them through it rather than importing them as siblings is
    what keeps this repo depending on the knowledge repo's *interface* instead of its directory
    layout -- and it is also what makes the dependency visible: these are the only two places
    this repo executes code it does not own.

    Loaded by path under a private name on purpose. A `sys.path.insert` plus a plain import
    binds whatever is already in `sys.modules`, which in a single pytest session is whatever
    imported first.
    """
    relative = entry.get(key)
    if not relative:
        raise CorpusContractError(
            f"{catalog_file}: family {entry.get('id')!r} declares no {key}",
        )
    path = catalog_file.parent / relative
    if not path.is_file():
        raise CorpusContractError(f"{catalog_file}: {key} -> {relative} does not exist")
    spec = importlib.util.spec_from_file_location(f"_corpus_{key}", path)
    if not spec or not spec.loader:
        raise CorpusContractError(f"cannot load {key} at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
