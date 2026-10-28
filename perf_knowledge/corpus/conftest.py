"""Fixtures for the corpus consumer tests.

These tests come in two kinds, and the split between them is the point of this file.

Most of them build a synthetic corpus in `tmp_path` and assert on selection, ranking and
write-back logic. Those are unit tests of this repo and need nothing checked out.

A minority -- the ones that take `corpus_catalog` -- assert against the *real* corpus: that
every catalogued path resolves, that every taxonomy GEMM operator reaches the GEMM family, that
the shipped cards' dtypes are already canonical. Those are integration tests of the seam between
this repo and the knowledge repository, and they cannot be made hermetic without defeating their
purpose: a fixture catalog would only prove this repo agrees with itself.

So they resolve the corpus through `KERNEL_KNOWLEDGE_DIR` and skip, loudly and with the reason,
when it is not set. CI sets it after checking the knowledge repository out. A skip here is a
statement that the seam went unverified on this run -- not that it passed.
"""

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _corpus_contract as CONTRACT  # noqa: E402


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    """The configured knowledge repository, or skip naming what to set."""
    try:
        return CONTRACT.knowledge_dir()
    except CONTRACT.CorpusUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def corpus_catalog(corpus_dir: Path) -> Path:
    """Path to the real `catalog.yaml`, with the consumer contract already checked.

    Resolving through `load_catalog` rather than joining the path by hand means a schema or
    vocabulary mismatch fails here, once, with the two lists printed -- instead of surfacing as
    a dozen unrelated assertion errors further in.
    """
    path, _ = CONTRACT.load_catalog(corpus_dir / "corpus" / "catalog.yaml")
    return path
