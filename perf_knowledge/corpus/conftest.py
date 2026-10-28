"""Fixtures for the corpus consumer tests.

The split these fixtures create is the point of the file. Most tests build a synthetic
corpus in `tmp_path` and need nothing checked out. The ones taking `corpus_catalog` assert
against the *real* corpus instead -- that every catalogued path resolves, that every taxonomy
GEMM operator reaches the GEMM family -- and cannot be made hermetic without defeating their
purpose, since a fixture catalog would only prove this repo agrees with itself.

Those resolve through `KERNEL_KNOWLEDGE_DIR` and skip, with the reason, when it is unset. A
skip here says the seam went unverified on this run -- not that it passed.
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
