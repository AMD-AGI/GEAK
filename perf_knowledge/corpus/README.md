# `corpus/` — the consumer side

The corpus itself is not here. Its evidence, its curated decision cards, its performance axes, the
generators that render them and the gates that keep them self-consistent live in the knowledge
repository ([KernelWiki](https://github.com/yueliu14/KernelWiki), under `corpus/`).
What stays in GEAK is the code that *spends* that knowledge during a kernel run.

The cut is the direction of data flow, not the file type:

| | lives in | because |
|---|---|---|
| evidence, decision cards, axes, rendered pages | knowledge repo | they are the artifact |
| extractor, renderers, normalizer, self-consistency gates | knowledge repo | they produce or verify the artifact |
| candidate selection, product validation, benchmark merge/registry | here | they read the artifact to plan or score a run |

A knowledge artifact and the thing that proves it correct have to move together, or the proof goes
stale the first time either side is edited alone.

## Resolving the corpus

Everything here reaches the corpus through `_corpus_contract.py`, never by joining a path by hand:

```py
import _corpus_contract as CONTRACT
catalog_file, catalog = CONTRACT.load_catalog()          # honours KERNEL_KNOWLEDGE_DIR
entry = CONTRACT.family(catalog, "gemm")
normalizer = CONTRACT.load_declared_module(catalog_file, entry, "performance_normalizer")
```

`load_catalog` checks two things before returning: that `schema_version` is one this consumer
understands, and that `catalog.yaml`'s `authoring_languages` still equals `kb.py`'s
`AUTHORING_LANGUAGES`. That second check is the one that matters — the language name is the join key
between a corpus record and a learned card, so a drift there does not raise, it silently stops
matching. The catalog declares the vocabulary; this repo conforms to it.

There is no bundled fallback corpus. With `KERNEL_KNOWLEDGE_DIR` unset, `CONTRACT.knowledge_dir()`
raises `CorpusUnavailable` naming the variable, rather than quietly planning a run off nothing.

## Tests

Two kinds, and `conftest.py` keeps them apart:

- **unit** — build a synthetic corpus in `tmp_path`, assert on selection, ranking and write-back.
  43 of them, hermetic, need no checkout.
- **seam** — take the `corpus_dir` / `corpus_catalog` fixtures and assert against the *real* corpus:
  every catalogued path resolves, every taxonomy GEMM operator reaches the GEMM family, shipped
  dtypes are canonical. 24 of them. Without a knowledge dir they skip, naming the variable.

A skip here says the seam went unverified on this run — not that it passed. CI checks the knowledge
repository out and sets `KERNEL_KNOWLEDGE_DIR`, so the seam is actually exercised there:

```sh
KERNEL_KNOWLEDGE_DIR=/path/to/KernelWiki python3 -m pytest perf_knowledge/corpus -q
```
