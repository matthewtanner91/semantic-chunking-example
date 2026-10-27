# Semantic chunking example

Reproduce the small handbook experiment accompanying Matt Tanner's semantic chunking article. The example uses local CPU embeddings; it needs no API key. Initial setup downloads Python packages and the public embedding model.

`handbook.txt` is a sample policy document for the tutorial, not a real company handbook or advice about workplace policies. It is the original experiment's 1,216-character input, preserved without a trailing newline. `sample_docs.py` adds the original single-topic travel example. No company handbook, article draft, or private review material is included.

## Run

The lock file is tested on **macOS arm64 with Python 3.13.7**. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and provide Python 3.13, then run from this directory:

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
uv venv --python python3.13 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements-macos-arm64.txt
.venv/bin/python example.py
```

The article's recorded output is:

```text
0 303 On-call engineers rotate weekly, starting Monday a
1 149 If you swap shifts, update the rotation calendar b
2 374 Travel expenses are reimbursed only with an itemiz
3 387 Company laptops must use full-disk encryption at a
```

Lengths are **characters**, not tokens. The follow-up recursive splitter uses a 254-token target with the model's tokenizer. It does not undo truncation that may have occurred during earlier embedding; the small fixture's sentence inputs avoid that issue. This example measures boundaries, not retrieval or answer quality.

## Experiments and tests

```sh
.venv/bin/python sweep_real.py
.venv/bin/python run_real.py
.venv/bin/python viz_data.py
.venv/bin/python -m unittest discover -s tests -v
RUN_MODEL_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

`sweep_real.py` varies buffer size and percentile. MIXED has two annotated transitions, beginning with `Travel expenses` and `Company laptops`. Only starts of chunks after the initial chunk count as detected transitions. SINGLE has no annotated transitions: its recall is reported as N/A and all its boundaries are extra splits. This corrects the historical sweep's erroneous `1/2` SINGLE score, which counted the initial chunk as a transition.

`run_real.py` compares four threshold types. `viz_data.py` prints distances, fixed 60-token windows, semantic boundaries, and sentence lengths without saving generated artifacts. Both diagnostic scripts use private `SemanticChunker` helpers specific to the pinned version. The default sentence regex splits `9 a.m. local time` into two segments: the handbook has 18 intended sentences and 19 regex segments.

The opt-in integration test checks the four chunk lengths, preservation of the complete input, and token lengths. The default tests need only Python's standard library and do not download a model. See [VALIDATION.md](VALIDATION.md) for actual test evidence.

## Reproducibility and provenance

The original October 12, 2026 experiment recorded Python 3.13.16 and Linux CPU PyTorch `2.14.1+cpu`. This Mac reproduction uses Python 3.13.7 and the macOS PyTorch `2.14.1` wheel, explicitly running on CPU. The original artifact did not record a tiktoken version; this repository pins `tiktoken==0.12.0`. The remaining directly recorded library versions are preserved in `requirements.in`, with transitive versions and hashes in the platform-specific lock file. Other platforms are not claimed to be tested.

Both embeddings and tokenizer use `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Model weights are downloaded separately and are not redistributed here; consult the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for its license. Floating-point details can vary across hardware and library builds.

LangChain [sunset `langchain-experimental`](https://github.com/langchain-ai/langchain-experimental/issues/87) in May 2026. This repository intentionally retains version 0.4.2 to reproduce the article. It is a teaching example, not a maintained production chunking framework.

The input source described these as sample documents used for the semantic chunking experiments. No separate ownership or redistribution license declaration accompanied the sample text. Dependencies and the separately downloaded model retain their own licenses.

All environments, model downloads, and caches live in ignored `.venv/` and `.cache/` directories. They can be removed after testing with `rm -rf .venv .cache` from this repository directory. Do not commit these directories or credentials.
