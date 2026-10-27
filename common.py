"""Shared reproducible model configuration."""
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ROOT = Path(__file__).resolve().parent


def embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=MODEL,
        model_kwargs={"revision": REVISION, "device": "cpu"},
    )


def handbook():
    return (ROOT / "handbook.txt").read_text(encoding="utf-8")


def score(chunks, topic_starts):
    """Count annotated transitions; the initial chunk is never a transition."""
    hits = sum(any(c.startswith(start) for c in chunks[1:]) for start in topic_starts)
    return hits, len(topic_starts), max(0, len(chunks) - 1 - hits)
