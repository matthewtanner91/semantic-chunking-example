"""Compare buffer sizes and percentile thresholds against topic annotations."""
from langchain_experimental.text_splitter import SemanticChunker
from common import embeddings, score
from sample_docs import MIXED, SINGLE


def main():
    emb = embeddings()
    for name, doc, starts in [
        ("MIXED", MIXED, ["Travel expenses", "Company laptops"]),
        ("SINGLE", SINGLE, []),
    ]:
        for buffer in [0, 1, 2]:
            for percentile in [80, 90, 95]:
                chunks = SemanticChunker(
                    emb, buffer_size=buffer, breakpoint_threshold_amount=percentile,
                ).split_text(doc)
                hits, total, extra = score(chunks, starts)
                label = f"{hits}/{total}" if total else "N/A (no annotated transitions)"
                print(f"{name} buffer={buffer} pct={percentile}: {len(chunks)} chunks, "
                      f"topic-shift hits={label}, extra splits={extra}",
                      [c[:22] for c in chunks])


if __name__ == "__main__":
    main()
