"""Compare threshold types and inspect sentence distances (version-specific)."""
from langchain_experimental.text_splitter import SemanticChunker
from common import embeddings
from sample_docs import MIXED, SINGLE


def main():
    emb = embeddings()
    for name, doc in [("MIXED", MIXED), ("SINGLE", SINGLE)]:
        for kind in ["percentile", "standard_deviation", "interquartile", "gradient"]:
            chunks = SemanticChunker(emb, breakpoint_threshold_type=kind).split_text(doc)
            print(name, kind, len(chunks), [c[:40] for c in chunks])
        chunker = SemanticChunker(emb)
        sentences = chunker._get_single_sentences_list(doc)
        distances, _ = chunker._calculate_sentence_distances(sentences)
        print(name, "n_sent", len(sentences), "dist", [round(float(d), 3) for d in distances])


if __name__ == "__main__":
    main()
