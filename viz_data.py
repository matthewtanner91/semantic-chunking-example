"""Print diagram data; private helpers require the pinned LangChain release."""
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter
from transformers import AutoTokenizer
from common import MODEL, REVISION, embeddings
from sample_docs import MIXED, SINGLE


def main():
    emb = embeddings()
    chunker = SemanticChunker(emb, buffer_size=0, breakpoint_threshold_amount=80)
    sentences = chunker._get_single_sentences_list(MIXED)
    distances, _ = chunker._calculate_sentence_distances(sentences)
    print("n_sent", len(sentences))
    for i, sentence in enumerate(sentences):
        print(i, sentence[:50])
    print("dist", [round(float(d), 3) for d in distances])
    print("thr80", round(float(np.percentile(distances, 80)), 3))
    print("len", len(MIXED), "topic starts", [MIXED.index(s) for s in ["Travel expenses", "Company laptops"]])
    fixed = TokenTextSplitter(chunk_size=60, chunk_overlap=0).split_text(MIXED)
    print("fixed 60-token chunks", len(fixed), "starts", [MIXED.index(c[:30]) for c in fixed])
    for chunk in fixed:
        print("  F|", chunk[:45], "...", chunk[-30:])
    semantic = SemanticChunker(emb, buffer_size=0, breakpoint_threshold_amount=80, min_chunk_size=100).split_text(MIXED)
    print("semantic starts", [MIXED.index(c[:30]) for c in semantic], [len(c) for c in semantic])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    for name, doc in [("handbook.txt", MIXED), ("SINGLE", SINGLE)]:
        segments = chunker._get_single_sentences_list(doc)
        print(name, "sentences", len(segments), "max wordpieces", max(len(tokenizer.tokenize(s)) for s in segments))


if __name__ == "__main__":
    main()
