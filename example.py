"""The article's semantic chunking example with a pinned CPU embedding model."""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from common import MODEL, REVISION, embeddings, handbook


def run():
    chunker = SemanticChunker(
        embeddings(), buffer_size=0, breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80, min_chunk_size=100,
    )
    docs = chunker.create_documents([handbook()])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    cap = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=254, chunk_overlap=0,
    )
    return cap.split_documents(docs)


if __name__ == "__main__":
    for i, doc in enumerate(run()):
        print(i, len(doc.page_content), doc.page_content[:50])
