from rag.pdf_processor import clean_text, chunk_documents

def test_clean_text():
    assert clean_text("A   B\n\n\nC") == "A B\n\nC"

def test_chunk_documents():
    pages = [{"source": "paper.pdf", "page": 1, "text": " ".join(["word"] * 300)}]
    chunks = chunk_documents(pages, chunk_words=100, overlap_words=20)
    assert len(chunks) >= 3
    assert chunks[0]["source"] == "paper.pdf"
    assert chunks[0]["page"] == 1
