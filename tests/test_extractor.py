from document_classifier.extractor import clean_text

def test_clean_text():
    text = "This is a Sample! Text, with PUNCTUATION."
    cleaned = clean_text(text)
    assert "sample" in cleaned
    assert "punctuation" in cleaned
    assert "this" not in cleaned  # should be removed as a stopword