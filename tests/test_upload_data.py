from buster.tokenizers import GPTTokenizer
from scripts.upload_data import chunk_text
from src.cfg import buster_cfg


def test_chunk_text():
    tokenizer = GPTTokenizer(buster_cfg.tokenizer_cfg["model_name"])

    # Test with a short text
    text = "This is a short text."
    token_limit = 10
    chunks = chunk_text(text, tokenizer, token_limit)
    assert all(tokenizer.num_tokens(chunk) <= token_limit for chunk in chunks)
    assert "".join(chunks) == text

    # Test with a long text
    text = "This is a long text. " * 100
    token_limit = 10
    chunks = chunk_text(text, tokenizer, token_limit)
    assert all(tokenizer.num_tokens(chunk) <= token_limit for chunk in chunks)
    assert "".join(chunks) == text

    # Test with a long non-latin text
    text = "นี่คือข้อความที่ยาว. " * 100
    token_limit = 10
    chunks = chunk_text(text, tokenizer, token_limit)
    assert all(tokenizer.num_tokens(chunk) <= token_limit for chunk in chunks)
    assert "".join(chunks) == text
