"""Tests for BaseTextSplitter and DefaultTextSplitter."""

import pytest

from raptor.utils import BaseTextSplitter, DefaultTextSplitter, split_text


class TestBaseTextSplitter:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseTextSplitter()

    def test_subclass_must_implement_split_text(self):
        class IncompleteSplitter(BaseTextSplitter):
            pass

        with pytest.raises(TypeError):
            IncompleteSplitter()

    def test_custom_splitter(self, tokenizer):
        class SentenceSplitter(BaseTextSplitter):
            def split_text(self, text, tokenizer, max_tokens):
                return [s.strip() for s in text.split(".") if s.strip()]

        splitter = SentenceSplitter()
        chunks = splitter.split_text("Hello world. Foo bar. Baz.", tokenizer, 100)
        assert len(chunks) == 3
        assert chunks[0] == "Hello world"
        assert chunks[1] == "Foo bar"
        assert chunks[2] == "Baz"


class TestDefaultTextSplitter:
    def test_instantiation_default(self):
        splitter = DefaultTextSplitter()
        assert splitter.overlap == 0

    def test_instantiation_with_overlap(self):
        splitter = DefaultTextSplitter(overlap=2)
        assert splitter.overlap == 2

    def test_is_base_text_splitter(self):
        splitter = DefaultTextSplitter()
        assert isinstance(splitter, BaseTextSplitter)

    def test_split_text_delegates_to_split_text_func(self, tokenizer):
        text = "Hello world. This is a test. Another sentence here."
        splitter = DefaultTextSplitter()
        result = splitter.split_text(text, tokenizer, max_tokens=100)
        expected = split_text(text, tokenizer, max_tokens=100, overlap=0)
        assert result == expected

    def test_split_text_with_small_max_tokens(self, tokenizer):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        splitter = DefaultTextSplitter()
        chunks = splitter.split_text(text, tokenizer, max_tokens=10)
        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should be reasonably sized
            tokens = len(tokenizer.encode(chunk))
            # Allow some slack since sentences aren't split mid-sentence
            assert tokens <= 20

    def test_split_empty_text(self, tokenizer):
        splitter = DefaultTextSplitter()
        chunks = splitter.split_text("", tokenizer, max_tokens=100)
        assert chunks == []

    def test_split_single_sentence(self, tokenizer):
        splitter = DefaultTextSplitter()
        chunks = splitter.split_text("Hello world", tokenizer, max_tokens=100)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_split_preserves_content(self, tokenizer):
        text = "Machine learning is great. Deep learning is powerful. NLP is fascinating."
        splitter = DefaultTextSplitter()
        chunks = splitter.split_text(text, tokenizer, max_tokens=100)
        combined = " ".join(chunks)
        # All key words should be present
        assert "Machine learning" in combined
        assert "Deep learning" in combined
        assert "NLP" in combined
