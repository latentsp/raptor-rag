"""Tests for model classes: instantiation and mocked API calls."""

from unittest.mock import MagicMock, patch

import pytest

from raptor.embedding_models import (
    BaseEmbeddingModel,
    LiteLLMEmbeddingModel,
)
from raptor.qa_models import (
    BaseQAModel,
    LiteLLMQAModel,
)
from raptor.summarization_models import (
    BaseSummarizationModel,
    LiteLLMSummarizationModel,
)


class TestBaseClasses:
    def test_base_summarization_is_abstract(self):
        with pytest.raises(TypeError):
            BaseSummarizationModel()

    def test_base_qa_is_abstract(self):
        with pytest.raises(TypeError):
            BaseQAModel()

    def test_base_embedding_is_abstract(self):
        with pytest.raises(TypeError):
            BaseEmbeddingModel()


class TestLiteLLMSummarizationModel:
    def test_instantiation(self):
        model = LiteLLMSummarizationModel(model="gpt-4o-mini")
        assert model.model == "gpt-4o-mini"

    def test_custom_prompts(self):
        model = LiteLLMSummarizationModel(
            model="gpt-4o",
            system_prompt="Custom system",
            user_prompt_template="Summarize: {context}",
        )
        assert model.system_prompt == "Custom system"
        assert model.user_prompt_template == "Summarize: {context}"

    @patch("litellm.completion")
    def test_summarize(self, mock_completion):
        mock_completion.return_value.choices = [MagicMock(message=MagicMock(content="LiteLLM summary"))]

        model = LiteLLMSummarizationModel()
        result = model.summarize("Some context")
        assert result == "LiteLLM summary"
        mock_completion.assert_called_once()


class TestLiteLLMQAModel:
    def test_instantiation(self):
        model = LiteLLMQAModel(model="gpt-4o-mini")
        assert model.model == "gpt-4o-mini"

    @patch("litellm.completion")
    def test_answer_question(self, mock_completion):
        mock_completion.return_value.choices = [MagicMock(message=MagicMock(content="  LiteLLM answer  "))]

        model = LiteLLMQAModel()
        result = model.answer_question("context", "question")
        assert result == "LiteLLM answer"


class TestLiteLLMEmbeddingModel:
    def test_instantiation(self):
        model = LiteLLMEmbeddingModel(model="text-embedding-ada-002")
        assert model.model == "text-embedding-ada-002"

    def test_custom_model(self):
        model = LiteLLMEmbeddingModel(model="text-embedding-3-small")
        assert model.model == "text-embedding-3-small"

    @patch("litellm.embedding")
    def test_create_embedding(self, mock_embedding):
        mock_embedding.return_value.data = [{"embedding": [0.1, 0.2, 0.3]}]

        model = LiteLLMEmbeddingModel()
        result = model.create_embedding("test text")
        assert result == [0.1, 0.2, 0.3]
        mock_embedding.assert_called_once()

    @patch("litellm.embedding")
    def test_create_embedding_strips_newlines(self, mock_embedding):
        mock_embedding.return_value.data = [{"embedding": [0.1]}]

        model = LiteLLMEmbeddingModel()
        model.create_embedding("hello\nworld")
        call_args = mock_embedding.call_args
        assert call_args.kwargs["input"] == ["hello world"]
