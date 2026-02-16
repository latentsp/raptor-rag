"""Integration tests that require API keys. Skipped in CI.

Run with: pytest tests/test_integration.py -m integration
Requires OPENAI_API_KEY environment variable (read natively by LiteLLM).
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def has_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


class TestLiteLLMIntegration:
    def test_litellm_summarization(self, has_openai_key):
        from raptor import LiteLLMSummarizationModel

        model = LiteLLMSummarizationModel(model="gpt-4o-mini")
        result = model.summarize("The cat sat on the mat. It was a sunny day.", max_tokens=50)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_litellm_qa(self, has_openai_key):
        from raptor import LiteLLMQAModel

        model = LiteLLMQAModel(model="gpt-4o-mini")
        result = model.answer_question("The capital of France is Paris.", "What is the capital of France?")
        assert isinstance(result, str)
        assert "Paris" in result

    def test_litellm_embedding(self, has_openai_key):
        from raptor import LiteLLMEmbeddingModel

        model = LiteLLMEmbeddingModel()
        result = model.create_embedding("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)


class TestEndToEnd:
    def test_full_pipeline(self, has_openai_key):
        from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

        config = RetrievalAugmentationConfig(
            tb_num_layers=1,
            tb_max_tokens=50,
        )
        ra = RetrievalAugmentation(config=config)

        text = (
            "Cinderella lived with her wicked stepmother and two stepsisters. "
            "She was forced to do all the housework. One day, a fairy godmother appeared "
            "and transformed her for the royal ball. At the ball, she danced with the prince. "
            "When the clock struck midnight, she ran away, leaving behind a glass slipper. "
            "The prince searched the kingdom and found Cinderella. They lived happily ever after."
        )

        ra.add_documents(text)
        assert ra.tree is not None
        assert ra.retriever is not None

        answer = ra.answer_question("Who did Cinderella dance with?")
        assert isinstance(answer, str)
        assert len(answer) > 0
