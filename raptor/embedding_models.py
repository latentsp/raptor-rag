from abc import ABC, abstractmethod

import litellm
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class LiteLLMEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        response = litellm.embedding(model=self.model, input=[text])
        return response.data[0]["embedding"]


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install raptor-rag[huggingface]"
            ) from None
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install raptor-rag[huggingface]"
            ) from None
        self.model = SentenceTransformer(model_name, device=device)

    def create_embedding(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
