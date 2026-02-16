import logging
from abc import ABC, abstractmethod

import litellm
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=500):
        pass


class LiteLLMSummarizationModel(BaseSummarizationModel):
    def __init__(
        self,
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        user_prompt_template="Write a summary of the following, including as many key details as possible: {context}:",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500):
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_template.format(context=context)},
            ],
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content
        if not content or not content.strip():
            logger.warning(
                "Empty summarization: model=%s, max_tokens=%d, finish_reason=%s, context_len=%d",
                self.model, max_tokens, choice.finish_reason, len(context),
            )
            raise ValueError("Summarization returned empty content")
        return content
