from abc import ABC, abstractmethod

import litellm
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class LiteLLMQAModel(BaseQAModel):
    def __init__(
        self,
        model="gpt-4o-mini",
        system_prompt="You are a Question Answering assistant.",
        user_prompt_template="Given Context: {context}\nAnswer the following question: {question}",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt_template.format(context=context, question=question),
                },
            ],
            "max_tokens": max_tokens,
        }
        if litellm.supports_reasoning(model=self.model):
            # Reasoning models need extra token budget for chain-of-thought
            kwargs["max_tokens"] = max(max_tokens, 2000)
        else:
            kwargs["temperature"] = 0
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content.strip()


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required. Install with: pip install raptor-rag[huggingface]") from None
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError:
            raise ImportError("transformers is required. Install with: pip install raptor-rag[huggingface]") from None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]
