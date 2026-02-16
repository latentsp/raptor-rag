"""Quick test of gpt-5-mini summarization with different max_tokens and context sizes."""
import litellm

context_short = "The cat sat on the mat. It was a sunny day and the birds were singing."
context_long = "word " * 500  # ~500 words

for ctx_name, ctx in [("short", context_short), ("long", context_long)]:
    for mt in [100, 500, 1000, 2000]:
        resp = litellm.completion(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Write a summary of the following, including as many key details as possible: {ctx}:",
                },
            ],
            max_tokens=mt,
        )
        c = resp.choices[0]
        content = c.message.content or ""
        print(
            f"ctx={ctx_name}, max_tokens={mt}: finish={c.finish_reason}, "
            f"content_len={len(content)}, usage={resp.usage}"
        )
