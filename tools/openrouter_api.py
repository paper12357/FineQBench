import os
import time
from openai import OpenAI

class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide OpenRouter API key via parameter or OPENROUTER_API_KEY environment variable.")
        
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def call_llm(self, prompt: str, model: str = 'deepseek/deepseek-chat', max_tokens: int = 50000) -> str:
        attempt = 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.8,
                    presence_penalty=0.2,
                    frequency_penalty=0.2
                )
                usage = resp.usage
                if usage:
                    self.prompt_tokens += usage.prompt_tokens or 0
                    self.completion_tokens += usage.completion_tokens or 0
                    self.total_tokens += usage.total_tokens or 0

                return resp.choices[0].message.content.strip()
            except Exception as e:
                attempt += 1
                wait_time = 0.2
                print(f"[LLM CALL] LLM call failed (attempt {attempt}): {e}")

                if attempt >= 5:
                    print("[LLM CALL ERROR] Reached max retries. Giving up.")
                    raise e
                time.sleep(wait_time)

    def get_and_reset_token_usage(self):
        ret = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        return ret