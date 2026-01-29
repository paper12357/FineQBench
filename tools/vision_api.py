import os
import time
from openai import OpenAI
import base64


class VisionClient:
    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1"):
        base = base_url or os.getenv("OPENAI_API_BASE")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(base_url=base, api_key=api_key)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def call_vision(self, prompt: str, image_path: str, model: str = "gpt-4.1") -> str:
        attempt = 0
        while True:
            try:
                base64_image = self.encode_image(image_path)

                response = self.client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                { "type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,${base64_image}",
                                },
                            ],
                        }
                    ],
                )

                return response.output_text

            except Exception as e:
                attempt += 1
                wait_time = 0.5
                print(f"[VISION CALL] Vision call failed (attempt {attempt}): {e}")
                if attempt >= 5:
                    print("[VISION CALL ERROR] Reached max retries. Giving up.")
                    raise e
                time.sleep(wait_time)

