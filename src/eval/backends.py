
from __future__ import annotations

import os
import random
import re
from typing import Protocol
from openai import RateLimitError



class LLMBackend(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        ...


class MockBackend(LLMBackend):
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        lower = prompt.lower()
        if 'answer with exactly one word' in lower or 'yes or no' in lower:
            return 'YES' if self.rng.random() > 0.5 else 'NO'
        if 'exactly one label' in lower:
            return self.rng.choice(['none', 'self', 'pair', 'conditional'])
        if 'json array of integers' in lower:
            return '[0]' if self.rng.random() > 0.5 else '[1]'
        if '<sentence>' in prompt and ('choose' in lower or 'choose the' in lower):
            m = re.search(r'<sentence>(.*?)</sentence>', prompt, flags=re.S)
            return m.group(0) if m else '<sentence>Placeholder sentence.</sentence>'
        return '<text>Placeholder output.</text>'


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str, base_url) -> None:
        self.model = model
        self.base_url = base_url
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError('Install with: pip install openai') from exc
        self._client = OpenAI(api_key=os.getenv('LLM_RESAYIL_API_KEY'), base_url=self.base_url)

    def generate(self, prompt: str, *, temperature=0.2, max_tokens=512) -> str:
        for attempt in range(5):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = resp.choices[0].message.content
                if content:
                    return content.strip()

            except RateLimitError as e:
                wait = getattr(e, "retry_after", None) or (2 ** attempt)
                wait += random.uniform(0, 1)  # jitter

                print(f"[RateLimit] sleeping {wait:.2f}s...")
                time.sleep(wait)

        raise RuntimeError("Failed after retries")


class GroqBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.model = model
        try:
            from groq import Groq
        except Exception as exc:
            raise RuntimeError('Install with: pip install groq') from exc
        self._client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 10256) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            #temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        if content:
            return content
        raise RuntimeError('Could not extract text from Groq response')

class GeminiBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.model = model
        try:
            from google import genai
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install with: pip install google-genai") from exc

        self._client = genai.Client()

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        text = getattr(resp, "text", None)
        if text and text.strip():
            return text.strip()

        try:
            chunks = []
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", []) or []:
                    t = getattr(part, "text", None)
                    if t:
                        chunks.append(t)
            joined = "".join(chunks).strip()
            if joined:
                return joined
        except Exception:
            pass

        raise RuntimeError("Could not extract text from Gemini response")