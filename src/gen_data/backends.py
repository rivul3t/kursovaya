from __future__ import annotations

import os
import random
import re
from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str: ...


class MockBackend:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        lower = prompt.lower()
        if "<sentence>" in prompt and "choose" in lower:
            m = re.search(r"<sentence>(.*?)</sentence>", prompt, flags=re.S)
            return m.group(0) if m else "<sentence>The statement is important.</sentence>"
        if "<statement>" in prompt and ("contradict" in lower or "suggest otherwise" in lower):
            m = re.search(r"<statement>(.*?)</statement>", prompt, flags=re.S)
            stmt = m.group(1).strip() if m else "The statement is false."
            return f"<statement>Not true: {stmt}</statement>"
        if "<document1>" in prompt or "conditional" in lower:
            return (
                "<document1>Alpha is the correct version.</document1>\n"
                "<document2>Alpha is the correct version.</document2>\n"
                "<document3>Alpha cannot be the correct version.</document3>"
            )
        return "<text>Placeholder output.</text>"


class OpenAICompatibleBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install with: pip install openai") from exc
        self._client = OpenAI()

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        resp = self._client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text
        try:
            chunks = []
            for item in resp.output:
                for c in getattr(item, "content", []):
                    t = getattr(c, "text", None)
                    if t:
                        chunks.append(t)
            if chunks:
                return "".join(chunks)
        except Exception:
            pass
        raise RuntimeError("Could not extract text from model response")


class GroqBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        try:
            from groq import Groq
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install with: pip install groq") from exc
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            #temperature=temperature,
            #max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        if content:
            return content
        raise RuntimeError("Could not extract text from Groq response")
