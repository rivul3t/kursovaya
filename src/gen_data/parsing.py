from __future__ import annotations

import re
from typing import Tuple


def extract_tagged_text(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, flags=re.S | re.I)
    if not m:
        return text.strip()
    return m.group(1).strip()


def extract_sentence(text: str) -> str:
    return extract_tagged_text(text, "sentence")


def extract_document(text: str) -> str:
    return extract_tagged_text(text, "document")


def extract_three_documents(text: str) -> Tuple[str, str, str]:
    d1 = extract_tagged_text(text, "document1")
    d2 = extract_tagged_text(text, "document2")
    d3 = extract_tagged_text(text, "document3")
    if not (d1 and d2 and d3):
        raise ValueError(f"Could not parse three documents from model output:\n{text}")
    return d1, d2, d3
