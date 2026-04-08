
from __future__ import annotations

import json
import re
from typing import List


def _clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip().lower()


def parse_yes_no(text: str) -> bool:
    t = _clean_text(text)
    if re.search(r'yes', t, re.IGNORECASE):
        return True
    if re.search(r'no', t, re.IGNORECASE):
        return False
    return False


def parse_conflict_type(text: str) -> str:
    t = _clean_text(text)
    for label in ('conditional', 'pair', 'self', 'none'):
        if label in t:
            return label
    return 'none'


def parse_indices(text: str) -> list[int]:
    import re, json

    m = re.search(r"<documents>(.*?)</documents>", text, re.S)
    if m:
        content = m.group(1)
    else:
        content = text

    nums = re.findall(r"\d+", content)
    return sorted(set(int(n) for n in nums))
