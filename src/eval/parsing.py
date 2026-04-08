
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


def parse_indices(text: str) -> List[int]:
    m = re.search(r'\[[^\]]*\]', text)
    if m:
        chunk = m.group(0)
        try:
            data = json.loads(chunk)
            if isinstance(data, list):
                out = []
                for x in data:
                    try:
                        out.append(int(x))
                    except Exception:
                        pass
                return sorted(set(out))
        except Exception:
            pass
    nums = re.findall(r'\d+', text)
    return sorted(set(int(n) for n in nums))
