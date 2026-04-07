from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Any

from .models import HotpotDoc, HotpotExample


def _normalize_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _flatten_strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, list):
        for y in x:
            yield from _flatten_strings(y)

def _row_to_example(row: dict, fallback_id: int) -> HotpotExample:
    docs = []

    context = row.get("context", {})
    title = context.get("title", f"doc_{fallback_id}")
    sents = context.get("sentences", [])

    sentences = []
    for s in _flatten_strings(sents):
        norm = _normalize_sentence(s)
        if norm:
            sentences.append(norm)

    if sentences:
        docs.append(HotpotDoc(title=str(title), sentences=sentences))

    # нормализуем только строки
    sentences = [_normalize_sentence(s) for s in sents if isinstance(s, str) and _normalize_sentence(s)]
    if sentences:
        docs.append(HotpotDoc(title=title, sentences=sentences))

    #print(f"Loaded {len(docs)} documents")

    supporting = []
    raw_sf = row.get("supporting_facts", []) or []
    for x in raw_sf:
        try:
            supporting.append((str(x[0]), int(x[1])))  # JSON format
        except Exception:
            if isinstance(raw_sf, dict):
                titles = raw_sf.get("title", [])
                sent_ids = raw_sf.get("sent_id", [])
                supporting = [(str(t), int(s)) for t, s in zip(titles, sent_ids)]
                break
            elif isinstance(x, dict):
                supporting.append((str(x.get("title")), int(x.get("sent_id"))))

    return HotpotExample(
        source_id=str(row.get("_id") or row.get("id") or row.get("source_id") or fallback_id),
        question=row.get("question"),
        answer=row.get("answer"),
        level=row.get("level"),
        qtype=row.get("type"),
        docs=docs,
        supporting_facts=supporting,
    )


def load_hotpot_examples(
    input_json: Optional[str],
    input_parquet: Optional[str],
    hf_split: Optional[str],
) -> List[HotpotExample]:
    if input_json:
        path = Path(input_json)
        if not path.exists():
            raise FileNotFoundError(f"Hotpot JSON file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_row_to_example(row, i) for i, row in enumerate(data)]

    if input_parquet:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install with: pip install datasets, pyarrow") from exc

        ds = load_dataset("parquet", data_files=input_parquet, split="train")
        return [_row_to_example(dict(row), i) for i, row in enumerate(ds)]

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install with: pip install datasets") from exc

    split = hf_split or "train"
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    examples = [_row_to_example(dict(row), i) for i, row in enumerate(ds)]
    examples = [ex for ex in examples if ex.docs] 
    
    return examples