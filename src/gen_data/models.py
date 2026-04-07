from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json


@dataclass
class HotpotDoc:
    title: str
    sentences: List[str]

    def text(self) -> str:
        return " ".join(s.strip() for s in self.sentences if s and s.strip()).strip()


@dataclass
class HotpotExample:
    source_id: str
    question: Optional[str]
    answer: Optional[str]
    level: Optional[str]
    qtype: Optional[str]
    docs: List[HotpotDoc]
    supporting_facts: List[Tuple[str, int]]


@dataclass
class GeneratedSample:
    sample_id: str
    source_hotpot_id: str
    conflict_type: str
    documents: List[Dict[str, str]]
    conflicting_doc_indices: List[int]
    metadata: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
