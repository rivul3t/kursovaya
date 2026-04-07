
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvalSample:
    sample_id: str
    source_hotpot_id: str
    conflict_type: str
    documents: List[Dict[str, str]]
    conflicting_doc_indices: List[int]
    metadata: Dict[str, Any]


@dataclass
class Prediction:
    sample_id: str
    pred_has_conflict: bool
    pred_conflict_type: str
    pred_conflicting_doc_indices: List[int]
    raw_text: str
