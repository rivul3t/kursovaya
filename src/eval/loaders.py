
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .types import EvalSample


def load_jsonl(path: str | Path) -> List[EvalSample]:
    p = Path(path)
    samples: List[EvalSample] = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                EvalSample(
                    sample_id=str(row['sample_id']),
                    source_hotpot_id=str(row.get('source_hotpot_id', '')),
                    conflict_type=str(row['conflict_type']),
                    documents=list(row.get('documents', [])),
                    conflicting_doc_indices=list(row.get('conflicting_doc_indices', [])),
                    metadata=dict(row.get('metadata', {})),
                )
            )
    return samples
