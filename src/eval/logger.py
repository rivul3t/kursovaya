import json
from pathlib import Path
from datetime import datetime


class JsonlLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, data: dict):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # 👈 важно