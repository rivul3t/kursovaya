
from __future__ import annotations

from typing import Dict, List

from .backends import LLMBackend
from .metrics import detection_metrics, segmentation_metrics, type_metrics
from .parsing import parse_conflict_type, parse_indices, parse_yes_no
from .prompts import build_detection_prompt, build_segmentation_prompt, build_type_prompt
from .types import EvalSample


class HotpotEvaluator:
    def __init__(self, llm: LLMBackend, prompt_style: str = 'basic') -> None:
        self.llm = llm
        self.prompt_style = prompt_style

    def predict_detection(self, sample: EvalSample) -> bool:
        prompt = build_detection_prompt(sample, prompt_style=self.prompt_style)
        raw = self.llm.generate(prompt)
        return parse_yes_no(raw)

    def predict_type(self, sample: EvalSample) -> str:
        prompt = build_type_prompt(sample, prompt_style=self.prompt_style)
        raw = self.llm.generate(prompt)
        return parse_conflict_type(raw)

    def predict_segmentation(self, sample: EvalSample, guided: bool) -> List[int]:
        prompt = build_segmentation_prompt(sample, guided=guided, prompt_style=self.prompt_style)
        raw = self.llm.generate(prompt)
        return parse_indices(raw)

    def evaluate_detection(self, samples: List[EvalSample]) -> Dict[str, float]:
        y_true = [s.conflict_type != 'none' for s in samples]
        y_pred = [self.predict_detection(s) for s in samples]
        return detection_metrics(y_true, y_pred)

    def evaluate_type(self, samples: List[EvalSample]) -> Dict[str, float]:
        y_true = [s.conflict_type for s in samples]
        y_pred = [self.predict_type(s) for s in samples]
        return type_metrics(y_true, y_pred)

    def evaluate_segmentation(self, samples: List[EvalSample], guided: bool) -> Dict[str, float]:
        y_true = [s.conflicting_doc_indices for s in samples]
        y_pred = [self.predict_segmentation(s, guided=guided) for s in samples]
        return segmentation_metrics(y_true, y_pred)

    def evaluate_all(self, samples: List[EvalSample]) -> Dict[str, Dict[str, float]]:
        return {
            'conflict_detection': self.evaluate_detection(samples),
            'type_detection': self.evaluate_type(samples),
            'segmentation_guided': self.evaluate_segmentation(samples, guided=True),
            'segmentation_blind': self.evaluate_segmentation(samples, guided=False),
        }
