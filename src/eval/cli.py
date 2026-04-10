
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import GroqBackend, MockBackend, OpenAIBackend, GeminiBackend
from .evaluator import HotpotEvaluator
from .loaders import load_jsonl
from .logger import JsonlLogger

logger = JsonlLogger("logs/run.jsonl")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate models on Hotpot contradiction tasks')
    parser.add_argument('--input-jsonl', type=str, required=True, help='Path to generated dataset JSONL')
    parser.add_argument('--output-json', type=str, required=True, help='Where to store metrics')
    parser.add_argument("--backend", choices=["mock", "openai", "groq", "gemini"], default="mock")
    parser.add_argument('--base-url', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini')
    parser.add_argument('--prompt-style', choices=['basic', 'cot'], default='basic')
    return parser.parse_args(argv)


def build_backend(args: argparse.Namespace):
    if args.backend == 'mock':
        return MockBackend()
    if args.backend == 'openai':
        return OpenAIBackend(model=args.model, base_url=args.base_url)
    if args.backend == 'groq':
        return GroqBackend(model=args.model)
    if args.backend == "gemini":
        return GeminiBackend(model=args.model)
    raise ValueError(f'Unsupported backend: {args.backend}')


def main(argv=None) -> int:
    args = parse_args(argv)
    samples = load_jsonl(args.input_jsonl)
    backend = build_backend(args)
    evaluator = HotpotEvaluator(llm=backend, prompt_style=args.prompt_style, logger=logger)
    metrics = evaluator.evaluate_all(samples)

    out = Path(args.output_json)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
