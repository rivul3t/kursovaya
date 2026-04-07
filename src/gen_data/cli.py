from __future__ import annotations

import argparse
from pathlib import Path

from .backends import GroqBackend, MockBackend, OpenAICompatibleBackend
from .generation import run_generation
from .loaders import load_hotpot_examples


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic contradiction dataset on HotpotQA")
    parser.add_argument("--input-json", type=str, default=None, help="Path to HotpotQA JSON file")
    parser.add_argument("--input-parquet", type=str, default=None, help="Path to HotpotQA parquet file")
    parser.add_argument("--hf-split", type=str, default=None, help="Optional HF split name if loading from datasets")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--num-samples", type=int, default=1867, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["mock", "openai", "groq"], default="mock")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Model name for OpenAI/Groq backend",
    )
    parser.add_argument("--none-ratio", type=float, default=0.3749)
    parser.add_argument("--self-ratio", type=float, default=0.2630)
    parser.add_argument("--pair-ratio", type=float, default=0.1907)
    parser.add_argument("--conditional-ratio", type=float, default=0.1714)
    parser.add_argument("--importance", choices=["most", "least"], default="most")
    parser.add_argument("--length", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--pair-position-cfg", choices=["near", "far"], default="near")
    parser.add_argument("--conditional-position-cfg", choices=["contiguous", "separate"], default="contiguous")
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args(argv)


def build_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return MockBackend(seed=args.seed)
    if args.backend == "openai":
        return OpenAICompatibleBackend(model=args.model)
    if args.backend == "groq":
        return GroqBackend(model=args.model)
    raise ValueError(f"Unsupported backend: {args.backend}")


def main(argv=None) -> int:
    args = parse_args(argv)
    examples = load_hotpot_examples(args.input_json, args.input_parquet, args.hf_split)
    backend = build_backend(args)
    run_generation(
        examples=examples,
        backend=backend,
        output_path=Path(args.output),
        num_samples=args.num_samples,
        seed=args.seed,
        none_ratio=args.none_ratio,
        self_ratio=args.self_ratio,
        pair_ratio=args.pair_ratio,
        conditional_ratio=args.conditional_ratio,
        importance=args.importance,
        length=args.length,
        pair_position_cfg=args.pair_position_cfg,
        conditional_position_cfg=args.conditional_position_cfg,
        max_retries=args.max_retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
