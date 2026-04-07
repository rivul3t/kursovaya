from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .backends import LLMBackend
from .models import GeneratedSample, HotpotDoc, HotpotExample
from .parsing import extract_document, extract_sentence, extract_tagged_text, extract_three_documents
from .prompts import (
    choose_sentence_prompt,
    conditional_docs_prompt,
    contradict_statement_prompt,
    expand_self_contradiction_prompt,
    pair_document_prompt,
)


def supporting_sentence_lookup(example: HotpotExample) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for title, sent_id in example.supporting_facts:
        mapping.setdefault(title, []).append(sent_id)
    return mapping


def choose_anchor_sentence(
    example: HotpotExample,
    rng: random.Random,
    importance: str = "most",
) -> Tuple[int, int, str]:
    if not example.docs:
        raise ValueError(f"Example {example.source_id} has no documents")

    sf_map = supporting_sentence_lookup(example)
    candidates: List[Tuple[int, int, str]] = []
    for di, doc in enumerate(example.docs):
        for si, sent in enumerate(doc.sentences):
            candidates.append((di, si, sent))

    if not candidates:
        raise ValueError(f"Example {example.source_id} has no sentences")

    if importance == "least":
        non_sf = [
            (di, si, sent)
            for di, doc in enumerate(example.docs)
            for si, sent in enumerate(doc.sentences)
            if not (doc.title in sf_map and si in sf_map[doc.title])
        ]
        if non_sf:
            non_sf.sort(key=lambda x: (len(x[2].split()), rng.random()))
            return non_sf[0]

    sf_candidates = [
        (di, si, sent)
        for di, doc in enumerate(example.docs)
        for si, sent in enumerate(doc.sentences)
        if (doc.title in sf_map and si in sf_map[doc.title])
    ]
    if sf_candidates:
        return rng.choice(sf_candidates)
    return rng.choice(candidates)


def _doc_dict(doc: HotpotDoc, text_override: Optional[str] = None) -> Dict[str, str]:
    return {"title": doc.title, "text": text_override if text_override is not None else doc.text()}


def _insert_document_at(docs: List[Dict[str, str]], new_doc: Dict[str, str], idx: int) -> List[Dict[str, str]]:
    return docs[:idx] + [new_doc] + docs[idx:]


def _resolve_insertion_index(n_docs: int, base_idx: int, cfg: str, rng: random.Random) -> int:
    if n_docs <= 0:
        return 0
    cfg = cfg.lower()
    if cfg in {"near", "contiguous"}:
        return min(n_docs, base_idx + 1)
    if cfg == "far":
        return 0 if base_idx > n_docs // 2 else n_docs
    if cfg == "separate":
        return rng.randint(0, n_docs)
    return min(n_docs, base_idx + 1)


def generate_negative_sample(example: HotpotExample, rng: random.Random) -> GeneratedSample:
    return GeneratedSample(
        sample_id=f"{example.source_id}-none-{rng.getrandbits(32):08x}",
        source_hotpot_id=example.source_id,
        conflict_type="none",
        documents=[_doc_dict(doc) for doc in example.docs],
        conflicting_doc_indices=[],
        metadata={
            "hotpot_question": example.question,
            "hotpot_answer": example.answer,
            "hotpot_type": example.qtype,
            "hotpot_level": example.level,
        },
    )


def generate_self_contradiction(
    example: HotpotExample,
    backend: LLMBackend,
    rng: random.Random,
    importance: str,
    length: str,
) -> GeneratedSample:
    doc_idx, sent_idx, anchor = choose_anchor_sentence(example, rng=rng, importance=importance)
    target_doc = example.docs[doc_idx]

    stmt_raw = backend.generate(choose_sentence_prompt(target_doc, importance), temperature=0.0, max_tokens=80)
    stmt = extract_sentence(stmt_raw) or anchor

    contrad_raw = backend.generate(contradict_statement_prompt(stmt), temperature=0.2, max_tokens=120)
    contradicted = extract_tagged_text(contrad_raw, "statement") or contrad_raw.strip()

    final_raw = backend.generate(
        expand_self_contradiction_prompt(target_doc, stmt, contradicted, length),
        temperature=0.5,
        max_tokens=320,
    )
    final_text = extract_document(final_raw) or f"{target_doc.text()} {contradicted}"

    documents = [_doc_dict(doc) for doc in example.docs]
    documents[doc_idx] = _doc_dict(target_doc, final_text)

    return GeneratedSample(
        sample_id=f"{example.source_id}-self-{rng.getrandbits(32):08x}",
        source_hotpot_id=example.source_id,
        conflict_type="self",
        documents=documents,
        conflicting_doc_indices=[doc_idx],
        metadata={
            "anchor_sentence": anchor,
            "selected_sentence": stmt,
            "contradictory_statement": contradicted,
            "importance": importance,
            "length": length,
            "hotpot_question": example.question,
            "hotpot_answer": example.answer,
            "hotpot_type": example.qtype,
            "hotpot_level": example.level,
            "selected_sentence_index": sent_idx,
        },
    )


def generate_pair_contradiction(
    example: HotpotExample,
    backend: LLMBackend,
    rng: random.Random,
    importance: str,
    length: str,
    position_cfg: str,
) -> GeneratedSample:
    base_idx, sent_idx, anchor = choose_anchor_sentence(example, rng=rng, importance=importance)
    base_doc = example.docs[base_idx]

    stmt_raw = backend.generate(choose_sentence_prompt(base_doc, importance), temperature=0.0, max_tokens=80)
    stmt = extract_sentence(stmt_raw) or anchor

    contrad_raw = backend.generate(contradict_statement_prompt(stmt), temperature=0.2, max_tokens=120)
    contradicted = extract_tagged_text(contrad_raw, "statement") or contrad_raw.strip()

    other_doc_raw = backend.generate(
        pair_document_prompt(base_doc, stmt, contradicted, length),
        temperature=0.5,
        max_tokens=300,
    )
    other_text = extract_document(other_doc_raw) or contradicted

    base_docs = [_doc_dict(doc) for doc in example.docs]
    insert_idx = _resolve_insertion_index(len(base_docs), base_idx, position_cfg, rng)

    synthetic_doc = {"title": f"synthetic_pair_{base_doc.title}", "text": other_text}
    new_docs = _insert_document_at(base_docs, synthetic_doc, insert_idx)

    if insert_idx <= base_idx:
        conflicted_indices = [base_idx + 1, insert_idx]
    else:
        conflicted_indices = [base_idx, insert_idx]

    return GeneratedSample(
        sample_id=f"{example.source_id}-pair-{rng.getrandbits(32):08x}",
        source_hotpot_id=example.source_id,
        conflict_type="pair",
        documents=new_docs,
        conflicting_doc_indices=sorted(conflicted_indices),
        metadata={
            "anchor_sentence": anchor,
            "selected_sentence": stmt,
            "contradictory_statement": contradicted,
            "importance": importance,
            "length": length,
            "position_cfg": position_cfg,
            "base_doc_index_before_insert": base_idx,
            "insert_index": insert_idx,
            "hotpot_question": example.question,
            "hotpot_answer": example.answer,
            "hotpot_type": example.qtype,
            "hotpot_level": example.level,
            "selected_sentence_index": sent_idx,
        },
    )


def generate_conditional_contradiction(
    example: HotpotExample,
    backend: LLMBackend,
    rng: random.Random,
    length: str,
    position_cfg: str,
) -> GeneratedSample:
    if not example.docs:
        raise ValueError(f"Example {example.source_id} has no documents")

    base_idx = rng.randrange(len(example.docs))
    topic_sentence = example.docs[base_idx].sentences[0]

    raw = backend.generate(conditional_docs_prompt(topic_sentence, length), temperature=0.6, max_tokens=450)
    d1, d2, d3 = extract_three_documents(raw)

    docs = [_doc_dict(doc) for doc in example.docs]
    insert_pos = _resolve_insertion_index(len(docs), base_idx, position_cfg, rng)

    if position_cfg == "separate":
        positions = sorted({rng.randint(0, len(docs)) for _ in range(3)})
        while len(positions) < 3:
            positions.append(rng.randint(0, len(docs)))
        positions = sorted(positions[:3])

        inserted = docs
        offset = 0
        for pos, txt, title in zip(positions, [d1, d2, d3], ["cond_1", "cond_2", "cond_3"]):
            inserted = _insert_document_at(inserted, {"title": title, "text": txt}, pos + offset)
            offset += 1
        new_docs = inserted
        conflicted_indices = positions
    else:
        inserted = docs
        for i, item in enumerate([
            {"title": "cond_1", "text": d1},
            {"title": "cond_2", "text": d2},
            {"title": "cond_3", "text": d3},
        ]):
            inserted = _insert_document_at(inserted, item, insert_pos + i)
        new_docs = inserted
        conflicted_indices = [insert_pos, insert_pos + 1, insert_pos + 2]

    return GeneratedSample(
        sample_id=f"{example.source_id}-cond-{rng.getrandbits(32):08x}",
        source_hotpot_id=example.source_id,
        conflict_type="conditional",
        documents=new_docs,
        conflicting_doc_indices=conflicted_indices,
        metadata={
            "topic_sentence": topic_sentence,
            "base_doc_index": base_idx,
            "position_cfg": position_cfg,
            "length": length,
            "hotpot_question": example.question,
            "hotpot_answer": example.answer,
            "hotpot_type": example.qtype,
            "hotpot_level": example.level,
        },
    )


def build_type_schedule(
    num_samples: int,
    none_ratio: float,
    self_ratio: float,
    pair_ratio: float,
    conditional_ratio: float,
    rng: random.Random,
) -> List[str]:
    ratios = [none_ratio, self_ratio, pair_ratio, conditional_ratio]
    names = ["none", "self", "pair", "conditional"]
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one ratio must be > 0")
    ratios = [r / total for r in ratios]

    counts = [int(num_samples * r) for r in ratios]
    while sum(counts) < num_samples:
        idx = max(range(4), key=lambda i: ratios[i] - (counts[i] / num_samples if num_samples else 0.0))
        counts[idx] += 1

    schedule = []
    for name, count in zip(names, counts):
        schedule.extend([name] * count)
    rng.shuffle(schedule)
    return schedule


def run_generation(
    examples: Sequence[HotpotExample],
    backend: LLMBackend,
    output_path,
    num_samples: int,
    seed: int,
    none_ratio: float,
    self_ratio: float,
    pair_ratio: float,
    conditional_ratio: float,
    importance: str,
    length: str,
    pair_position_cfg: str,
    conditional_position_cfg: str,
    max_retries: int,
) -> None:
    rng = random.Random(seed)
    schedule = build_type_schedule(
        num_samples=num_samples,
        none_ratio=none_ratio,
        self_ratio=self_ratio,
        pair_ratio=pair_ratio,
        conditional_ratio=conditional_ratio,
        rng=rng,
    )
    if not examples:
        raise ValueError("No HotpotQA examples loaded")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, target_type in enumerate(schedule, start=1):
            example = examples[rng.randrange(len(examples))]
            for attempt in range(1, max_retries + 1):
                try:
                    if target_type == "none":
                        sample = generate_negative_sample(example, rng)
                    elif target_type == "self":
                        sample = generate_self_contradiction(example, backend, rng, importance, length)
                    elif target_type == "pair":
                        sample = generate_pair_contradiction(example, backend, rng, importance, length, pair_position_cfg)
                    elif target_type == "conditional":
                        sample = generate_conditional_contradiction(example, backend, rng, length, conditional_position_cfg)
                    else:
                        raise ValueError(f"Unknown target_type: {target_type}")

                    f.write(sample.to_json() + "\n")
                    if i % 25 == 0 or i == num_samples:
                        print(f"[{i}/{num_samples}] wrote {sample.conflict_type} sample {sample.sample_id}")
                    break
                except Exception as exc:
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Failed to generate sample {i}/{num_samples} of type {target_type}: {exc}"
                        ) from exc
                    time.sleep(0.2 * attempt)
