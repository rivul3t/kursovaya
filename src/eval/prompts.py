from __future__ import annotations

from .types import EvalSample


def format_documents(sample: EvalSample) -> str:
    chunks = []
    for i, doc in enumerate(sample.documents):
        title = str(doc.get("title", f"doc_{i}"))
        text = str(doc.get("text", "")).strip()
        chunks.append(f"[DOC {i}] {title}\n{text}")
    return "\n\n".join(chunks)


def build_detection_prompt(sample: EvalSample, prompt_style: str = "basic") -> str:
    docs = format_documents(sample)

    if prompt_style == "cot":
        return (
            "You are a context validator for retrieved documents.\n"
            "Task: decide whether there is any contradiction in the set of documents.\n"
            "Think step by step, then answer with exactly one line: YES or NO.\n\n"
            f"Documents:\n{docs}"
        )

    return (
        "You are a context validator for retrieved documents.\n"
        "Task: decide whether there is any contradiction in the set of documents.\n"
        "Answer with exactly one word: YES or NO.\n\n"
        f"Documents:\n{docs}"
    )


def build_type_prompt(sample: EvalSample, prompt_style: str = "basic") -> str:
    docs = format_documents(sample)

    if prompt_style == "cot":
        return (
            "You are a context validator for retrieved documents.\n"
            "Task: identify the conflict type if any. Labels: none, self, pair, conditional.\n"
            "1. Self-Contradiction: Conflicting information within a single document.\n"
            "2. Pair Contradiction: Conflicting information between two documents.\n"
            "3. Conditional Contradiction: Three documents where the third \n"
            "document makes the first two contradict each other.\n"
            "Think step by step, then answer with exactly one label.\n\n"
            f"Documents:\n{docs}"
        )

    return (
        "You are a context validator for retrieved documents.\n"
        "Task: identify the conflict type if any. Labels: none, self, pair, conditional.\n"
        "1. Self-Contradiction: Conflicting information within a single document.\n"
        "2. Pair Contradiction: Conflicting information between two documents.\n"
        "3. Conditional Contradiction: Three documents where the third \n"
        "document makes the first two contradict each other.\n"
        "Answer with exactly one label.\n\n"
        f"Documents:\n{docs}"
    )


def build_segmentation_prompt(
    sample: EvalSample,
    guided: bool,
    prompt_style: str = "basic",
) -> str:
    docs = format_documents(sample)

    conflict_type = sample.conflict_type if guided else "unknown"

    return (
        "Given a set of documents and a known conflict type, your task is to identify "
        "which document(s) contain the conflicting information.\n\n"

        f"Conflict Type: {conflict_type}\n\n"

        "Instructions:\n"
        "1. Carefully read all the provided documents.\n"
        "2. Keep in mind the given conflict type or detect by yourself if type is unknown.\n"
        "3. Analyze the content to identify which document(s) contribute to the specified type of contradiction.\n"
        "4. List the numbers of the documents that contain the conflicting information.\n"
        "5. Think step by step before answering.\n\n"

        "Your response should be in the following format:\n"
        "<documents>[List the numbers of the documents, separated by commas]</documents>\n\n"

        "Definitions of Conflict Types:\n"
        "- Self-Contradiction: Conflicting information within a single document.\n"
        "- Pair Contradiction: Conflicting information between two documents.\n"
        "- Conditional Contradiction: Three documents where the third document makes the first two contradict each other, although they don’t contradict directly.\n\n"

        f"Here are the documents:\n{docs}"
    )