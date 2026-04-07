from __future__ import annotations

from .models import HotpotDoc


def choose_sentence_prompt(doc: HotpotDoc, importance: str) -> str:
    return f'''Choose the {importance} sentence from the given document.
Only output the chosen sentence wrapped in <sentence></sentence> tags.

Document:
<title>{doc.title}</title>
<text>{doc.text()}</text>
'''


def contradict_statement_prompt(statement: str) -> str:
    return f'''Modify the given statement to suggest otherwise instead of the original.
Only output the modified statement wrapped in <statement></statement> tags.

Statement:
<statement>{statement}</statement>
'''


def expand_self_contradiction_prompt(original_doc: HotpotDoc, anchor: str, contradicted: str, length: str) -> str:
    return f'''Rewrite the document so that it remains fluent but contains a contradiction.
Keep the original topic and style. The contradictory part should be subtle.
Target length: {length}.

Original document:
<title>{original_doc.title}</title>
<text>{original_doc.text()}</text>

Anchor sentence:
<sentence>{anchor}</sentence>

Contradictory statement to include:
<statement>{contradicted}</statement>

Return only the final document text wrapped in <document></document> tags.
'''


def pair_document_prompt(base_doc: HotpotDoc, anchor: str, contradicted: str, length: str) -> str:
    return f'''Create a second document that contradicts the first document.
The new document should be coherent, fluent, and about the same topic.
Target length: {length}.

First document:
<title>{base_doc.title}</title>
<text>{base_doc.text()}</text>

Anchor sentence:
<sentence>{anchor}</sentence>

Contradictory statement to build around:
<statement>{contradicted}</statement>

Return only the new document text wrapped in <document></document> tags.
'''


def conditional_docs_prompt(topic_sentence: str, length: str) -> str:
    return f'''Generate three short documents on the same topic.
Constraints:
1) Document 1 and Document 2 must not contradict each other.
2) Document 3 must not directly contradict Document 1 or Document 2.
3) However, the information in Document 3 must make Document 1 and Document 2 mutually exclusive.
4) Keep them fluent and concise.
Target length: {length}.

Topic sentence:
<sentence>{topic_sentence}</sentence>

Return exactly three documents using these tags:
<document1>...</document1>
<document2>...</document2>
<document3>...</document3>
'''
