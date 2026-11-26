"""Question answering grounded in the text of one uploaded agreement."""
import json
import re

import faiss
import numpy as np
import streamlit as st
from groq import BadRequestError
from sentence_transformers import SentenceTransformer

from pipeline import AnalysisError, _complete

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models retrieve better when queries (but not passages) carry this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_TARGET = 650
# Only near-empty sections (a stray heading or title) are folded into the next one.
MIN_SECTION = 40
TOP_K = 4

HEADING_LINE = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVX]{1,4}\.|[A-Z]\.)\s+[A-Z][^.]{1,70}\.?$")
SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z0-9(\"“'])")


def _is_heading(line):
    return len(line) <= 80 and bool(HEADING_LINE.match(line)) and not line.rstrip().endswith((",", ";"))


def _sections(text):
    """Group lines under the numbered headings they follow; text without line breaks is one section."""
    sections, heading, lines = [], "", []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _is_heading(line):
            if lines or heading:
                sections.append((heading, " ".join(lines)))
            heading, lines = line.rstrip("."), []
        else:
            lines.append(line)
    if lines or heading:
        sections.append((heading, " ".join(lines)))
    return sections


def _split_long(body):
    sentences = [s.strip() for s in SENTENCE_END.split(" ".join(body.split())) if s.strip()]
    pieces, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > CHUNK_TARGET:
            pieces.append(" ".join(current))
            current = current[-1:]
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(text):
    """Split an agreement into passages that follow its own clause structure where it has one."""
    merged = []
    for heading, body in _sections(text):
        if merged and len(merged[-1][1]) < MIN_SECTION:
            prev_heading, prev_body = merged.pop()
            if heading:
                merged.append((heading, " ".join(part for part in (prev_body, body) if part)))
            else:
                merged.append((prev_heading, " ".join(part for part in (prev_body, body) if part)))
        else:
            merged.append((heading, body))

    chunks = []
    for heading, body in merged:
        for piece in _split_long(body) or [""]:
            if piece or heading:
                chunks.append({"heading": heading, "text": piece})
    return chunks


@st.cache_resource(show_spinner=False)
def _embedder():
    # Prefer the local copy so the feature works offline once the model has been downloaded.
    try:
        return SentenceTransformer(EMBED_MODEL, device="cpu", local_files_only=True)
    except Exception:
        pass
    try:
        return SentenceTransformer(EMBED_MODEL, device="cpu")
    except Exception as exc:
        raise AnalysisError(
            "Document questions need a small language model (about 130 MB) that is downloaded once. "
            "It could not be downloaded, so check the internet connection and try again."
        ) from exc


def _embed(texts, query=False):
    prefix = QUERY_PREFIX if query else ""
    vectors = _embedder().encode(
        [prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")


def _passage_text(chunk):
    return f'{chunk["heading"]}. {chunk["text"]}' if chunk["heading"] else chunk["text"]


@st.cache_resource(show_spinner=False, max_entries=32)
def _document_index(digest, _chunks):
    vectors = _embed([_passage_text(c) for c in _chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(question, digest, chunks, k=TOP_K):
    if not chunks:
        return []
    index = _document_index(digest, chunks)
    scores, ids = index.search(_embed([question], query=True), min(k, len(chunks)))
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def _ask_model(messages):
    for _ in range(2):
        try:
            parsed = json.loads(_complete(messages, temperature=0.1, json_mode=True))
        except (BadRequestError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AnalysisError("The answer could not be prepared. Please ask the question again.")


def answer_question(question, digest, text):
    chunks = chunk_document(text)
    hits = retrieve(question, digest, chunks)
    if not hits:
        return {"question": question, "answered": False, "answer": "This document has no readable text to search.",
                "passages": []}

    passages = [chunks[i] for i, _ in hits]
    listing = "\n\n".join(f"[{n}] {_passage_text(p)}" for n, p in enumerate(passages, 1))
    parsed = _ask_model([
        {
            "role": "system",
            "content": (
                "You answer questions about one specific agreement for a reader who is not a lawyer. "
                "Use ONLY the numbered passages from that agreement. Do not use outside knowledge about law, "
                "regulations or other contracts, and do not say whether anything is legal or fair. "
                "If the passages do not contain the answer, set answered to false and say briefly that this "
                "agreement does not address it. Otherwise answer in one to four plain sentences, quote figures "
                "exactly as written, and cite passage numbers like [2]. "
                'Return ONLY JSON: {"answered": true or false, "answer": "...", "sources": [passage numbers]}'
            ),
        },
        {"role": "user", "content": f"Passages from the agreement:\n\n{listing}\n\nQuestion: {question}\n\nJSON:"},
    ])

    answered = parsed.get("answered") is True
    answer = " ".join(str(parsed.get("answer", "")).split()) or "This agreement does not address that."
    cited = []
    for source in parsed.get("sources") or []:
        if isinstance(source, int) and not isinstance(source, bool) and 1 <= source <= len(passages) \
                and source not in cited:
            cited.append(source)
    if answered and not cited:
        cited = list(range(1, len(passages) + 1))

    return {
        "question": question,
        "answered": answered,
        "answer": answer,
        "passages": [{"n": n, **passages[n - 1]} for n in cited] if answered else [],
    }
