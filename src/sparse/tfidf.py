"""TF-IDF + cosine 检索器（课程基线 M1）。"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.schemas import Chunk, Hit

TOKEN_RE = re.compile(r"[a-z0-9]+")

_corpus_matrix: np.ndarray | None = None
_vectorizer: TfidfVectorizer | None = None
_chunks: list[Chunk] | None = None


def _tokenize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def build(chunks: list[Chunk]) -> None:
    global _corpus_matrix, _vectorizer, _chunks
    _chunks = chunks
    _vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+", sublinear_tf=True)
    _corpus_matrix = normalize(_vectorizer.fit_transform([_tokenize(c.text) for c in chunks]))


def retrieve(query: str, top_k: int = 50) -> list[Hit]:
    assert _corpus_matrix is not None, "先调用 build()"
    q = normalize(_vectorizer.transform([_tokenize(query)]))
    scores = (_corpus_matrix @ q.T).toarray().ravel()
    order = np.argsort(-scores)[:top_k]
    return [
        Hit(chunk_id=_chunks[i].chunk_id, ref_id=_chunks[i].ref_id,
            score=float(scores[i]), rank=rank + 1, snippet=_chunks[i].text[:200])
        for rank, i in enumerate(order) if scores[i] > 0
    ]
