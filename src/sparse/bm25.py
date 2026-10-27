"""BM25 检索器（M2，rank-bm25）。"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.schemas import Chunk, Hit

TOKEN_RE = re.compile(r"[a-z0-9]+")

_bm25: BM25Okapi | None = None
_chunks: list[Chunk] | None = None


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build(chunks: list[Chunk]) -> None:
    global _bm25, _chunks
    _chunks = chunks
    _bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])


def retrieve(query: str, top_k: int = 50) -> list[Hit]:
    assert _bm25 is not None, "先调用 build()"
    scores = _bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [
        Hit(chunk_id=_chunks[i].chunk_id, ref_id=_chunks[i].ref_id,
            score=float(scores[i]), rank=rank + 1, snippet=_chunks[i].text[:200])
        for rank, i in enumerate(order) if scores[i] > 0
    ]
