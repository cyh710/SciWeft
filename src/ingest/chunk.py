"""chunk_documents：逐页文本 -> 跨页固定长度分块（W1）。

策略：按空白分词近似 token 计数，
chunk_size=400 词（约 500 tokens），overlap=40 词。块跨页时记录 page_start/page_end，
保证证据可回溯到页码。chunk_id = {ref_id}:{seq}，唯一且持久。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.schemas import Chunk

WORD_RE = re.compile(r"\S+")

DEFAULT_CHUNK_WORDS = 400
DEFAULT_OVERLAP_WORDS = 40


def _iter_page_words(pages: list[str]):
    """yield (word, page_index)，跳过空页。"""
    for page_idx, text in enumerate(pages):
        for word in WORD_RE.findall(text):
            yield word, page_idx


def chunk_document(ref_id: str, pages: list[str], chunk_words: int, overlap_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[tuple[str, int]] = []
    seq = 0
    step = max(chunk_words - overlap_words, 1)
    for item in _iter_page_words(pages):
        buffer.append(item)
        if len(buffer) >= chunk_words:
            words = [w for w, _ in buffer]
            page_start = buffer[0][1]
            page_end = buffer[-1][1]
            chunks.append(Chunk(
                chunk_id=f"{ref_id}:{seq}",
                ref_id=ref_id,
                text=" ".join(words),
                page_start=page_start,
                page_end=page_end,
            ))
            seq += 1
            buffer = buffer[step:]  # 保留 overlap 部分，跨块共享上下文
    if buffer and (not chunks or len(buffer) > overlap_words):  # 尾块：太短并与上块重叠过多时并入上块
        words = [w for w, _ in buffer]
        if chunks and len(buffer) <= overlap_words:
            last = chunks[-1]
            last.text += " " + " ".join(words)
            last.page_end = buffer[-1][1]
        else:
            chunks.append(Chunk(
                chunk_id=f"{ref_id}:{seq}",
                ref_id=ref_id,
                text=" ".join(words),
                page_start=buffer[0][1],
                page_end=buffer[-1][1],
            ))
    return chunks


def chunk_pages_dir(pages_dir: Path, chunk_words: int = DEFAULT_CHUNK_WORDS,
                    overlap_words: int = DEFAULT_OVERLAP_WORDS) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for pages_json in sorted(pages_dir.glob("*.json")):
        if pages_json.name.startswith("_"):
            continue
        data = json.loads(pages_json.read_text(encoding="utf-8"))
        all_chunks.extend(chunk_document(data["ref_id"], data["pages"], chunk_words, overlap_words))
    return all_chunks
