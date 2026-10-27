"""分块不变量单测：chunk_id 唯一、可回溯 ref_id/页码、长度受控、全文近似无损。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.chunk import chunk_document


def make_pages(n_pages: int, words_per_page: int, marker_prefix="p"):
    return [" ".join(f"{marker_prefix}{pi}w{wi}" for wi in range(words_per_page)) for pi in range(n_pages)]


def test_chunk_ids_unique_and_traceable():
    chunks = chunk_document("doc1", make_pages(10, 120), chunk_words=400, overlap_words=40)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.ref_id == "doc1" for c in chunks)
    assert all(c.page_start <= c.page_end for c in chunks)


def test_chunk_length_bounded():
    chunks = chunk_document("doc2", make_pages(30, 100), chunk_words=400, overlap_words=40)
    # 每块 ≤ 400 词（尾块并入上块时允许略超）
    assert all(len(c.text.split()) <= 440 for c in chunks)


def test_content_nearly_lossless():
    pages = make_pages(5, 200)
    total_words = 5 * 200
    chunks = chunk_document("doc3", pages, chunk_words=400, overlap_words=40)
    words_in_chunks = sum(len(c.text.split()) for c in chunks)
    # overlap 会带来重复，但信息不能丢：并集词数应等于原文
    joined = " ".join(c.text for c in chunks).split()
    assert len(joined) >= total_words
    assert set(joined) == {w for page in pages for w in page.split()}


def test_short_doc_single_chunk():
    chunks = chunk_document("doc4", ["only a few words here"], chunk_words=400, overlap_words=40)
    assert len(chunks) == 1
    assert chunks[0].text == "only a few words here"
