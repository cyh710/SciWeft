"""修复项回归测试（自包含，不依赖竞赛数据）。

覆盖：split 的 NaN 旗标、检索器未初始化的显式异常、dense 缓存键内容哈希、
llm 缓存键含 base_url 与 configure() 注入、generate 的 JSON 提取与解析失败信号、
pipeline 装配与失败兜底、bootstrap 重采样的重复保留。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.schemas import Answer, Chunk, Hit


# ------------------------------------------------------------------ split.py：NaN 旗标
def test_primary_type_ignores_nan_flags():
    """旗标列存在但单元格为空（NaN）时按 0 处理，不得把题归进 is_NA。"""
    from src.eval.split import primary_type
    row = pd.Series({"is_NA": float("nan"), "Math": float("nan"), "Table": 1.0})
    assert primary_type(row) == "Table"
    row_all_nan = pd.Series({"is_NA": float("nan"), "Quote": float("nan")})
    assert primary_type(row_all_nan) == "Quote"
    assert primary_type(pd.Series({"is_NA": 1.0})) == "is_NA"


# ------------------------------------------------------------------ 检索器：显式异常
def test_retrievers_raise_runtime_error_before_build():
    import src.dense.dense as d
    import src.sparse.bm25 as b
    import src.sparse.tfidf as t
    for mod, state_attr in ((b, "_bm25"), (t, "_corpus_matrix"), (d, "_corpus")):
        saved = getattr(mod, state_attr)
        setattr(mod, state_attr, None)
        try:
            with pytest.raises(RuntimeError, match="build"):
                mod.retrieve("q")
        finally:
            setattr(mod, state_attr, saved)


# ------------------------------------------------------------------ dense.py：缓存键
def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, ref_id=cid.split(":")[0], text="x")


def test_dense_cache_key_is_content_hashed():
    from src.dense.dense import cache_key
    a = [_chunk("docA:0"), _chunk("docA:1")]
    assert cache_key(a) == cache_key([_chunk("docA:0"), _chunk("docA:1")])
    assert cache_key(a) != cache_key([_chunk("docA:0")])                       # 长度不同
    assert cache_key(a) != cache_key([_chunk("docB:0"), _chunk("docB:1")])     # 同长度不同内容


# ------------------------------------------------------------------ llm.py：缓存键与 configure
def test_llm_cache_key_includes_base_url(monkeypatch):
    from src.rag import llm
    monkeypatch.setattr(llm, "BASE_URL", "https://api.deepseek.com")
    k1 = llm._cache_key("s", [{"role": "user", "content": "q"}], 0.0, 100)
    monkeypatch.setattr(llm, "BASE_URL", "http://127.0.0.1:8765/v1")
    k2 = llm._cache_key("s", [{"role": "user", "content": "q"}], 0.0, 100)
    assert k1 != k2


def test_llm_configure_applies_generation_section(tmp_path):
    from src.rag import llm
    saved = (llm.CACHE_DIR, llm.MAX_RETRIES, llm.DEFAULT_TEMPERATURE)
    try:
        llm.configure(cache_dir=str(tmp_path / "llm"), max_retries=1, temperature=0.7)
        assert llm.CACHE_DIR == tmp_path / "llm"
        assert llm.MAX_RETRIES == 1
        assert llm.DEFAULT_TEMPERATURE == 0.7
    finally:
        llm.CACHE_DIR, llm.MAX_RETRIES, llm.DEFAULT_TEMPERATURE = saved


# ------------------------------------------------------------------ generate.py：JSON 提取
def test_extract_json_object_tolerates_prose_and_fences():
    from src.rag.generate import _extract_json_object
    assert _extract_json_object('{"a": 1}') == {"a": 1}
    assert _extract_json_object('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json_object('{"a": 1}\n\nNote: {"b": 2}') == {"a": 1}   # 取第一个完整对象
    assert _extract_json_object('no json at all') is None
    assert _extract_json_object('{"a": 1') is None                          # 半截 JSON


def test_generate_parse_failure_is_labeled_not_disguised(monkeypatch):
    """解析失败要保留信号（[parse-failed]），不能伪装成「语料无证据」。"""
    from src.rag import generate
    monkeypatch.setattr(generate, "chat", lambda *a, **k: "Sorry, not JSON.")
    ans = generate.answer_question("q", [_hit("gao2025:0")], {}, unit_hint="terawatt hours")
    assert ans.is_blank and ans.value == "is_blank"
    assert ans.explanation.startswith("[parse-failed]")
    assert ans.unit == "terawatt hours"   # 兜底行同样保留题面单位


def _hit(cid: str, rank: int = 1) -> Hit:
    return Hit(chunk_id=cid, ref_id=cid.split(":")[0], score=1.0, rank=rank, snippet="s")


# ------------------------------------------------------------------ pipeline.py
def _cfg(**over):
    cfg = {
        "run_name": "t",
        "retrieval": {"method": "bm25", "top_k": 3, "evidence_strategy": "top_chunks",
                      "evidence_k": 2, "keep_neighbor_chunks": False, "neighbor_words": 0},
        "rerank": {"enabled": False, "model": "m", "top_k_after": 2},
        "augment": {"enabled": False, "strategies": []},
        "generation": {"cache_dir": "", "max_retries": 1, "temperature": 0},
    }
    cfg.update(over)
    return cfg


class _StubRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query, top_k=50, **kw):
        self.calls.append((query, top_k))
        return self.hits[:top_k]


def test_pipeline_wires_augment_and_evidence(monkeypatch):
    """augment.enabled 时逐变体检索并做跨变体融合；证据窗口按 config 取数。"""
    from src.rag import pipeline
    chunks = [_chunk("d:0"), _chunk("d:1")]
    stub = _StubRetriever([_hit("d:0", 1), _hit("d:1", 2), _hit("d:0", 3)])
    monkeypatch.setattr(pipeline, "build_retriever", lambda m, c, cd: stub)
    variants = [{"text": "orig", "strategy": "original"}, {"text": "expanded", "strategy": "expansion"}]
    monkeypatch.setattr(pipeline, "augment_queries", lambda q, s: variants)

    cfg = _cfg()
    cfg["augment"] = {"enabled": True, "strategies": ["expansion"]}
    captured = {}
    monkeypatch.setattr(pipeline, "answer_question",
                        lambda q, hits, texts, unit_hint="": captured.update(hits=hits, texts=texts)
                        or Answer(question_id="", value="42"))
    pipe = pipeline.QAPipeline(cfg, chunks)
    pipe.run("q", unit_hint="GW")
    assert [q for q, _ in stub.calls] == ["orig", "expanded"]          # 原查询 + 变体都检索
    assert len(captured["hits"]) == cfg["retrieval"]["evidence_k"]     # 证据数来自 config


def test_pipeline_rerank_respects_config(monkeypatch):
    from src.rag import pipeline
    chunks = [_chunk("d:0"), _chunk("d:1")]
    stub = _StubRetriever([_hit("d:0", 1), _hit("d:1", 2)])
    monkeypatch.setattr(pipeline, "build_retriever", lambda m, c, cd: stub)
    seen = {}
    monkeypatch.setattr(pipeline, "rerank",
                        lambda q, hits, cbid, top_k_after=8: seen.update(top_k_after=top_k_after) or hits)
    monkeypatch.setattr(pipeline, "answer_question",
                        lambda q, hits, texts, unit_hint="": Answer(question_id="", value="42"))
    cfg = _cfg()
    cfg["rerank"] = {"enabled": True, "model": "m", "top_k_after": 7}
    pipeline.QAPipeline(cfg, chunks).run("q")
    assert seen["top_k_after"] == 7


def test_pipeline_passes_evidence_budget_from_config(monkeypatch):
    """retrieval.max_total_words 要真的传进 select_evidence；未配置时为 None（历史行为）。"""
    from src.rag import pipeline
    chunks = [_chunk("d:0"), _chunk("d:1")]
    stub = _StubRetriever([_hit("d:0", 1)])
    monkeypatch.setattr(pipeline, "build_retriever", lambda m, c, cd: stub)
    seen = {}

    def fake_select(hits, cbid, **kw):
        seen.update(kw)
        return hits, {}

    monkeypatch.setattr(pipeline, "select_evidence", fake_select)
    monkeypatch.setattr(pipeline, "answer_question",
                        lambda q, hits, texts, unit_hint="": Answer(question_id="", value="42"))
    cfg = _cfg()
    cfg["retrieval"]["max_total_words"] = 4000
    pipeline.QAPipeline(cfg, chunks).run("q")
    assert seen["max_total_words"] == 4000
    pipeline.QAPipeline(_cfg(), chunks).run("q")
    assert seen["max_total_words"] is None


def test_pipeline_build_retriever_rejects_unknown_method():
    from src.rag.pipeline import build_retriever
    with pytest.raises(ValueError):
        build_retriever("splade", [], REPO / "data" / "cache")


def test_fallback_answer_keeps_question_unit():
    from src.rag.pipeline import fallback_answer
    ans = fallback_answer("terawatt hours", reason="boom")
    assert ans.is_blank and ans.value == "is_blank" and ans.unit == "terawatt hours"
    assert ans.explanation.startswith("boom")
    assert fallback_answer("is_blank").unit == "is_blank"


# ------------------------------------------------------------------ bootstrap_ci.py
def _import_bootstrap():
    sys.path.insert(0, str(REPO / "scripts"))
    import bootstrap_ci
    return bootstrap_ci


def test_resample_ids_keeps_duplicates():
    bc = _import_bootstrap()
    import numpy as np
    ids = [f"q{i}" for i in range(64)]
    out = bc.resample_ids(ids, np.random.default_rng(42))
    assert len(out) == len(ids)
    assert len(set(out)) < len(ids)      # 有放回抽样必然（几乎必然）带重复；重复不得被折叠


def test_aligned_frames_aligns_pred_to_resampled_solution():
    bc = _import_bootstrap()
    sol = pd.DataFrame([{"id": f"q{i}", "answer_value": str(i)} for i in range(5)])
    pred = pd.DataFrame([{"id": f"q{i}", "answer_value": str(9 - i)} for i in range(5)])
    sample = ["q1", "q1", "q3"]
    s, p = bc.aligned_frames(sol, pred, sample)
    assert list(s["id"]) == ["q1__0", "q1__1", "q3__2"]   # 重复 id 加后缀保留
    assert list(p["id"]) == list(s["id"])                 # 预测行按重复 id 逐一对齐
    assert list(p["answer_value"]) == ["8", "8", "6"]     # q1→8, q1→8, q3→6
