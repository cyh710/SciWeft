"""问答管线装配层：cfg -> 组件装配 -> 单题推理（run_eval / submit / diagnose 共用）。

此前 submit.py 是 W1 占位实现：无查询增强、无重排、检索器只支持 bm25/tfidf，
与 run_eval.py 是两条不同的管线——同一配置产出的提交文件复现不出 run_eval
测得的分数。本模块把装配收敛到一处，两个入口只负责 I/O 与格式化。

验收：同一配置下 run_eval 的 predictions.csv 与 submit 的产出逐行一致。
"""
from __future__ import annotations

from pathlib import Path

from src.augment.query_augment import augment_queries, augment_queries_adaptive, fuse_variant_hits
from src.rag import llm
from src.rag.evidence import select_evidence
from src.rag.generate import answer_question
from src.rag.rerank import rerank
from src.schemas import Answer, Chunk, Hit
from src.sparse import bm25, tfidf
from src.dense import dense, hybrid

SUBMIT_COLUMNS = ["id", "question", "answer", "answer_value", "answer_unit",
                  "ref_id", "ref_url", "supporting_materials", "explanation"]


def build_retriever(method: str, chunks: list[Chunk], cache_dir: Path):
    """按方法名构建并索引检索器，返回模块（retrieve/build 走模块级函数，与历史一致）。"""
    if method not in ("tfidf", "bm25", "dense", "hybrid"):
        raise ValueError(f"未知检索方法: {method!r}（可选 tfidf|bm25|dense|hybrid）")
    mod = {"tfidf": tfidf, "bm25": bm25, "dense": dense, "hybrid": hybrid}[method]
    mod.build(chunks, *( [cache_dir] if method in ("dense", "hybrid") else [] ))
    return mod


def hybrid_kwargs(rcfg: dict) -> dict:
    """hybrid 检索的加权 RRF 参数（非 hybrid 方法返回空 dict）。"""
    if rcfg.get("method") != "hybrid":
        return {}
    return {"w_sparse": rcfg.get("fusion_w_sparse", 0.5),
            "w_dense": rcfg.get("fusion_w_dense", 0.5),
            "rrf_k": rcfg.get("rrf_k", 60)}


class QAPipeline:
    """一次装配、逐题运行。装配成本（索引 + 模型加载）只付一次。"""

    def __init__(self, cfg: dict, chunks: list[Chunk], cache_dir: Path | None = None):
        self.cfg = cfg
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        rcfg = cfg["retrieval"]
        self.method = rcfg["method"]
        self.retriever = build_retriever(self.method, chunks, cache_dir or Path("data/cache"))
        self.top_k = rcfg["top_k"]
        self.retrieve_kwargs = hybrid_kwargs(rcfg)
        self.ev_cfg = {
            "strategy": rcfg.get("evidence_strategy", "top_chunks"),
            "n_evidence": rcfg.get("evidence_k", cfg.get("rerank", {}).get("top_k_after", 8)),
            "n_docs": rcfg.get("evidence_docs", 4),
            "per_doc": rcfg.get("evidence_per_doc", 2),
            "keep_neighbors": rcfg.get("keep_neighbor_chunks", True),
            "neighbor_words": rcfg.get("neighbor_words", 200),
            "max_total_words": rcfg.get("max_total_words"),
        }
        self.rerank_cfg = cfg.get("rerank", {})
        aug_cfg = cfg.get("augment", {})
        self.augment_enabled = bool(aug_cfg.get("enabled"))
        self.augment_strategies = aug_cfg.get("strategies", [])
        gen = cfg.get("generation", {})
        llm.configure(cache_dir=gen.get("cache_dir"), max_retries=gen.get("max_retries"),
                      temperature=gen.get("temperature"))

    def run(self, question: str, unit_hint: str = "") -> Answer:
        """单题全链路：增强 RRF 融合 -> 重排 -> 证据窗口 -> 生成。失败由调用方兜底。"""
        if self.augment_enabled:
            variants = (augment_queries_adaptive(question) if self.augment_strategies == ["adaptive"]
                        else augment_queries(question, self.augment_strategies))
            hit_lists = [self.retriever.retrieve(v["text"], top_k=self.top_k, **self.retrieve_kwargs)
                         for v in variants]
            hits = fuse_variant_hits(hit_lists, top_k=self.top_k)
        else:
            hits = self.retriever.retrieve(question, top_k=self.top_k, **self.retrieve_kwargs)
        if self.rerank_cfg.get("enabled"):
            hits = rerank(question, hits, self.chunks_by_id,
                          top_k_after=self.rerank_cfg.get("top_k_after", 8))
        ev_hits, ev_texts = select_evidence(hits, self.chunks_by_id, **self.ev_cfg)
        return answer_question(question, ev_hits, ev_texts, unit_hint=unit_hint)


def fallback_answer(unit_hint: str = "", reason: str = "") -> Answer:
    """单题失败兜底：按官方约定整行拒答，但保留题面给定单位。

    与 generate.answer_question 的拒答路径保持同一单位口径——异常兜底此前把
    unit 写成 is_blank，与正常拒答不一致，评分时额外丢分。
    """
    unit = unit_hint.strip() if unit_hint and unit_hint not in ("is_blank", "nan", "") else "is_blank"
    return Answer(
        question_id="", value="is_blank", unit=unit,
        answer_text="Unable to answer with confidence based on the provided documents.",
        citation_ref_ids=[], evidence="is_blank",
        explanation=(reason or "generation failed")[:300], is_blank=True,
    )


def ref_id_field(refs: list[str]) -> str:
    """照 train_QA 约定：单引用也是字符串化列表。"""
    return str(refs) if refs else "is_blank"


def ref_url_field(refs: list[str], url_map: dict[str, str]) -> str:
    if not refs:
        return "is_blank"
    urls = [url_map.get(r, "") for r in refs]
    return urls[0] if len(urls) == 1 else str(urls)
