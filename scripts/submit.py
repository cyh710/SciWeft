"""生成 test 预测并做提交 schema 校验（W1 占位提交 / W5 正式提交）。

用法：python scripts/submit.py --config configs/baseline.yaml --out experiments/submissions/final.csv
校验项（比赛硬要求）：
- 列名与顺序同 train_QA.csv；
- explanation 每行非空（空值过不了提交校验）；
- is_blank 行的 answer_value / ref_id / ref_url / supporting_materials 全部 is_blank；
- 引用的 ref_id 必须存在于 metadata.csv；
- answer_value 内不得含 < > ~。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.rag.evidence import select_evidence
from src.rag.generate import answer_question
from src.schemas import Answer, Chunk
from src.sparse import bm25, tfidf

SUBMIT_COLUMNS = ["id", "question", "answer", "answer_value", "answer_unit",
                  "ref_id", "ref_url", "supporting_materials", "explanation"]
BLANK = "is_blank"


def validate(pred: pd.DataFrame, valid_refs: set[str]) -> list[str]:
    problems: list[str] = []
    if list(pred.columns) != SUBMIT_COLUMNS:
        problems.append(f"列名/顺序不符: {list(pred.columns)}")
    for r in pred.itertuples(index=False):
        if not str(r.explanation).strip():
            problems.append(f"{r.id}: explanation 为空")
        if r.answer_value in ("", "nan", None):
            problems.append(f"{r.id}: answer_value 为空")
        if str(r.answer_value) != BLANK and any(c in str(r.answer_value) for c in "<>~"):
            problems.append(f"{r.id}: answer_value 含禁用符号 < > ~")
        if r.answer_value == BLANK:
            for col in ("ref_id", "ref_url", "supporting_materials"):
                if str(getattr(r, col)) != BLANK:
                    problems.append(f"{r.id}: 拒答行 {col} 未置为 {BLANK}")
        else:
            for ref in str(r.ref_id).strip("[]").split(", "):
                ref = ref.strip("'\" ")
                if ref and ref != BLANK and ref not in valid_refs:
                    problems.append(f"{r.id}: 引用 {ref} 不在 metadata.csv")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    chunks = []
    with open(REPO / cfg["data"]["processed_dir"] / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**__import__("json").loads(line)))
    chunks_by_id = {c.chunk_id: c for c in chunks}

    retriever = {"tfidf": tfidf, "bm25": bm25}[cfg["retrieval"]["method"]]
    retriever.build(chunks)
    r = cfg["retrieval"]
    ev_cfg = {"strategy": r.get("evidence_strategy", "top_chunks"),
              "n_evidence": r.get("evidence_k", cfg["rerank"]["top_k_after"]),
              "n_docs": r.get("evidence_docs", 4),
              "per_doc": r.get("evidence_per_doc", 2),
              "keep_neighbors": r.get("keep_neighbor_chunks", True),
              "neighbor_words": r.get("neighbor_words", 200)}

    meta = pd.read_csv(REPO / "data" / "raw" / "metadata.csv", encoding="utf-8-sig")
    url_map = dict(zip(meta["id"], meta["url"]))
    valid_refs = set(meta["id"])

    qs = pd.read_csv(REPO / "data" / "raw" / "test_Q.csv", encoding="utf-8-sig")
    if args.limit:
        qs = qs.head(args.limit)

    rows, n_fail = [], 0
    for q in qs.itertuples(index=False):
        hits = retriever.retrieve(q.question, top_k=r["top_k"])
        ev_hits, ev_texts = select_evidence(hits, chunks_by_id, **ev_cfg)
        try:
            ans: Answer = answer_question(q.question, ev_hits, ev_texts, unit_hint=str(q.answer_unit))
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            ans = Answer(question_id="", value=BLANK, unit=BLANK,
                         answer_text="Unable to answer with confidence based on the provided documents.",
                         citation_ref_ids=[], evidence=BLANK,
                         explanation=f"generation error: {exc!r}"[:300], is_blank=True)
        refs = ans.citation_ref_ids
        rows.append({
            "id": q.id, "question": q.question, "answer": ans.answer_text,
            "answer_value": ans.value, "answer_unit": ans.unit,
            "ref_id": str(refs) if refs else BLANK,
            "ref_url": (url_map.get(refs[0], "") if len(refs) == 1 else
                        (str([url_map.get(x, "") for x in refs]) if refs else BLANK)),
            "supporting_materials": ans.evidence, "explanation": ans.explanation,
        })
        print(f"[{len(rows)}/{len(qs)}] {q.id}: value={ans.value!r} refs={refs}")

    pred = pd.DataFrame(rows, columns=SUBMIT_COLUMNS)
    problems = validate(pred, valid_refs)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"-> {out}")
    if n_fail:
        print(f"[warn] {n_fail} 题生成失败（已按 is_blank 兜底）")
    if problems:
        print(f"[FAIL] schema 校验发现 {len(problems)} 个问题：")
        for p in problems[:20]:
            print("   ", p)
        sys.exit(1)
    print("[ok] schema 校验通过")


if __name__ == "__main__":
    main()
