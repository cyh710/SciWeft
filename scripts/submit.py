"""生成 test 预测并做提交 schema 校验（W1 占位提交 / W5 正式提交）。

用法：python scripts/submit.py --config configs/baseline.yaml --out experiments/submissions/final.csv
管线与 run_eval.py 完全共用（src/rag/pipeline.py）：同一配置下两入口产出逐行一致，
增强 / 重排 / dense / hybrid 配置在提交时同样生效。

校验项（比赛硬要求）：
- 列名与顺序同 train_QA.csv；
- explanation 每行非空（空值过不了提交校验）；
- is_blank 行的 answer_value / ref_id / ref_url / supporting_materials 全部 is_blank；
- 引用的 ref_id 必须存在于 metadata.csv；
- answer_value 内不得含 < > ~。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.rag.pipeline import QAPipeline, SUBMIT_COLUMNS, fallback_answer, ref_id_field, ref_url_field
from src.schemas import Chunk

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
            chunks.append(Chunk(**json.loads(line)))
    pipe = QAPipeline(cfg, chunks, cache_dir=REPO / "data" / "cache")

    meta = pd.read_csv(REPO / "data" / "raw" / "metadata.csv", encoding="utf-8-sig")
    url_map = dict(zip(meta["id"], meta["url"]))
    valid_refs = set(meta["id"])

    qs = pd.read_csv(REPO / "data" / "raw" / "test_Q.csv", encoding="utf-8-sig")
    if args.limit:
        qs = qs.head(args.limit)

    rows, n_fail = [], 0
    for q in qs.itertuples(index=False):
        unit_hint = str(getattr(q, "answer_unit", ""))
        try:
            ans = pipe.run(q.question, unit_hint=unit_hint)
        except Exception as exc:  # noqa: BLE001  单题失败不中断全跑
            n_fail += 1
            ans = fallback_answer(unit_hint, reason=f"generation error: {exc!r}")
        rows.append({
            "id": q.id, "question": q.question, "answer": ans.answer_text,
            "answer_value": ans.value, "answer_unit": ans.unit,
            "ref_id": ref_id_field(ans.citation_ref_ids),
            "ref_url": ref_url_field(ans.citation_ref_ids, url_map),
            "supporting_materials": ans.evidence, "explanation": ans.explanation,
        })
        print(f"[{len(rows)}/{len(qs)}] {q.id}: value={ans.value!r} refs={ans.citation_ref_ids}")

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
