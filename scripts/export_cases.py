"""错误案例导出（W4）：把错题连问题/期望/预测/题型聚成一张可复核的表。

用法：python scripts/export_cases.py --run <实验目录名> [--out cases.csv]
错误定义：Score.py 组件任一失败（value/unit/ref/NA）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.retrieval_metrics import parse_gold_ref_ids

TYPE_COLS = ["Quote", "Table", "Figure", "Math", "is_NA", "CrossPaper", "Reconcile"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = REPO / "experiments" / args.run
    pred = pd.read_csv(d / "predictions.csv", encoding="utf-8-sig").set_index("id")
    sol = pd.read_csv(d / "solution.csv", encoding="utf-8-sig").set_index("id")

    rows = []
    for qid in sol.index:
        if qid not in pred.index:
            continue
        p, s = pred.loc[qid], sol.loc[qid]
        gold_refs = parse_gold_ref_ids(s["ref_id"])
        try:
            pred_refs = parse_gold_ref_ids(p["ref_id"])
        except Exception:
            pred_refs = []
        na_gold = str(s["answer_value"]).strip() == "is_blank"
        value_ok = (str(p["answer_value"]).strip() == str(s["answer_value"]).strip()) if na_gold \
            else str(p["answer_value"]).strip() == str(s["answer_value"]).strip()
        ref_ok = set(x.lower() for x in gold_refs) == set(x.lower() for x in pred_refs)
        types = [t for t in TYPE_COLS if t in sol.columns and int(s.get(t, 0) or 0) == 1]
        # 宽松数值比对交给 Score.py；此处只做严格串等，作为"候选错题"导出，人工+Score.py 复核
        if (not value_ok) or (not ref_ok):
            rows.append({
                "id": qid, "types": "|".join(types) or "-",
                "question": str(s["question"])[:300],
                "gold_value": s["answer_value"], "pred_value": p["answer_value"],
                "gold_unit": s["answer_unit"], "pred_unit": p["answer_unit"],
                "gold_refs": s["ref_id"], "pred_refs": p["ref_id"],
                "gold_evidence": str(s["supporting_materials"])[:200],
                "pred_evidence": str(p["supporting_materials"])[:200],
                "explanation": str(p["explanation"])[:200],
            })
    out = Path(args.out) if args.out else d / "cases.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"候选错题 {len(rows)}/{len(sol)} -> {out}")


if __name__ == "__main__":
    main()
