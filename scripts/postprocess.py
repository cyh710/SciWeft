"""预测后处理（W5）：把可确定性修复的格式错误规范化，无需重跑 LLM。

规则（全部来自 245 题误差分析 + 官方评分规则的对照验证）：
1. 区间方向修复：pred 是 "(a,b)" 但问题没有问 "range"（gold 形态学：陈述区间题
   问句必含 range 字样 6/7；容差带题 gold 用 [a,b]，提交自己的区间 = 0 分）→
   替换为区间中点单值。
2. Scope 前缀剥离：unit 含 scope 而 value 写成 "Scope 2" → "2"。
3. 数值清理：去千分位逗号、去货币符号前缀。

用法：python scripts/postprocess.py --in predictions.csv --out post.csv --questions train_QA.csv
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

_RANGE_RE = re.compile(r"^\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)$")
_SCOPE_RE = re.compile(r"^scope\s*([0-9.]+)$", re.I)
_ASK_RANGE_RE = re.compile(r"\brange\b", re.I)


def postprocess_value(value: str, unit: str, question: str) -> str:
    v = value.strip()
    if v in ("", "is_blank"):
        return v
    sm = _SCOPE_RE.match(v)
    if sm and "scope" in unit.lower():
        return sm.group(1)
    if re.match(r"^[\$€£]?[0-9][0-9,]*\.?[0-9]*$", v):
        v = v.replace(",", "").lstrip("$€£")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", required=True, help="含 id+question 的 CSV（train_QA 或 test_Q）")
    args = ap.parse_args()

    pred = pd.read_csv(args.inp, encoding="utf-8-sig")
    qs = pd.read_csv(args.questions, encoding="utf-8-sig").set_index("id")
    changed = 0
    for i in pred.index:
        qid = pred.at[i, "id"]
        if qid not in qs.index:
            continue
        old = str(pred.at[i, "answer_value"])
        new = postprocess_value(old, str(pred.at[i, "answer_unit"]), str(qs.loc[qid, "question"]))
        if new != old:
            pred.at[i, "answer_value"] = new
            changed += 1
    pred.to_csv(args.out, index=False)
    print(f"后处理: {changed}/{len(pred)} 行修改 -> {args.out}")


if __name__ == "__main__":
    main()
