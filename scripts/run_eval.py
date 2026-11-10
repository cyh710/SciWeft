"""统一评价入口（W1 闭环）：检索 -> LLM 生成 -> 提交格式 CSV -> 官方 Score.py。

用法：
  python scripts/run_eval.py --config configs/baseline.yaml --split dev --limit 10   # 试跑
  python scripts/run_eval.py --config configs/baseline.yaml --split dev             # dev 全量 + Score.py
  python scripts/run_eval.py --config configs/baseline.yaml --split test            # 冻结后预测 test
产出：experiments/<run_name>/predictions.csv + score 输出摘要。

管线装配在 src/rag/pipeline.py（与 submit.py 共用，保证两入口产出一致）。
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.retrieval_metrics import parse_gold_ref_ids
from src.rag.pipeline import QAPipeline, SUBMIT_COLUMNS, fallback_answer, ref_id_field, ref_url_field
from src.schemas import Chunk


def load_chunks(path: Path) -> tuple[list[Chunk], dict[str, str]]:
    chunks, texts = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = Chunk(**json.loads(line))
            chunks.append(c)
            texts[c.chunk_id] = c.text
    return chunks, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "all", "test"])
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    parser.add_argument("--no-score", action="store_true", help="跳过 Score.py（test 无答案时）")
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    out_dir = REPO / "experiments" / f"{cfg['run_name']}_{args.split}_{datetime.now().strftime('%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks, _ = load_chunks(REPO / cfg["data"]["processed_dir"] / "chunks.jsonl")
    pipe = QAPipeline(cfg, chunks, cache_dir=REPO / "data" / "cache")
    rerank_on = bool(cfg.get("rerank", {}).get("enabled"))
    if rerank_on:
        print(f"[rerank] {cfg['rerank']['model']} top_k_after={cfg['rerank']['top_k_after']}")
    aug_cfg = cfg.get("augment", {})
    if bool(aug_cfg.get("enabled")):
        print(f"[augment] strategies={aug_cfg.get('strategies')}")

    url_map = dict(zip(
        pd.read_csv(REPO / "data" / "raw" / "metadata.csv", encoding="utf-8-sig")["id"],
        pd.read_csv(REPO / "data" / "raw" / "metadata.csv", encoding="utf-8-sig")["url"],
    ))

    qs = pd.read_csv(REPO / "data" / "raw" / ("test_Q.csv" if args.split == "test" else "train_QA.csv"),
                     encoding="utf-8-sig")
    if args.split in ("train", "dev"):
        split = pd.read_csv(REPO / "data" / "processed" / "split.csv")
        qs = qs[qs["id"].isin(split[split["split"] == args.split]["id"])]
    # all = train+dev 全量 245 题（W4 消融用，统计功效优先；配置均已先在 dev 冻结）
    if args.limit:
        qs = qs.head(args.limit)

    rows = []
    n_fail = 0
    for q in qs.itertuples(index=False):
        unit_hint = str(getattr(q, "answer_unit", ""))
        try:
            ans = pipe.run(q.question, unit_hint=unit_hint)
        except Exception as exc:  # noqa: BLE001  单题失败不中断全跑，记为拒答
            n_fail += 1
            ans = fallback_answer(unit_hint, reason=f"generation error: {exc!r}")
        rows.append({
            "id": q.id, "question": q.question,
            "answer": ans.answer_text, "answer_value": ans.value, "answer_unit": ans.unit,
            "ref_id": ref_id_field(ans.citation_ref_ids),
            "ref_url": ref_url_field(ans.citation_ref_ids, url_map),
            "supporting_materials": ans.evidence, "explanation": ans.explanation,
        })
        print(f"[{len(rows)}/{len(qs)}] {q.id}: value={ans.value!r} refs={ans.citation_ref_ids}")
    if n_fail:
        print(f"[warn] {n_fail} 题生成失败，已按 is_blank 兜底（见 explanation）")

    pred_path = out_dir / "predictions.csv"
    pd.DataFrame(rows, columns=SUBMIT_COLUMNS).to_csv(pred_path, index=False)
    print(f"-> {pred_path}")

    # 快速检索侧自检（dev/train 时）：引用的文档是否命中 gold
    if args.split in ("train", "dev"):
        gold_map = {r.id: parse_gold_ref_ids(r.ref_id) for r in qs.itertuples(index=False)}
        hit_any = sum(1 for r in rows
                      if set(parse_gold_ref_ids(r["ref_id"])) & set(gold_map[r["id"]]))
        print(f"[retrieval check] 引用含至少一个 gold 文档的题: {hit_any}/{len(rows)}")

    if not args.no_score and args.split in ("train", "dev", "all"):
        sol_path = out_dir / "solution.csv"
        qs.to_csv(sol_path, index=False)
        result = subprocess.run(
            [sys.executable, str(REPO / "Score.py"), str(sol_path), str(pred_path)],
            capture_output=True, text=True, cwd=REPO,
        )
        if result.returncode != 0:
            print(f"[warn] Score.py 退出码 {result.returncode}（根目录缺 Score.py？）\n{result.stderr[-500:]}")
        print(result.stdout)
        (out_dir / "score_output.txt").write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
