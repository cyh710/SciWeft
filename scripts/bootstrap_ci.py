"""Bootstrap 置信区间（W4）：对 (solution, predictions) 重采样题目 N 次，
每次用原版 Score.py 重打分——不复制评分逻辑，杜绝口径漂移。

用法：python scripts/bootstrap_ci.py --solution <csv> --pred <csv> [--n 200]
输出：OVERALL 均值与 95% 区间（按百分位）。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PYTHON = str(REPO / ".venv" / "Scripts" / "python.exe")


def score_once(sol_csv: Path, pred_csv: Path) -> float:
    r = subprocess.run([PYTHON, str(REPO / "Score.py"), str(sol_csv), str(pred_csv)],
                       capture_output=True, text=True, cwd=REPO)
    m = re.search(r"OVERALL SCORE\s*:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"Score.py 输出异常:\n{r.stdout[-500:]}\n{r.stderr[-300:]}")
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sol = pd.read_csv(args.solution, encoding="utf-8-sig")
    pred = pd.read_csv(args.pred, encoding="utf-8-sig")
    # Score.py 按 id 对齐，两个文件重采样同一批 id 即可
    ids = sol["id"].tolist()
    rng = pd.Series(range(len(ids))).sample(frac=1.0, random_state=args.seed)  # noqa: F841

    import random
    random.seed(args.seed)
    scores = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i in range(args.n):
            sample_ids = random.choices(ids, k=len(ids))
            s = sol[sol["id"].isin(sample_ids)]
            p = pred[pred["id"].isin(sample_ids)]
            sp, pp = tdp / f"s{i}.csv", tdp / f"p{i}.csv"
            s.to_csv(sp, index=False)
            p.to_csv(pp, index=False)
            scores.append(score_once(sp, pp))
    scores.sort()
    mean = sum(scores) / len(scores)
    lo, hi = scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores)) - 1]
    print(f"n_resample={args.n}  OVERALL mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  (point est 见 Score.py 单次输出)")


if __name__ == "__main__":
    main()
