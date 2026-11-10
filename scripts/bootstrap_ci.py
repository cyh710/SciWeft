"""Bootstrap 置信区间（W4）：对 (solution, predictions) 有放回重采样题目 N 次，
每次用原版 Score.py 重打分——不复制评分逻辑，杜绝口径漂移。

实现要点：按「位置」重采样后对齐预测行，重复 id 用后缀去重（此前用 isin 过滤
会把重复 id 折叠成一行，重采样退化为 ~63% 覆盖的随机子集，区间含义错误）。

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

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def resample_ids(ids: list, rng: np.random.Generator) -> list:
    """有放回抽 len(ids) 个位置对应的 id：bootstrap 必然带重复，重复不能被折叠。"""
    idx = rng.integers(0, len(ids), len(ids))
    return [ids[i] for i in idx]


def aligned_frames(sol: pd.DataFrame, pred: pd.DataFrame, sample_ids: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 solution/pred 按重采样后的 id（含重复）对齐；重复 id 按槽位加后缀防折叠。"""
    s = sol.set_index("id").loc[sample_ids].copy()   # 索引唯一时，loc 重复标签 = 重复取该行
    p = pred.set_index("id").loc[sample_ids].copy()
    new_ids = pd.Index([f"{qid}__{j}" for j, qid in enumerate(sample_ids)], name="id")
    s.index, p.index = new_ids, new_ids
    return s.reset_index(), p.reset_index()


def score_once(sol_csv: Path, pred_csv: Path) -> float:
    r = subprocess.run([sys.executable, str(REPO / "Score.py"), str(sol_csv), str(pred_csv)],
                       capture_output=True, text=True, cwd=REPO)
    # 兼容官方 "OVERALL SCORE : x" 与本地替身的 "OVERALL: x"
    m = re.search(r"OVERALL(?: SCORE)?\s*:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"Score.py 输出异常:\n{r.stdout[-500:]}\n{r.stderr[-300:]}")
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sol = pd.read_csv(args.solution, encoding="utf-8-sig")
    pred = pd.read_csv(args.pred, encoding="utf-8-sig")
    # Score.py 按 id 对齐，两个文件重采样同一批 id 即可
    ids = sol["id"].tolist()
    rng = np.random.default_rng(args.seed)
    scores = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i in range(args.n):
            s, p = aligned_frames(sol, pred, resample_ids(ids, rng))
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
