"""train/dev 分层划分（W1）：按题型分层 8:2，冻结种子，写 data/processed/split.csv。

分层键取「主题型」：按稀有度优先（is_NA > Math > Figure > Reconcile > CrossPaper
> Table > Quote），保证每个题型在 dev 中都有代表。划分一经生成就冻结——后续所有
实验的 dev 指标都基于同一份 split.csv，改划分 = 作废之前所有结果。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PRIORITY = ["is_NA", "Math", "Figure", "Reconcile", "CrossPaper", "Table", "Quote"]
DEV_FRACTION = 0.2
SEED = 42


def primary_type(row: pd.Series) -> str:
    for flag in PRIORITY:
        if bool(row.get(flag, 0)):
            return flag
    return "Quote"


def make_split(train_csv: Path, out_csv: Path, dev_fraction: float = DEV_FRACTION, seed: int = SEED) -> pd.DataFrame:
    train = pd.read_csv(train_csv, encoding="utf-8-sig")
    train["primary_type"] = train.apply(primary_type, axis=1)
    dev_ids = []
    for type_name, group in train.groupby("primary_type"):
        n_dev = max(1, round(len(group) * dev_fraction))
        dev_ids.extend(group.sample(n=min(n_dev, len(group)), random_state=seed)["id"].tolist())
    train["split"] = train["id"].isin(dev_ids).map({True: "dev", False: "train"})
    cols = ["id", "primary_type", "split"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    train[cols].to_csv(out_csv, index=False)
    summary = train.groupby(["primary_type", "split"]).size().unstack(fill_value=0)
    print(summary)
    print(f"total: train={int((train['split']=='train').sum())} dev={int((train['split']=='dev').sum())} -> {out_csv}")
    return train[cols]


if __name__ == "__main__":
    make_split(REPO / "data" / "raw" / "train_QA.csv", REPO / "data" / "processed" / "split.csv")
