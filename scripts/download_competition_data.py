"""下载 WattBot 2026 官方数据到 data/raw/。

前置条件（缺一不可，脚本会逐项检查并给出指引）：
1. 已在浏览器登录 Kaggle 并接受比赛规则：
   https://www.kaggle.com/competitions/WattBot2026/rules
2. ~/.kaggle/kaggle.json 存在（Kaggle -> Settings -> API -> Create New Token）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
COMPETITION = "WattBot2026"
EXPECTED = ["metadata.csv", "train_QA.csv", "test_Q.csv", "Score.py"]
KEEP_AT_ROOT = ["Score.py"]            # CLI 约定 python Score.py ... 在仓库根执行
KEEP_IN_DOCS = ["CONTRIBUTE_QUESTIONS.md", "WRITEUP_TEMPLATE.md"]


def die(msg: str) -> None:
    print(f"[x] {msg}")
    sys.exit(1)


def main() -> None:
    missing = [f for f in EXPECTED if not (RAW / f).exists()]
    if not missing:
        print(f"[ok] {RAW} 已包含全部官方文件，无需下载：{EXPECTED}")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not (kaggle_json.exists() or env_token):
        die(
            "缺少 Kaggle 凭证。两种方式任选：\n"
            "  A. 仓库根 .env 写一行 KAGGLE_API_TOKEN=KGAT_...（Settings -> API 生成）；\n"
            "  B. kaggle.json（用户名+旧式 key）放到 ~/.kaggle/。\n"
            "两种方式都必须先在浏览器接受比赛规则：\n"
            "  https://www.kaggle.com/competitions/WattBot2026/rules"
        )

    try:
        import kaggle  # noqa: F401  装了 kaggle CLI 才会成功
    except ImportError:
        die("缺少 kaggle 包：pip install kaggle 后重跑。")

    print(f"[..] 下载 {COMPETITION} 到 {RAW} ...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION,
         "-p", str(RAW), "--unzip", "--force"],
    )
    if result.returncode != 0:
        die("kaggle 下载失败。若提示 403，多半是还没在网页上接受比赛规则。")

    still_missing = [f for f in EXPECTED if not (RAW / f).exists()]
    if still_missing:
        die(f"下载完成但缺少文件：{still_missing}")

    for name in KEEP_AT_ROOT:
        if (RAW / name).exists():
            shutil.copy2(RAW / name, REPO / name)
            print(f"[ok] {name} -> 仓库根")
    for name in KEEP_IN_DOCS:
        if (RAW / name).exists():
            shutil.copy2(RAW / name, REPO / "docs" / name)
            print(f"[ok] {name} -> docs/")

    print("[done] 官方数据就绪。下一步：python scripts/download_corpus.py")


if __name__ == "__main__":
    main()
