"""Day-1 冒烟：契约可导入、配置可解析、CLI 桩可发现。"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_schemas_importable():
    from src.schemas import Answer, Chunk, Document, Hit  # noqa: F401


def test_baseline_config_parses():
    cfg = yaml.safe_load((REPO / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    assert cfg["retrieval"]["method"] in {"tfidf", "bm25", "dense", "hybrid"}
    assert cfg["generation"]["temperature"] == 0
    assert cfg["eval"]["split"] in {"train", "dev", "test"}


def test_cli_stubs_exist():
    for name in ("build_index.py", "run_eval.py", "submit.py"):
        assert (REPO / "scripts" / name).exists()
