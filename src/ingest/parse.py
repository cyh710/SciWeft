"""parse_documents：PDF -> 逐页文本（W1）。

用 PyMuPDF 提取可复制文本，保留页码边界；解析失败的文档记 parse_ok=False 并
写入失败清单，绝不中断全库。输出 data/processed/pages/{ref_id}.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pymupdf

from src.schemas import Document

REPO = Path(__file__).resolve().parents[2]


def parse_documents(metadata_csv: Path, corpus_dir: Path, pages_dir: Path) -> list[Document]:
    meta = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    pages_dir.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = []
    failures: list[dict] = []

    for row in meta.itertuples(index=False):
        pdf = corpus_dir / f"{row.id}.pdf"
        doc = Document(ref_id=row.id, url=row.url, title=str(row.title), year=str(row.year))
        if not pdf.exists():
            doc.parse_ok, doc.parse_error = False, "pdf_missing"
            failures.append({"ref_id": row.id, "error": "pdf_missing"})
            docs.append(doc)
            continue
        try:
            with pymupdf.open(pdf) as pdf_doc:
                pages = [page.get_text("text") for page in pdf_doc]
            doc.page_count = len(pages)
            (pages_dir / f"{row.id}.json").write_text(
                json.dumps({"ref_id": row.id, "pages": pages}, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            doc.parse_ok, doc.parse_error = False, repr(exc)[:200]
            failures.append({"ref_id": row.id, "error": doc.parse_error})
        docs.append(doc)

    if failures:
        (pages_dir / "_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(d.parse_ok for d in docs)
    print(f"parse: ok={ok}/{len(docs)}  failures={len(failures)}")
    return docs
