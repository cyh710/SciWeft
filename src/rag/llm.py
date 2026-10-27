"""LLM 客户端：OpenAI / Anthropic 双协议 + 磁盘缓存 + 限流退避 + 重试（W1）。

凭证从仓库根 .env 读取（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_KIND），绝不进 Git。
LLM_KIND=openai      -> POST {base}/chat/completions
LLM_KIND=anthropic   -> POST {base}/v1/messages

同题同条件命中磁盘缓存；429 限流按 Retry-After 退避；
思考型模型若把 max_tokens 吃光导致无正文，自动翻倍预算重试。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "deepseek-flash")
KIND = os.environ.get("LLM_KIND", "openai").lower()
CACHE_DIR = REPO / "cache" / "llm"
MAX_RETRIES = 5
MIN_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "2"))  # 请求间隔秒数，防 429
_last_call = 0.0


class _RateLimited(Exception):
    """429：服务端限流，按 Retry-After 或指数退避等待后重试。"""

    def __init__(self, retry_after: float = 0):
        super().__init__("rate limited (429)")
        self.retry_after = retry_after


def _post(payload: dict) -> dict:
    if KIND == "anthropic":
        r = requests.post(
            f"{BASE_URL}/v1/messages",
            headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json=payload, timeout=240,
        )
    else:
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=240,
        )
    if r.status_code == 429:
        raise _RateLimited(float(r.headers.get("Retry-After", 0) or 0))
    r.raise_for_status()
    return r.json()


def _extract_text(data: dict) -> str:
    if KIND == "anthropic":
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def chat(messages: list[dict], system: str = "", max_tokens: int = 6000, temperature: float = 0.0) -> str:
    """调用 LLM 并返回正文文本。带磁盘缓存、限流退避与预算自适应。"""
    key = hashlib.sha256(json.dumps(
        {"m": MODEL, "k": KIND, "s": system, "msg": messages, "t": temperature, "k2": max_tokens},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cache = CACHE_DIR / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["text"]

    if KIND == "anthropic":
        payload = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}
        if system:
            payload["system"] = system
    else:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        payload = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature, "messages": msgs}

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        global _last_call
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            data = _post(payload)
            _last_call = time.time()
            text = _extract_text(data)
            if not text.strip():
                payload["max_tokens"] = min(payload["max_tokens"] * 2, 16000)
                raise ValueError("empty text (thinking ate the budget?)")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"text": text, "usage": data.get("usage", {}),
                                         "model": MODEL}, ensure_ascii=False), encoding="utf-8")
            return text
        except _RateLimited as exc:
            last_err = exc
            time.sleep(max(exc.retry_after, 20 * (attempt + 1)))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt * 3)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES + 1} attempts: {last_err}")
