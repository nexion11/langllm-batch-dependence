"""Main-study settings (docs/main_study_plan.md). Generation settings, language table and audience text
follow LangLLM's config.yaml (commit 06109dc) so new essays are comparable with the original ones."""
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "new"
PROMPTS = ROOT / "prompts" / "new"
BUDGET_USD = 15.0

CANDIDATES = {  # N1: the five LangLLM subjects
    "gpt": "openai/gpt-5.5", "gemini": "google/gemini-3.5-flash", "claude": "anthropic/claude-opus-4.7",
    "grok": "x-ai/grok-4.3", "deepseek": "deepseek/deepseek-v4-pro-0813",
}
OUTSIDE = {  # N2: outside reference models, none from a candidate's vendor; run locally in Ollama (addendum 2)
    "llama": "llama3.1:8b", "mistral": "mistral-small3.2:24b", "qwen": "qwen3:14b",
}
OLLAMA = "http://localhost:11434"
THINKING_MODELS = {"qwen3:14b"}  # get think=false
PROMPT_WRITER = "meta-llama/llama-4-maverick"
PROMPT_REVIEWER = "qwen/qwen3.7-plus"
GENERATION = {"temperature": 0.7, "max_tokens": 4000, "reasoning": {"effort": "low", "exclude": True}, "seed": 1000}

LANGUAGES = {
    "en": {"name_en": "English", "native": "English", "length_hint": "about 350 words"},
    "es": {"name_en": "Spanish", "native": "español", "length_hint": "unas 350 palabras"},
    "zh": {"name_en": "Chinese", "native": "中文", "length_hint": "大约600字"},
    "ru": {"name_en": "Russian", "native": "русский", "length_hint": "около 350 слов"},
    "ja": {"name_en": "Japanese", "native": "日本語", "length_hint": "800字程度"},
    "tr": {"name_en": "Turkish", "native": "Türkçe", "length_hint": "yaklaşık 350 kelime"},
    "hi": {"name_en": "Hindi", "native": "हिन्दी", "length_hint": "लगभग 350 शब्द"},
}
AUDIENCE = {
    "general": "a general adult reader (roughly 8th-grade reading level: short sentences, everyday vocabulary)",
    "educated": "an educated adult reader (roughly 12th-grade reading level: varied sentences, some technical terms explained)",
    "expert": "an expert reader (graduate level: dense argument, technical vocabulary used without explanation)",
}


def schemas() -> list[dict]:
    return json.loads((ROOT / "prompts" / "schemas_new.json").read_text(encoding="utf-8"))["schemas"]


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set: copy .env.example to .env and add the key")
    return key


def spent_usd() -> float:
    """Sum of logged API cost across every call recorded in data/new and prompts/new."""
    total = 0.0
    for p in list(NEW.glob("*.jsonl")) + list(PROMPTS.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float((json.loads(line).get("usage") or {}).get("cost") or 0)
    return total


def chat(model: str, messages: list[dict], *, temperature: float, max_tokens: int, seed: int | None = None,
         reasoning: dict | None = None, retries: int = 5) -> dict:
    if spent_usd() > BUDGET_USD:
        raise SystemExit(f"budget guard: logged spend is above ${BUDGET_USD}, stopping")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "usage": {"include": True}}
    if seed is not None:
        body["seed"] = seed
    if reasoning:
        body["reasoning"] = reasoning
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                                          "X-Title": "llm-fingerprint-paper"})
    err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            if "error" in data or not data.get("choices"):
                raise RuntimeError(str(data.get("error") or data)[:300])
            return data
        except urllib.error.HTTPError as e:
            err = f"{e.code}: {e.read()[:300]!r}"
            if e.code in (400, 401, 402, 403, 404):
                raise RuntimeError(err)
        except Exception as e:  # timeouts, connection resets, error envelopes
            err = str(e)[:300]
        time.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError(f"gave up after {retries} attempts: {err}")


def ollama_digests() -> dict:
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
        return {m["name"]: m["digest"] for m in json.loads(r.read())["models"]}


def ollama_chat(model: str, messages: list[dict], *, temperature: float, max_tokens: int, seed: int | None = None,
                retries: int = 3) -> dict:
    """Local Ollama call, returned in the same shape as an OpenRouter response so text_of and usage_of work."""
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192, "seed": seed}}
    if model in THINKING_MODELS:
        body["think"] = False
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            return {"model": d.get("model"), "provider": "ollama-local",
                    "choices": [{"message": {"content": d["message"]["content"]},
                                 "finish_reason": "length" if d.get("done_reason") == "length" else d.get("done_reason")}],
                    "usage": {"prompt_tokens": d.get("prompt_eval_count"), "completion_tokens": d.get("eval_count"), "cost": 0}}
        except Exception as e:
            err = str(e)[:300]
            time.sleep(5)
    raise RuntimeError(f"ollama gave up after {retries} attempts: {err}")


def text_of(resp: dict) -> str:
    c = resp["choices"][0]["message"].get("content") or ""
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c.strip()


def usage_of(resp: dict) -> dict:
    u = resp.get("usage") or {}
    out = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")}
    out["reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return out
