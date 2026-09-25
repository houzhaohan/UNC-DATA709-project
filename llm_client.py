"""
LLM client: Azure OpenAI Responses API with retry, backoff, and checkpoint.
Follows the test.py client.responses.create() pattern.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

import config


class LLMClient:
    def __init__(self):
        self.client = OpenAI(base_url=config.ENDPOINT, api_key=config.API_KEY)
        self.model = config.DEPLOYMENT

    # ── Core call ────────────────────────────────────────────────────────────
    def call(self, prompt: str, max_retries: int = 5) -> str:
        """
        Call the LLM and return raw text, with exponential-backoff retries.

        responses.create return structure:
          resp.output -> [ResponseOutputMessage, ...]
            message.content -> [ResponseOutputText, ...]
              .text -> the actual answer text
        """
        for attempt in range(max_retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                )
                text = self._extract_text(resp)
                if text.strip():
                    return text
                raise RuntimeError("LLM returned empty text")
            except Exception as e:
                wait = 2 ** attempt + 1
                print(f"[llm] Attempt {attempt+1} failed: {e}; retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed {max_retries} consecutive times")

    @staticmethod
    def _extract_text(resp) -> str:
        """Extract full text from responses.create return object."""
        parts = []
        for msg in resp.output or []:
            # msg.content is a list of ResponseOutputText
            for chunk in getattr(msg, "content", []) or []:
                t = getattr(chunk, "text", None)
                if t:
                    parts.append(t)
            # compat with dict form
            if isinstance(msg, dict):
                for chunk in msg.get("content", []) or []:
                    if isinstance(chunk, dict) and chunk.get("text"):
                        parts.append(chunk["text"])
        return "".join(parts)


# ── JSON parsing ─────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    # 1. Try parsing the whole thing directly
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. Grab the first { ... } span
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 3. Grab the first [ ... ] wrap
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 4. Last resort: return an empty shell with the original text attached
    return {
        "decision": None,
        "confidence": None,
        "reason": text[:200],
        "counterfactual": [],
        "_parse_error": True,
    }


# ── Checkpoint read/write ────────────────────────────────────────────────────
def ckpt_load(path: Path) -> dict[int, dict]:
    """Load a JSONL checkpoint and return {idx: result}."""
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["idx"]] = rec
            except Exception:
                pass
    return out


def ckpt_save(path: Path, idx: int, group: str, result: dict, raw_prompt: str = "") -> None:
    """Append one checkpoint record."""
    rec = {"idx": idx, "group": group, "result": result, "prompt": raw_prompt[:500]}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
