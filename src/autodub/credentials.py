"""Read a local DeepSeek key without executing configuration or serializing the key."""
import os
import re
from pathlib import Path


def deepseek_key(path: Path) -> str:
    existing = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if existing:
        return existing
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8-sig").strip()
    keys = set(re.findall(r"\bsk-[A-Za-z0-9_-]{8,}\b", content))
    if len(keys) != 1:
        raise ValueError("API key file must contain exactly one DeepSeek key")
    return keys.pop()
