"""Shared JSON parsing utilities for LLM response handling.

LLM responses frequently contain markdown fences and literal newlines inside
JSON string values. These helpers normalise the raw text before parsing.
"""

import json
import re
from typing import Optional


def strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from an LLM response."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    return re.sub(r"\n?```\s*$", "", text.strip()).strip()


def fix_json_newlines(json_str: str) -> str:
    """Escape literal newlines inside JSON string values.

    Standard JSON forbids bare newline characters inside string values.
    LLMs often emit them anyway. This walks the string character-by-character
    so it only touches newlines that are actually inside a quoted value.
    """
    result: list[str] = []
    in_string = False
    escape_next = False
    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == "\\" and in_string:
            result.append(char)
            escape_next = True
        elif char == '"':
            in_string = not in_string
            result.append(char)
        elif char == "\n" and in_string:
            result.append("\\n")
        else:
            result.append(char)
    return "".join(result)


def parse_llm_json(response: str) -> Optional[dict]:
    """Strip fences, fix literal newlines, then parse JSON.

    Returns the parsed dict on success, or None if parsing fails.
    Callers are responsible for logging any failure context.
    """
    text = strip_markdown_fences(response)
    text = fix_json_newlines(text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


__all__ = ["strip_markdown_fences", "fix_json_newlines", "parse_llm_json"]
