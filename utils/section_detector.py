from typing import List, Dict, Any, Optional
import re
import time
from loguru import logger


def _map_title_to_canonical(title: str) -> str:
    t = title.lower()
    if t.startswith("intro") or t.startswith("background") or t.startswith("overview"):
        return "Introduction"
    if (
        t.startswith("method")
        or t.startswith("materials")
        or t.startswith("methodology")
        or t.startswith("experimental")
    ):
        return "Methods"
    if t.startswith("result") or t.startswith("finding"):
        return "Results"
    if t.startswith("discussion") or t.startswith("analysis"):
        return "Discussion"
    if t.startswith("conclusion") or t.startswith("summary") or t.startswith("future"):
        return "Conclusion"
    return title.strip().title()


def _token_positions(text: str) -> List[str]:
    # Return list of tokens (whitespace-separated)
    return re.findall(r"\S+", text)


# Precompile header regex for performance
_TITLES = r"(Introduction|Background|Overview|Methods?|Materials?|Methodology|Experimental|Results?|Findings?|Discussion|Analysis|Conclusion|Summary|Future Work)"
# pattern covers: markdown headers (# Head), numbered headers (1. Head, I. Head, (1) Head), and lines that equal the title
_HEADER_PATTERN = rf"^(?:\s*#{{1,6}}\s*(?P<title>{_TITLES})\b.*$)|^(?:\s*(?:\d{{1,2}}|[IVXLCDM]+|\(\d{{1,2}}\))[\.\)]?\s*(?P<title_num>{_TITLES})\b.*$)|^(?P<title_only>{_TITLES})\s*$"
_HEADER_RE = re.compile(_HEADER_PATTERN, re.MULTILINE | re.IGNORECASE)


def detect_sections(
    text: str,
    paper_id: Optional[str] = None,
    fallback_token_size: int = 1000,
    min_section_length: int = 100,
) -> List[Dict[str, Any]]:
    """Detect standard sections in scientific paper text using regex.

    Args:
        text: Full PDF text
        paper_id: Optional paper ID for logging
        fallback_token_size: Token count for naive splits if under 2 sections detected
        min_section_length: minimum chars for a detected section to be kept

    Returns:
        List of section dicts with name, text, start_idx, end_idx
    """
    start_time = time.perf_counter()
    try:
        logger.debug(
            {
                "event": "section_detection_start",
                "paper_id": paper_id,
                "length": len(text),
            }
        )
        if not text:
            logger.debug(
                {
                    "event": "section_detection",
                    "paper_id": paper_id,
                    "sections": 0,
                    "fallback": True,
                }
            )
            return [{"name": "full_text", "text": "", "start_idx": 0, "end_idx": 0}]

        # Use module-level compiled regex to find headers
        matches: List[tuple] = []
        for m in _HEADER_RE.finditer(text):
            title = m.group("title") or m.group("title_num") or m.group("title_only")
            if title:
                matches.append((title, m.start()))

        sections: List[Dict[str, Any]] = []

        if len(matches) < 2:
            # Fallback to approximate char-based chunks for speed
            logger.warning(
                {
                    "event": "section_detection_fallback_triggered",
                    "paper_id": paper_id,
                    "matches_found": len(matches),
                    "reason": "matches<2",
                }
            )
            chars_per_chunk = fallback_token_size * 4

            chunks: List[Dict[str, Any]] = []
            start_idx = 0
            text_len = len(text)
            while start_idx < text_len:
                end_idx = min(text_len, start_idx + chars_per_chunk)
                # try to extend to nearest sentence boundary (within next 1000 chars)
                m = re.search(r"[\.\?!][\"']?\s+", text[end_idx : end_idx + 1000])
                if m:
                    end_idx = end_idx + m.end()
                chunk_text = text[start_idx:end_idx].strip()
                if chunk_text:
                    if len(chunk_text) >= min_section_length:
                        chunks.append(
                            {
                                "name": f"chunk_{len(chunks)}",
                                "text": chunk_text,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                            }
                        )
                    else:
                        if chunks:
                            prev = chunks[-1]
                            prev_text = text[prev["start_idx"] : end_idx].strip()
                            prev.update({"text": prev_text, "end_idx": end_idx})
                        else:
                            chunks.append(
                                {
                                    "name": "chunk_0",
                                    "text": chunk_text,
                                    "start_idx": start_idx,
                                    "end_idx": end_idx,
                                }
                            )
                start_idx = end_idx

            elapsed = time.perf_counter() - start_time
            logger.debug(
                {
                    "event": "section_detection",
                    "paper_id": paper_id,
                    "sections": len(chunks),
                    "fallback": True,
                    "elapsed_s": elapsed,
                    "matches_found": len(matches),
                }
            )
            return (
                chunks
                if chunks
                else [
                    {
                        "name": "full_text",
                        "text": text,
                        "start_idx": 0,
                        "end_idx": len(text),
                    }
                ]
            )

        # Build sections from matches (matches is list of (title, start))
        anchors: List[Dict[str, Any]] = []
        for title, start in matches:
            canonical = _map_title_to_canonical(title)
            anchors.append({"name": canonical, "start": start})

        # Sort anchors by start (should already be ordered)
        anchors = sorted(anchors, key=lambda a: a["start"])

        for idx, a in enumerate(anchors):
            start_idx = a["start"]
            end_idx = anchors[idx + 1]["start"] if idx + 1 < len(anchors) else len(text)
            sect_text = text[start_idx:end_idx].strip()
            sections.append(
                {
                    "name": a["name"],
                    "text": sect_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

        # Filter short sections if requested
        filtered = [s for s in sections if len(s.get("text", "")) >= min_section_length]
        if len(filtered) < 2:
            # If filtering removed too many, keep original sections (prefer detection over nothing)
            filtered = sections

        elapsed = time.perf_counter() - start_time
        logger.debug(
            {
                "event": "section_detection",
                "paper_id": paper_id,
                "sections": len(filtered),
                "fallback": False,
                "names": [s["name"] for s in filtered],
                "elapsed_s": elapsed,
            }
        )
        return filtered

    except Exception as e:  # Never raise
        logger.debug(
            {"event": "section_detection_error", "paper_id": paper_id, "error": str(e)}
        )
        return [
            {
                "name": "full_text",
                "text": text or "",
                "start_idx": 0,
                "end_idx": len(text or ""),
            }
        ]
