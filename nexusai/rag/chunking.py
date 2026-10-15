"""Structure-aware chunking for Markdown and plain text.

Strategy:
  1. Split the document into sections at Markdown headings, and remember the
     heading path ("Setup > Docker") for each section.
  2. Pack whole paragraphs into chunks of up to `chunk_size` characters.
     Paragraphs that are too long are split at sentence boundaries, and as a
     last resort at the character limit.
  3. Start each chunk with the last `overlap` characters of the previous one
     (aligned to a sentence where possible), so facts that straddle a chunk
     boundary are still retrievable.

Chunks never cross a section boundary, so each one stays on a single topic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class TextChunk:
    ordinal: int
    text: str
    heading: str


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs. Text before the first heading has an empty path."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    stack: list[tuple[int, str]] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING_RE.match(line)
        if match:
            level, title = len(match.group(1)), match.group(2).strip()
            stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
            sections.append((" > ".join(t for _, t in stack), []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _split_long(paragraph: str, limit: int) -> list[str]:
    if len(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(paragraph):
        while len(sentence) > limit:  # a single giant "sentence": hard split
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last `overlap` characters of text, starting at a sentence boundary when one exists."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    boundary = _SENTENCE_RE.search(tail)
    if boundary:
        return tail[boundary.end():]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks: list[TextChunk] = []
    for heading, body in split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        pieces = [piece for p in paragraphs for piece in _split_long(p, chunk_size - overlap)]
        current = ""
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > chunk_size:
                chunks.append(TextChunk(len(chunks), current, heading))
                carry = _tail(current, overlap)
                current = f"{carry}\n\n{piece}" if carry else piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
        if current:
            chunks.append(TextChunk(len(chunks), current, heading))
    return chunks
