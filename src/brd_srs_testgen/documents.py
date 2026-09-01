from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import TypeVar

from pydantic import BaseModel
from pypdf import PdfReader

from .models import DocumentChunk, SourceReference


class DocumentError(ValueError):
    pass


T = TypeVar("T", bound=BaseModel)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def _evidence_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", normalize_text(text).casefold()))


def _grounded_excerpt(excerpt: str, evidence: str) -> str | None:
    normalized = normalize_text(evidence)
    words = _evidence_key(excerpt).split()
    if len(words) < 5:
        return None
    evidence_tokens = list(re.finditer(r"\w+", normalized))
    evidence_words = [match.group().casefold() for match in evidence_tokens]
    if "..." in excerpt or "…" in excerpt:
        segments = [
            _evidence_key(segment).split()
            for segment in re.split(r"(?:\.\.\.|…)", excerpt)
            if _evidence_key(segment)
        ]
        positions: list[tuple[int, int]] = []
        cursor = 0
        for segment in segments:
            position = next(
                (
                    index
                    for index in range(cursor, len(evidence_words) - len(segment) + 1)
                    if evidence_words[index : index + len(segment)] == segment
                ),
                None,
            )
            if position is None:
                positions = []
                break
            positions.append((position, position + len(segment) - 1))
            cursor = position + len(segment)
        if positions:
            return normalized[
                evidence_tokens[positions[0][0]].start() : evidence_tokens[
                    positions[-1][1]
                ].end()
            ]
    candidates: list[tuple[float, int, int]] = []
    window_size = max(len(words) * 2, len(words) + 10)
    for start, word in enumerate(evidence_words):
        if word != words[0]:
            continue
        window = evidence_words[start : start + window_size]
        matches = [
            match
            for match in SequenceMatcher(
                None, words, window, autojunk=False
            ).get_matching_blocks()
            if match.size
        ]
        coverage = sum(match.size for match in matches) / len(words)
        if coverage >= 0.9:
            final = matches[-1]
            candidates.append(
                (coverage, start + matches[0].b, start + final.b + final.size - 1)
            )
    if not candidates:
        match = SequenceMatcher(
            None, words, evidence_words, autojunk=False
        ).find_longest_match()
        if match.size < 8 or match.size * 5 < len(words) * 2:
            return None
        return normalized[
            evidence_tokens[match.b].start() : evidence_tokens[
                match.b + match.size - 1
            ].end()
        ]
    _coverage, first, last = max(
        candidates, key=lambda item: (item[0], -(item[2] - item[1]))
    )
    return normalized[evidence_tokens[first].start() : evidence_tokens[last].end()]


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as error:
        raise DocumentError("The uploaded file is not a readable PDF.") from error
    if reader.is_encrypted:
        raise DocumentError("Encrypted PDFs are not supported.")
    try:
        return [
            (number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, 1)
        ]
    except Exception as error:
        raise DocumentError("PDF text extraction failed.") from error


def _section_heading(raw_text: str) -> str:
    for line in raw_text.splitlines():
        candidate = normalize_text(line)
        if 3 <= len(candidate) <= 100 and (
            candidate.isupper() or re.match(r"^\d+(?:\.\d+)*\s+\S", candidate)
        ):
            return candidate
    return ""


def _pieces(text: str, max_chars: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end == len(text):
            yield text[start:end]
            return
        cut = text.rfind(" ", start, end + 1)
        end = cut if cut > start else end
        yield text[start:end]
        start = end


def chunk_pages(
    pages: Iterable[tuple[int, str]], max_chars: int = 6_000
) -> list[DocumentChunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[DocumentChunk] = []
    for page_number, raw_text in pages:
        text = normalize_text(raw_text)
        if not text:
            continue
        section = _section_heading(raw_text)
        pieces = filter(None, map(normalize_text, _pieces(text, max_chars)))
        for sequence, piece in enumerate(pieces, 1):
            digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"p{page_number:04d}-c{sequence:03d}-{digest[:12]}",
                    page_number=page_number,
                    section=section,
                    text=piece,
                    content_hash=digest,
                )
            )
    if not chunks:
        raise DocumentError("PDF contains insufficient extractable text.")
    return chunks


def parse_pdf(pdf_bytes: bytes, max_chars: int = 6_000) -> list[DocumentChunk]:
    return chunk_pages(extract_pages(pdf_bytes), max_chars=max_chars)


def verify_source_reference(
    reference: SourceReference, chunks: list[DocumentChunk]
) -> bool:
    chunk = next((item for item in chunks if item.chunk_id == reference.chunk_id), None)
    excerpt = _evidence_key(reference.excerpt)
    return bool(
        excerpt
        and chunk
        and chunk.page_number == reference.page_number
        and excerpt in _evidence_key(chunk.text)
    )


def canonicalize_source_references(value: T, chunks: list[DocumentChunk]) -> T:
    data = value.model_dump(mode="json")

    def visit(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            if {"chunk_id", "page_number", "excerpt"} <= node.keys():
                excerpt = _evidence_key(str(node["excerpt"]))
                matches = [
                    chunk
                    for chunk in chunks
                    if excerpt and excerpt in _evidence_key(chunk.text)
                ]
                if matches:
                    chunk = next(
                        (item for item in matches if item.chunk_id == node["chunk_id"]),
                        next(
                            (
                                item
                                for item in matches
                                if item.page_number == node["page_number"]
                            ),
                            matches[0],
                        ),
                    )
                    node.update(
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        section=chunk.section,
                    )
                else:
                    grounded = [
                        (chunk, exact_excerpt)
                        for chunk in chunks
                        if (
                            exact_excerpt := _grounded_excerpt(
                                str(node["excerpt"]), chunk.text
                            )
                        )
                    ]
                    if grounded:
                        chunk, exact_excerpt = next(
                            (
                                item
                                for item in grounded
                                if item[0].chunk_id == node["chunk_id"]
                            ),
                            next(
                                (
                                    item
                                    for item in grounded
                                    if item[0].page_number == node["page_number"]
                                ),
                                grounded[0],
                            ),
                        )
                        node.update(
                            chunk_id=chunk.chunk_id,
                            page_number=chunk.page_number,
                            section=chunk.section,
                            excerpt=exact_excerpt,
                        )
            for item in node.values():
                visit(item)

    visit(data)
    return type(value).model_validate(data)


def render_chunks(chunks: Iterable[DocumentChunk]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id} | page {item.page_number} | {item.section}]\n{item.text}"
        for item in chunks
    )
