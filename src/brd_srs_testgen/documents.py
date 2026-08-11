from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable

from pypdf import PdfReader

from .models import DocumentChunk, SourceReference


class DocumentError(ValueError):
    pass


def normalize_text(text: str) -> str:
    return " ".join(text.split())


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
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            yield remaining
            return
        cut = remaining.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        yield remaining[:cut]
        remaining = remaining[cut:]


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
        for sequence, piece in enumerate(_pieces(text, max_chars), 1):
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
    return bool(
        chunk
        and chunk.page_number == reference.page_number
        and normalize_text(reference.excerpt).casefold()
        in normalize_text(chunk.text).casefold()
    )


def render_chunks(chunks: Iterable[DocumentChunk]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id} | page {item.page_number} | {item.section}]\n{item.text}"
        for item in chunks
    )
