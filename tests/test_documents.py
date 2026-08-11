from types import SimpleNamespace
from unittest.mock import patch

import pytest

from brd_srs_testgen.documents import (
    DocumentError,
    chunk_pages,
    extract_pages,
    verify_source_reference,
)
from brd_srs_testgen.models import SourceReference


def test_chunk_ids_are_stable_page_aware_and_bounded() -> None:
    pages = [(1, "AUTHENTICATION\n" + "word " * 20), (2, "2.1 Audit\nAudit text")]

    first = chunk_pages(pages, max_chars=30)
    second = chunk_pages(pages, max_chars=30)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert {item.page_number for item in first} == {1, 2}
    assert all(len(item.text) <= 30 for item in first)
    assert first[0].section == "AUTHENTICATION"


def test_source_excerpt_must_exist_in_referenced_chunk() -> None:
    chunks = chunk_pages([(1, "The system shall authenticate users.")])
    valid = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        excerpt="system shall authenticate users",
    )
    invalid = valid.model_copy(update={"excerpt": "invented evidence"})

    assert verify_source_reference(valid, chunks)
    assert not verify_source_reference(invalid, chunks)


def test_empty_pdf_text_is_rejected() -> None:
    with pytest.raises(DocumentError, match="extractable text"):
        chunk_pages([(1, "  "), (2, "")])


def test_encrypted_pdf_is_rejected() -> None:
    reader = SimpleNamespace(is_encrypted=True, pages=[])
    with patch("brd_srs_testgen.documents.PdfReader", return_value=reader):
        with pytest.raises(DocumentError, match="Encrypted"):
            extract_pages(b"pdf")
