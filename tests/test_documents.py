import hashlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pypdf import PdfWriter

from brd_srs_testgen.documents import (
    DocumentError,
    canonicalize_source_references,
    chunk_pages,
    extract_pages,
    normalize_text,
    verify_source_reference,
)
from brd_srs_testgen.models import (
    Requirement,
    RequirementBatch,
    RequirementPriority,
    RequirementType,
    SourceReference,
)


def test_chunk_ids_are_stable_page_aware_and_bounded() -> None:
    pages = [(1, "AUTHENTICATION\n" + "word " * 20), (2, "2.1 Audit\nAudit text")]

    first = chunk_pages(pages, max_chars=30)
    second = chunk_pages(pages, max_chars=30)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert {item.page_number for item in first} == {1, 2}
    assert all(len(item.text) <= 30 for item in first)
    assert first[0].section == "AUTHENTICATION"


def test_chunk_splits_preserve_normalized_text() -> None:
    text = "  alpha   beta\tgamma\n delta  "

    chunks = chunk_pages([(1, text)], max_chars=10)

    assert " ".join(chunk.text for chunk in chunks) == normalize_text(text)
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.text == normalize_text(chunk.text) for chunk in chunks)
    assert all(
        chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        for chunk in chunks
    )


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


def test_source_excerpt_ignores_pdf_bullet_punctuation() -> None:
    chunks = chunk_pages([(1, "• Easy to use • Easy to learn")])
    reference = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        excerpt="Easy to use. Easy to learn.",
    )

    assert verify_source_reference(reference, chunks)


def test_empty_source_excerpt_is_rejected() -> None:
    chunks = chunk_pages([(1, "The system shall authenticate users.")])
    reference = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        excerpt="  ",
    )

    assert not verify_source_reference(reference, chunks)


def test_canonicalizes_grounded_reference_metadata() -> None:
    chunks = chunk_pages([(1, "The system shall authenticate users.")])
    batch = RequirementBatch(
        requirements=[
            Requirement(
                requirement_id="REQ-001",
                title="Authenticate",
                description="Authenticate users.",
                requirement_type=RequirementType.FUNCTIONAL,
                module="Access",
                priority=RequirementPriority.HIGH,
                source_references=[
                    SourceReference(
                        chunk_id="p00001-c001-wrong",
                        page_number=99,
                        excerpt="system shall authenticate users",
                    )
                ],
            )
        ]
    )

    fixed = canonicalize_source_references(batch, chunks)

    assert verify_source_reference(fixed.requirements[0].source_references[0], chunks)


def test_canonicalizes_extract_with_skipped_intermediate_words() -> None:
    chunks = chunk_pages(
        [
            (1, "Unrelated introduction."),
            (
                2,
                "System prompts the user to select a product from the list. "
                "System removes the product.",
            ),
        ]
    )
    reference = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        excerpt="System prompts the user to select a product. System removes the product.",
    )
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Remove product",
        description="Remove a product.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Inventory",
        priority=RequirementPriority.HIGH,
        source_references=[reference],
    )

    fixed = canonicalize_source_references(
        RequirementBatch(requirements=[requirement]), chunks
    )

    assert fixed.requirements[0].source_references[0].excerpt == chunks[1].text.rstrip(
        "."
    )
    assert verify_source_reference(fixed.requirements[0].source_references[0], chunks)


def test_canonicalizes_extract_interrupted_by_step_numbers() -> None:
    chunks = chunk_pages(
        [
            (
                1,
                "Basic Scenario: 1. Customer clicks button or link to initiate "
                "logout process. 2. System terminates the session cookie. "
                "3. System displays home page.",
            )
        ]
    )
    reference = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        excerpt=(
            "Customer clicks the button or link to initiate logout process. "
            "System terminates the session cookie. System displays home page."
        ),
    )
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Logout",
        description="Log out.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Access",
        priority=RequirementPriority.HIGH,
        source_references=[reference],
    )

    fixed = canonicalize_source_references(
        RequirementBatch(requirements=[requirement]), chunks
    )

    assert verify_source_reference(fixed.requirements[0].source_references[0], chunks)


def test_canonicalizes_excerpt_with_explicit_omission() -> None:
    chunks = chunk_pages(
        [
            (
                1,
                "System prompts the administrator for user details. "
                "System validates every required field. "
                "System creates a new account with desired privileges.",
            )
        ]
    )
    reference = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        excerpt=(
            "System prompts the administrator for user details. ... "
            "System creates a new account with desired privileges."
        ),
    )
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Create user",
        description="Create a user.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Access",
        priority=RequirementPriority.HIGH,
        source_references=[reference],
    )

    fixed = canonicalize_source_references(
        RequirementBatch(requirements=[requirement]), chunks
    )

    assert verify_source_reference(fixed.requirements[0].source_references[0], chunks)


def test_canonicalizes_substantial_exact_passage_from_composite_excerpt() -> None:
    chunks = chunk_pages(
        [
            (
                7,
                "The requirements specify access controls for system records. "
                "The System must allow the user to limit access to cases to "
                "specified users or user groups.",
            )
        ]
    )
    claimed_excerpt = (
        "The System must allow the user to limit access to cases to specified "
        "users or user groups. The requirements specify unrelated controls "
        "for correspondences, files, records, and system functionality."
    )
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Limit case access",
        description="Limit case access to specified users or groups.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Access",
        priority=RequirementPriority.HIGH,
        source_references=[
            SourceReference(
                chunk_id=chunks[0].chunk_id,
                page_number=7,
                excerpt=claimed_excerpt,
            )
        ],
    )

    fixed = canonicalize_source_references(
        RequirementBatch(requirements=[requirement]), chunks
    )
    reference = fixed.requirements[0].source_references[0]

    assert reference.excerpt != claimed_excerpt
    assert verify_source_reference(reference, chunks)


def test_empty_pdf_text_is_rejected() -> None:
    with pytest.raises(DocumentError, match="extractable text"):
        chunk_pages([(1, "  "), (2, "")])


def test_encrypted_pdf_is_rejected() -> None:
    reader = SimpleNamespace(is_encrypted=True, pages=[])
    with patch("brd_srs_testgen.documents.PdfReader", return_value=reader):
        with pytest.raises(DocumentError, match="Encrypted"):
            extract_pages(b"pdf")


def test_blank_pdf_retains_page_accounting() -> None:
    pdf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(pdf)

    assert extract_pages(pdf.getvalue()) == [(1, "")]


def test_malformed_pdf_is_rejected() -> None:
    with pytest.raises(DocumentError, match="readable PDF"):
        extract_pages(b"not a PDF")
