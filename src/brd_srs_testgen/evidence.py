"""Lossless, versioned evidence references for model handoffs."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from .documents import normalize_text
from .models import (
    CandidateRequirement, DocumentChunk, Requirement, Scenario,
    SourceReference, StrictModel, TestCase, ScenarioType, TestPriority, TestStep, CriticSeverity,
)

EVIDENCE_VERSION = 'spans-v1'


class SpanReference(StrictModel):
    evidence_id: str


class SpanCandidate(CandidateRequirement):
    source_references: list[SpanReference] = Field(min_length=1)


class ChunkDisposition(StrictModel):
    chunk_id: str
    status: Literal['extracted', 'non_testable', 'unresolved']
    reason: str = Field(min_length=1)


class ExtractionBatch(StrictModel):
    candidates: list[SpanCandidate]
    coverage: list[ChunkDisposition]


class GapAudit(StrictModel):
    missing_candidates: list[SpanCandidate]
    unresolved_chunk_ids: list[str]


class SpanScenario(Scenario):
    source_references: list[SpanReference] = Field(min_length=1)


class SpanScenarios(StrictModel):
    scenarios: list[SpanScenario]


class SpanTestCase(TestCase):
    source_references: list[SpanReference] = Field(min_length=1)


class SpanTests(StrictModel):
    test_cases: list[SpanTestCase]


class TestDesign(StrictModel):
    """One executable case and its scenario share scope, preconditions and evidence."""
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    scenario_type: ScenarioType
    requirement_ids: list[str] = Field(min_length=1)
    priority: TestPriority
    preconditions: list[str] = Field(default_factory=list)
    test_data: dict[str, JsonValue] = Field(default_factory=dict)
    steps: list[TestStep] = Field(min_length=1)
    source_references: list[SpanReference] = Field(min_length=1)


class TestDesignBatch(StrictModel):
    designs: list[TestDesign]


def alias_evidence(messages):
    """Shorten reference fields/registry labels only; leave quoted source text intact."""
    stable_to_local = {}
    def replace(match):
        stable = match[2]
        alias = stable_to_local.setdefault(stable, f'E{len(stable_to_local) + 1}')
        return match[1] + alias
    pattern = r'(?m)(^|"evidence_id":\s*")(E-[0-9a-f]{16})(?=:|")'
    wire = [{**message, 'content': re.sub(pattern, replace, message['content'])} for message in messages]
    return wire, {alias: stable for stable, alias in stable_to_local.items()}


def restore_evidence(value, aliases):
    if isinstance(value, BaseModel):
        return type(value).model_validate(restore_evidence(value.model_dump(mode='json'), aliases))
    if isinstance(value, list):
        return [restore_evidence(item, aliases) for item in value]
    if isinstance(value, dict):
        if set(value) == {'evidence_id'}:
            key = value['evidence_id']
            if key not in aliases:
                raise ValueError(f'Unknown or out-of-scope evidence ID {str(key)[:80]!r}. '
                                 f'Copy an exact label E1 through E{len(aliases)} from this task; never invent or reuse another task\'s label.')
            return {'evidence_id': aliases[key]}
        return {key: restore_evidence(item, aliases) for key, item in value.items()}
    return value


class SpanFinding(StrictModel):
    finding_id: str = Field(pattern=r'^FIND-\d{3,}$')
    severity: CriticSeverity
    finding_type: str = Field(min_length=1)
    artifact_ids: list[str] = Field(min_length=1)
    repair_kind: Literal['citation', 'relationship', 'semantic'] = 'semantic'
    required_action: str = Field(min_length=1)
    source_references: list[SpanReference] = Field(min_length=1)


class SpanCritique(StrictModel):
    accepted: bool
    findings: list[SpanFinding]


class RequirementEdit(StrictModel):
    title: str
    description: str
    ambiguities: list[str]
    dependency_ids: list[str]


class CuratorChoice(StrictModel):
    candidate_id: str
    action: Literal['retain', 'merge', 'reject']
    merge_into: str | None = None
    reason: str = Field(min_length=1)
    edit: RequirementEdit | None = None


class CuratorChoices(StrictModel):
    choices: list[CuratorChoice]


class EvidenceRegistry:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.spans: dict[str, dict] = {}
        self.by_chunk: dict[str, list[str]] = {c.chunk_id: [] for c in chunks}
        for chunk in chunks:
            words = list(re.finditer(r'\S+', chunk.text))
            # Adjacent spans remain visible together; overlap only a short trailing fragment.
            for start in range(0, len(words), 25):
                end = min(start + 25, len(words))
                if end - start < 5:
                    start = max(0, end - 5)
                if end - start >= 5:
                    self.add(chunk, words[start].start(), words[end - 1].end())
        self.base_ids = set(self.spans)

    def add(self, chunk: DocumentChunk, start: int, end: int) -> str:
        identity = f'{EVIDENCE_VERSION}:{chunk.chunk_id}:{chunk.content_hash}:{start}:{end}'
        key = 'E-' + hashlib.sha256(identity.encode()).hexdigest()[:16]
        if key not in self.spans:
            self.spans[key] = {
                'evidence_id': key, 'chunk_id': chunk.chunk_id, 'start': start, 'end': end,
                'page_number': chunk.page_number, 'section': chunk.section,
                'excerpt': chunk.text[start:end],
            }
            self.by_chunk[chunk.chunk_id].append(key)
        return key

    def reference_id(self, reference: dict) -> str:
        chunk = self.chunks.get(reference['chunk_id'])
        if chunk is None:
            raise ValueError('Citation refers to unavailable evidence.')
        quote = normalize_text(reference['excerpt'])
        text = normalize_text(chunk.text)
        start = text.find(quote)
        if start < 0 or not 5 <= len(quote.split()) <= 25:
            raise ValueError('Citation is not an exact 5-to-25-word source span.')
        # DocumentChunk text is canonical whitespace-normalized text.
        if text != chunk.text:
            raise ValueError('Evidence registry requires canonical chunk text.')
        return self.add(chunk, start, start + len(quote))

    def compact(self, value):
        if isinstance(value, BaseModel):
            value = value.model_dump(mode='json')
        if isinstance(value, list):
            return [self.compact(x) for x in value]
        if isinstance(value, dict):
            if {'chunk_id', 'page_number', 'excerpt'} <= value.keys():
                return {'evidence_id': self.reference_id(value)}
            return {key: self.compact(item) for key, item in value.items()}
        return value

    def hydrate(self, value, allowed_chunk_ids: set[str]):
        if isinstance(value, BaseModel):
            value = value.model_dump(mode='json')
        if isinstance(value, list):
            return [self.hydrate(x, allowed_chunk_ids) for x in value]
        if isinstance(value, dict):
            if set(value) == {'evidence_id'}:
                span = self.spans.get(value['evidence_id'])
                if span is None or span['chunk_id'] not in allowed_chunk_ids:
                    raise ValueError('Unknown or out-of-scope evidence ID.')
                return {key: span[key] for key in SourceReference.model_fields}
            return {key: self.hydrate(item, allowed_chunk_ids) for key, item in value.items()}
        return value

    def render(self, chunks: list[DocumentChunk], *, focus=None) -> str:
        selected = None
        if focus is not None:
            anchors = set()
            def visit(value):
                if isinstance(value, dict):
                    if 'evidence_id' in value:
                        anchors.add(value['evidence_id'])
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)
            visit(focus)
            allowed = {c.chunk_id for c in chunks}
            selected = set()
            for key in anchors:
                span = self.spans.get(key)
                if span is None or span['chunk_id'] not in allowed:
                    raise ValueError('Unknown or out-of-scope evidence ID.')
                # ponytail: local windows for handoffs; extraction/audits still read full source.
                selected.add(key)
                selected.update(other for other in self.by_chunk[span['chunk_id']]
                    if other in self.base_ids and self.spans[other]['start'] < span['end'] + 256
                    and self.spans[other]['end'] > span['start'] - 256)
        return '\n'.join(
            f"[{c.chunk_id} page={c.page_number} section={c.section}]\n" + '\n'.join(
                f"{key}: {self.spans[key]['excerpt']}" for key in self.by_chunk[c.chunk_id]
                if selected is None or key in selected
            ) + ('\nNo citeable five-word span; retain as unresolved context: ' + c.text
                 if not self.by_chunk[c.chunk_id] else '')
            for c in chunks
        )

    def dump(self) -> list[dict]:
        return list(self.spans.values())


def compact_index(requirements: list[Requirement]) -> list[dict]:
    return [{'id': r.requirement_id, 'title': r.title, 'module': r.module,
             'dependencies': r.dependency_ids} for r in requirements]


def record_table(records: list[dict]) -> dict:
    """Serialize repeated field names once, retaining every value and its order."""
    columns = list(dict.fromkeys(key for row in records for key in row))
    return {'columns': columns, 'rows': [[row.get(key) for key in columns] for row in records]}


def review_artifacts(bundle) -> dict:
    """Share only exactly equal fields with the test's referenced scenario."""
    data = bundle.model_dump(mode='json')
    scenarios = {row['scenario_id']: row for row in data['scenarios']}
    for case in data['test_cases']:
        scenario = scenarios.get(case['scenario_id'], {})
        shared = [key for key in ('title', 'preconditions', 'requirement_ids', 'source_references')
                  if key in scenario and case[key] == scenario[key]]
        case['scenario_fields'] = shared
        for key in shared:
            del case[key]
    return {kind: record_table(rows) for kind, rows in data.items()}


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
