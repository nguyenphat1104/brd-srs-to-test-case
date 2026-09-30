"""Build review catalogs from assistant-authored rows and exact source spans.

No network or model calls. The generated drafts remain unapproved.
"""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from brd_srs_testgen.documents import chunk_pages, extract_pages, verify_source_reference
from brd_srs_testgen.models import CoverageUnitBatch, SourceReference


def main():
    root=Path(__file__).resolve().parent
    plan=json.loads((root/'authorization.json').read_text())
    rows=list(csv.DictReader((root/'catalog-drafts.tsv').open(),delimiter='\t'))
    results={}
    errors=[]
    for document in plan['documents']:
        key=document['document_id']; data=Path(document['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==document['sha256']
        chunks=chunk_pages(extract_pages(data))
        units=[]
        for row in [r for r in rows if r['document']==key]:
            matches=[c for c in chunks if c.page_number==int(row['page']) and row['anchor'] in c.text]
            if len(matches)!=1:
                errors.append(f"{key} page {row['page']}: {row['title']}: unmatched anchor {row['anchor']!r}")
                continue
            chunk=matches[0];start=chunk.text.index(row['anchor'])
            words=list(re.finditer(r'\S+',chunk.text[start:]))
            # The entire excerpt is copied, without fuzzy rewriting or joining fragments.
            end=start+words[min(25,len(words))-1].end()
            excerpt=chunk.text[start:end]
            reference={'chunk_id':chunk.chunk_id,'page_number':chunk.page_number,'section':chunk.section,'excerpt':excerpt}
            assert verify_source_reference(SourceReference.model_validate(reference),chunks)
            units.append({'unit_id':f'CU-{len(units)+1:03d}','title':row['title'],'description':row['description'],
                          'unit_type':row['type'],'source_references':[reference]})
        results[key]=(document,chunks,CoverageUnitBatch(units=units))
    if errors:
        raise ValueError('\n'.join(errors))
    for key,(document,chunks,batch) in results.items():
        assert batch.units and len({u.unit_id for u in batch.units})==len(batch.units)
        assert all(verify_source_reference(ref,chunks) for u in batch.units for ref in u.source_references)
        (root/f'{key}-catalog.draft.json').write_text(batch.model_dump_json(indent=2)+'\n')
        lines=[f"# {key} — {Path(document['path']).name}",'',
               '**Status: assistant-authored draft; human review and approval required.**','',
               'These are proposed testable obligations, not generated test cases. Page numbers below are PDF page numbers. Read source-review-notes.md for ambiguities and scope decisions. Exact citation matching does not by itself establish that a description is supported or exhaustive.','',
               'The raw Gemini attempt is retained separately. Descriptive glossary entries and document-purpose statements from that attempt are not treated as independent product obligations.','',
               f'Total proposed units: **{len(batch.units)}**. Source SHA-256: `{document["sha256"]}`.','',
               '| ID | Proposed obligation | PDF page | Exact source excerpt |','|---|---|---:|---|']
        for unit in batch.units:
            ref=unit.source_references[0]
            values=[unit.unit_id,unit.description,str(ref.page_number),ref.excerpt]
            lines.append('| '+' | '.join(v.replace('|','\\|') for v in values)+' |')
        (root/f'{key}-catalog-review.md').write_text('\n'.join(lines)+'\n')
        print(f'{key}: {len(batch.units)} units; every citation is an exact source span; types {dict(Counter(u.unit_type for u in batch.units))}')


if __name__=='__main__':
    main()
