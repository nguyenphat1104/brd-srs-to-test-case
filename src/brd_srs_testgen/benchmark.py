"""Reproducible research runs, blinded exports and paired document-level analysis.

Run `python -m brd_srs_testgen.benchmark --help`. No model calls occur in
inventory, freeze or summarize. Execution requires an approved frozen specification.
"""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean
from uuid import NAMESPACE_URL, uuid5

from .documents import chunk_pages, extract_pages, verify_source_reference
from .models import CoverageCatalog, CoverageCatalogStatus, CoverageUnitBatch, RunResult, RunStatus, RunType, default_agent_setups
from .runner import EVALUATOR_VERSION, JUDGE_TOKEN_CEILING, ProviderSettings, implementation_snapshot, run_generation
from .storage import RunRepository, StorageError

DEVELOPMENT_HASH = 'fe7216c0e95efcd917d0b7d150ee8a4841131ad9c4f9b2e3b9316a803aa80a14'
CONDITIONS = {
    'single': (RunType.SINGLE_PROMPT, 'corrected'),
    'staged': (RunType.STAGED_SINGLE_AGENT, 'corrected'),
    'multi_corrected': (RunType.CENTRALIZED_MULTI_AGENT, 'corrected'),
    'multi_efficient': (RunType.CENTRALIZED_MULTI_AGENT, 'efficient'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inventory(directory: Path):
    rows = []
    for path in sorted(directory.glob('*.pdf')):
        data = path.read_bytes()
        item = {'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        try:
            pages = extract_pages(data)
            chunks = chunk_pages(pages)
            item.update(pages=len(pages),chunks=len(chunks),characters=sum(len(c.text) for c in chunks))
        except ValueError as error:
            item['error'] = str(error)
        item['development_only'] = item['sha256'] == DEVELOPMENT_HASH
        rows.append(item)
    return rows


def freeze(spec: dict) -> dict:
    if spec.get('held_out_approved') is not True:
        raise ValueError('Confirm held-out document selection before freezing the benchmark.')
    if type(spec.get('max_total_budget_tokens')) is not int or spec['max_total_budget_tokens'] < 1:
        raise ValueError('A positive total token-accounting budget is required.')
    if type(spec.get('repeats')) is not int or spec['repeats'] < 1:
        raise ValueError('repeats must be positive.')
    if type(spec.get('run_token_ceiling')) is not int or spec['run_token_ceiling'] < 1:
        raise ValueError('run_token_ceiling must be positive.')
    margin = spec.get('f1_tolerance')
    if margin is not None and (not isinstance(margin,(int,float)) or not 0 <= margin <= 1):
        raise ValueError('f1_tolerance must be null or between zero and one.')
    conditions = spec.get('conditions', list(CONDITIONS))
    if set(conditions) != set(CONDITIONS) or len(conditions) != len(CONDITIONS):
        raise ValueError('Include all three architectures plus the corrected multi-agent control.')
    documents = []
    for index, filename in enumerate(spec['documents'],1):
        path = Path(filename).resolve()
        data = path.read_bytes()
        document_hash = hashlib.sha256(data).hexdigest()
        if document_hash == DEVELOPMENT_HASH:
            raise ValueError('Telescope is development data, not held out.')
        chunk_pages(extract_pages(data))
        documents.append({'document_id':f'D{index:03d}','path':str(path),'sha256':document_hash})
    if not documents or len({d['sha256'] for d in documents}) != len(documents):
        raise ValueError('Held-out documents must be nonempty and unique.')
    result = {**spec,'conditions':conditions,'documents':documents,
              'implementation':implementation_snapshot(),'evaluator_version':EVALUATOR_VERSION,
              'seed':spec.get('seed',20260929), 'protocol':'cost-quality-v1'}
    result['manifest_sha256'] = digest(result)
    return result


def validate_manifest(manifest):
    if manifest['manifest_sha256'] != digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('Frozen manifest was modified.')
    if manifest['implementation'] != implementation_snapshot() or manifest['evaluator_version'] != EVALUATOR_VERSION:
        raise ValueError('Implementation/evaluator changed after freezing; freeze a new experiment.')
    for document in manifest['documents']:
        if hashlib.sha256(Path(document['path']).read_bytes()).hexdigest() != document['sha256']:
            raise ValueError('A held-out document changed after freezing.')


def trial_order(manifest):
    trials = [(d,c,r) for d in manifest['documents'] for c in manifest['conditions'] for r in range(manifest['repeats'])]
    random.Random(manifest['seed']).shuffle(trials)
    return trials


def _trial_id(manifest, document, condition, repeat):
    return uuid5(NAMESPACE_URL,f"{manifest['manifest_sha256']}:{document['sha256']}:{condition}:{repeat}").hex


def source_pack(manifest, output):
    """Independent annotation inputs, prepared before any generated suites exist."""
    validate_manifest(manifest)
    output.mkdir(parents=True,exist_ok=True)
    for document in manifest['documents']:
        chunks=chunk_pages(extract_pages(Path(document['path']).read_bytes()))
        (output/f"{document['document_id']}-source.json").write_text(json.dumps({
            'document_id':document['document_id'],'document_hash':document['sha256'],
            'chunks':[chunk.model_dump(mode='json') for chunk in chunks]},indent=2)+'\n')
    (output/'catalog-schema.json').write_text(json.dumps(CoverageUnitBatch.model_json_schema(),indent=2)+'\n')


def import_catalog(manifest, document_id, units, repository, *, approve=False):
    validate_manifest(manifest)
    document=next((d for d in manifest['documents'] if d['document_id']==document_id),None)
    if document is None:
        raise ValueError('Document is not in the frozen experiment.')
    batch=CoverageUnitBatch.model_validate(units)
    chunks=chunk_pages(extract_pages(Path(document['path']).read_bytes()))
    if any(not verify_source_reference(ref,chunks) for unit in batch.units for ref in unit.source_references):
        raise ValueError('Every catalog reference must be an exact source citation.')
    catalog=CoverageCatalog(catalog_id=f"{document['sha256'][:16]}-{hashlib.sha256(EVALUATOR_VERSION.encode()).hexdigest()[:12]}",
        document_hash=document['sha256'],evaluator_version=EVALUATOR_VERSION,
        status=CoverageCatalogStatus.MACHINE_FROZEN,units=batch.units,created_at=datetime.now(UTC))
    existing=repository.load_coverage_catalog(document['sha256'],EVALUATOR_VERSION)
    if existing is not None:
        if existing.units != catalog.units:
            raise ValueError('An immutable catalog already exists with different units.')
        catalog=existing
    else:
        catalog=repository.save_coverage_catalog(catalog)
    return repository.approve_coverage_catalog(catalog.catalog_id) if approve else catalog


def execute(manifest, repository, output: Path, *, generation=run_generation):
    validate_manifest(manifest)
    # Require source catalogs to be frozen and independently approved before inspecting results.
    for document in manifest['documents']:
        catalog = repository.load_coverage_catalog(document['sha256'],EVALUATOR_VERSION)
        if catalog is None or catalog.status is not CoverageCatalogStatus.APPROVED:
            raise ValueError(f"{document['document_id']} needs an independently approved coverage catalog.")
    output.mkdir(parents=True,exist_ok=True)
    rows, spent = [], 0
    for document, condition, repeat in trial_order(manifest):
        request_id = _trial_id(manifest,document,condition,repeat)
        run_id = f"{document['sha256'][:12]}-{request_id}"
        try:
            result = repository.load_run(run_id)
        except StorageError as error:
            if str(error) != "Run does not exist.":
                raise
            result = None
        if result is not None:
            spent += sum(c.budget_tokens for c in result.call_attempts)
            if result.manifest.status is RunStatus.RUNNING:
                raise ValueError('An interrupted experiment trial exists. Do not silently reuse it as a fresh repetition; inspect its accounted attempts.')
        else:
            reserve = manifest['run_token_ceiling'] + JUDGE_TOKEN_CEILING
            if spent + reserve > manifest['max_total_budget_tokens']:
                break
            run_type, profile = CONDITIONS[condition]
            settings = ProviderSettings(provider='gemini',model=manifest['model'],
                thinking_level=manifest.get('thinking_level','minimal'),
                token_ceiling=manifest['run_token_ceiling'], pipeline_profile=profile,
                api_key=os.environ.get('GEMINI_API_KEY',''), agent_setups=default_agent_setups())
            result = generation(Path(document['path']).read_bytes(),Path(document['path']).name,
                                run_type,settings,repository=repository,request_id=request_id)
            spent += sum(c.budget_tokens for c in result.call_attempts)
        rows.append({'document_id':document['document_id'],'condition':condition,'repeat':repeat,
                     'run_id':run_id,'result':result.model_dump(mode='json')})
        (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        if spent > manifest['max_total_budget_tokens']:
            break
    export_blind(rows,output)
    summary = summarize(rows,manifest.get('f1_tolerance'),manifest['seed'])
    summary['planned_trials'] = len(trial_order(manifest))
    summary['recorded_trials'] = len(rows)
    summary['budget_accounted_tokens'] = spent
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def export_blind(rows,output):
    blind = output/'blind'; private = output/'private'
    blind.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
    mapping = {}
    for row in rows:
        result = RunResult.model_validate(row['result'])
        if result.bundle is None:
            continue
        key = hashlib.sha256(('blinded:'+row['run_id']).encode()).hexdigest()[:16]
        (blind/f'{key}.json').write_text(json.dumps({'suite_id':key,'document_id':row['document_id'],
            'artifacts':result.bundle.model_dump(mode='json')},indent=2)+'\n')
        mapping[key] = {field:row[field] for field in ('run_id','condition','repeat','document_id')}
    rating_path = blind/'ratings.csv'
    if not rating_path.exists():
        with rating_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['suite_id','rater_id','source_coverage','groundedness','executability','redundancy_control','critical_errors','notes'])
            writer.writerows([key,'','','','','','',''] for key in sorted(mapping))
    (private/'unblinding.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (blind/'rating-instructions.md').write_text('Rate source coverage, groundedness, executability and redundancy independently on the existing 1–4 rubric. Check expected-result correctness, comparison operators, negation and critical boundaries against the source. Record critical errors separately. Do not view private unblinding data until ratings are frozen.\n')


def paired_interval(differences,seed=20260929):
    if len(differences) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(differences,k=len(differences))) for _ in range(2000))
    return [means[49],means[1949]]


def summarize(rows, tolerance=None, seed=20260929):
    result = {'conditions':{},'paired_document_differences':{},
              'quality_preservation':'Not established: requires complete evaluations and blinded human ratings, including critical-error checks.'}
    document_scores = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row['condition']==condition]
        runs = [RunResult.model_validate(row['result']) for row in selected]
        completed = [r for r in runs if r.manifest.status is RunStatus.COMPLETED]
        phases = {phase:sum(c.budget_tokens for r in runs for c in r.call_attempts if c.phase==phase)
                  for phase in ('generation','catalog','evaluation')}
        scores = [r.coverage.f1 for r in runs if r.coverage is not None]
        result['conditions'][condition] = {
            'attempts':len(runs),'completed':len(completed),'evaluation_unavailable':sum(r.coverage is None for r in runs),
            'budget_tokens_by_phase':phases,
            'generation_tokens_per_completed_suite':phases['generation']/len(completed) if completed else None,
            'mean_f1_when_available':mean(scores) if scores else None,
            'mean_latency_seconds':mean(r.metrics.latency_seconds for r in runs if r.metrics) if any(r.metrics for r in runs) else None,
            'human_approved_cost_per_suite':None,
        }
        for document in {r['document_id'] for r in selected}:
            values = [RunResult.model_validate(r['result']) for r in selected if r['document_id']==document]
            if values and all(r.coverage is not None for r in values):
                document_scores[(document,condition)] = mean(r.coverage.f1 for r in values)
    for comparator in ('single','staged','multi_corrected'):
        documents = sorted({d for d,c in document_scores if c==comparator and (d,'multi_efficient') in document_scores})
        differences = [document_scores[(d,'multi_efficient')]-document_scores[(d,comparator)] for d in documents]
        interval = paired_interval(differences,seed)
        result['paired_document_differences'][comparator] = {
            'documents':documents,'mean_f1_difference':mean(differences) if differences else None,
            'bootstrap_95_interval':interval,'predeclared_f1_tolerance':tolerance,
            'f1_noninferiority':'inconclusive' if interval is None or tolerance is None else
                ('interval exceeds negative tolerance' if interval[0] >= -tolerance else 'not supported'),
            'note':'Exploratory document-level bootstrap; more repeats do not replace independent documents. Missing trials and human ratings prevent a quality-equivalence claim.',
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('inventory');p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
    p=commands.add_parser('freeze');p.add_argument('spec',type=Path);p.add_argument('output',type=Path)
    p=commands.add_parser('run');p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    p=commands.add_parser('source-pack');p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    p=commands.add_parser('import-catalog');p.add_argument('manifest',type=Path);p.add_argument('document_id');p.add_argument('catalog',type=Path);p.add_argument('--approve',action='store_true',help='Confirm independent human review of the source catalog.')
    p=commands.add_parser('summarize');p.add_argument('results',type=Path);p.add_argument('output',type=Path);p.add_argument('--f1-tolerance',type=float)
    args=parser.parse_args()
    if args.command=='inventory': value=inventory(args.directory)
    elif args.command=='freeze': value=freeze(json.loads(args.spec.read_text()))
    elif args.command=='summarize': value=summarize(json.loads(args.results.read_text()),args.f1_tolerance)
    elif args.command=='source-pack':
        source_pack(json.loads(args.manifest.read_text()),args.output);print(f'Wrote {args.output}');return
    elif args.command=='import-catalog':
        repo=RunRepository(os.environ['DATABASE_URL']);repo.initialize()
        catalog=import_catalog(json.loads(args.manifest.read_text()),args.document_id,json.loads(args.catalog.read_text()),repo,approve=args.approve)
        print(f'{catalog.catalog_id}: {catalog.status.value}');return
    else:
        repo=RunRepository(os.environ['DATABASE_URL']);repo.initialize()
        print(json.dumps(execute(json.loads(args.manifest.read_text()),repo,args.output),indent=2));return
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(f'Wrote {args.output}')


if __name__=='__main__':
    main()
