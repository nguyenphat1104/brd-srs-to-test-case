from copy import deepcopy
from types import SimpleNamespace

import pytest

from brd_srs_testgen import benchmark
from brd_srs_testgen.models import CoverageCatalogStatus, RunStatus
from brd_srs_testgen.storage import StorageError
from tests.factories import completed_run


def specification(tmp_path, monkeypatch):
    path=tmp_path/'heldout.pdf';path.write_bytes(b'held-out fixture')
    monkeypatch.setattr(benchmark,'extract_pages',lambda _: [(1,'The system shall authenticate registered users.')])
    return {'held_out_approved':True,'documents':[str(path)],'repeats':2,
            'max_total_budget_tokens':1000000,'run_token_ceiling':100000,
            'model':'gemini-test','thinking_level':'minimal','f1_tolerance':None}


def test_frozen_manifest_detects_modified_document_or_config(tmp_path,monkeypatch):
    spec=specification(tmp_path,monkeypatch)
    manifest=benchmark.freeze(spec)
    benchmark.validate_manifest(manifest)
    changed=deepcopy(manifest);changed['repeats']=3
    with pytest.raises(ValueError,match='modified'):
        benchmark.validate_manifest(changed)
    (tmp_path/'heldout.pdf').write_bytes(b'changed')
    with pytest.raises(ValueError,match='document changed'):
        benchmark.validate_manifest(manifest)


def test_freezing_requires_holdout_selection_and_spend_limit(tmp_path,monkeypatch):
    spec=specification(tmp_path,monkeypatch)
    for update in ({'held_out_approved':False},{'max_total_budget_tokens':0}):
        with pytest.raises(ValueError): benchmark.freeze({**spec,**update})
    monkeypatch.setattr(benchmark,'DEVELOPMENT_HASH',benchmark.hashlib.sha256(b'held-out fixture').hexdigest())
    with pytest.raises(ValueError,match='development data'):
        benchmark.freeze(spec)


def test_budget_stops_before_any_paid_request_and_catalog_must_be_approved(tmp_path,monkeypatch):
    spec=specification(tmp_path,monkeypatch);spec['max_total_budget_tokens']=199999
    manifest=benchmark.freeze(spec)
    class Repo:
        approved=True
        def load_coverage_catalog(self,*_): return SimpleNamespace(status=CoverageCatalogStatus.APPROVED if self.approved else CoverageCatalogStatus.MACHINE_FROZEN)
        def load_run(self,*_): raise StorageError('Run does not exist.')
    repo=Repo()
    result=benchmark.execute(manifest,repo,tmp_path/'out',generation=lambda *args,**kwargs: pytest.fail('Unexpected paid call'))
    assert result['recorded_trials']==0
    assert result['budget_accounted_tokens']==0
    repo.approved=False
    with pytest.raises(ValueError,match='approved coverage catalog'):
        benchmark.execute(manifest,repo,tmp_path/'other')


def test_blind_export_excludes_condition_tokens_model_and_run_id(tmp_path):
    result=completed_run()
    rows=[{'document_id':'D001','condition':'multi_efficient','repeat':0,'run_id':result.manifest.run_id,'result':result.model_dump(mode='json')}]
    benchmark.export_blind(rows,tmp_path)
    text=next((tmp_path/'blind').glob('*.json')).read_text()
    for secret in ('multi_efficient',result.manifest.run_id,'charged_tokens','gemma4','latency_seconds'):
        assert secret not in text
    assert (tmp_path/'private'/'unblinding.json').exists()


def test_missing_evaluations_are_unavailable_not_zero_and_no_equivalence_claim():
    result=completed_run()
    rows=[{'document_id':'D001','condition':'multi_efficient','repeat':0,'run_id':result.manifest.run_id,'result':result.model_dump(mode='json')}]
    summary=benchmark.summarize(rows)
    assert summary['conditions']['multi_efficient']['mean_f1_when_available'] is None
    assert summary['conditions']['multi_efficient']['evaluation_unavailable']==1
    assert summary['paired_document_differences']['multi_corrected']['bootstrap_95_interval'] is None
    assert 'Not established' in summary['quality_preservation']
    assert benchmark.paired_interval([.1,.1,.1])==[pytest.approx(.1),pytest.approx(.1)]


def test_independent_catalog_import_requires_exact_evidence_and_explicit_approval(tmp_path,monkeypatch,repository):
    import json
    spec=specification(tmp_path,monkeypatch)
    manifest=benchmark.freeze(spec)
    benchmark.source_pack(manifest,tmp_path/'annotations')
    source=json.loads((tmp_path/'annotations'/'D001-source.json').read_text())['chunks'][0]
    unit={'unit_id':'CU-001','title':'Authentication','description':source['text'],'unit_type':'functional',
          'source_references':[{'chunk_id':source['chunk_id'],'page_number':source['page_number'],'section':source['section'],'excerpt':source['text']}]}
    units={'units':[unit]}
    catalog=benchmark.import_catalog(manifest,'D001',units,repository)
    assert catalog.status is CoverageCatalogStatus.MACHINE_FROZEN
    approved=benchmark.import_catalog(manifest,'D001',units,repository,approve=True)
    assert approved.status is CoverageCatalogStatus.APPROVED
    assert approved.units==catalog.units
    bad=deepcopy(units);bad['units'][0]['source_references'][0]['excerpt']='The system shall never authenticate registered users.'
    with pytest.raises(ValueError,match='exact source citation'):
        benchmark.import_catalog(manifest,'D001',bad,repository)


def test_database_failure_is_not_treated_as_permission_to_buy_another_trial(tmp_path,monkeypatch):
    manifest=benchmark.freeze(specification(tmp_path,monkeypatch))
    class Repo:
        def load_coverage_catalog(self,*_): return SimpleNamespace(status=CoverageCatalogStatus.APPROVED)
        def load_run(self,*_): raise StorageError('Database is unavailable.')
    with pytest.raises(StorageError,match='unavailable'):
        benchmark.execute(manifest,Repo(),tmp_path/'out',generation=lambda *_args,**_kwargs: pytest.fail('Unexpected paid call'))
