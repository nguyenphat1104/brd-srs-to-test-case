"""Generate source-only catalog drafts within the approved pilot's shared budget.

Run in the existing application runtime with GEMINI_API_KEY and PYTHONPATH set.
This script never approves catalogs and never generates condition outputs.
"""
import argparse
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path

from brd_srs_testgen.benchmark import digest
from brd_srs_testgen.coverage import extract_coverage_catalog
from brd_srs_testgen.documents import chunk_pages, extract_pages
from brd_srs_testgen.models import AgentSetup
from brd_srs_testgen.pipelines import PipelineContext
from brd_srs_testgen.providers import BudgetLedger
from brd_srs_testgen.runner import (
    EVALUATOR_VERSION, JUDGE_MODEL, JUDGE_PROVIDER, ProviderSettings,
    _make_judge_provider, implementation_snapshot,
)


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    plan=json.loads((root/'authorization.json').read_text())
    assert plan['held_out_approved'] is True
    assert 0 < plan['catalog_preparation_ceiling'] <= plan['user_approved_total_budget_tokens'] == 1600000
    for document in plan['documents']:
        assert hashlib.sha256(Path(document['path']).read_bytes()).hexdigest()==document['sha256']
    if args.dry_run:
        print(json.dumps({'documents':len(plan['documents']),'catalog_ceiling':plan['catalog_preparation_ceiling'],'total_authorized':plan['user_approved_total_budget_tokens'],'paid_calls':0}))
        return
    with (root/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=root/'preparation-status.json'
        if status_path.exists():
            raise RuntimeError('Preparation already has a status record. Inspect/reconcile it; do not buy duplicate calls.')
        snapshot=implementation_snapshot()
        status={'state':'running','started_at':datetime.now(UTC).isoformat(),'authorization_sha256':digest(plan),
                'implementation':snapshot,'evaluator_version':EVALUATOR_VERSION,'documents':[],
                'budget_accounted_tokens':0,'human_approved':False}
        write_json(status_path,status)
        ledger=BudgetLedger(plan['catalog_preparation_ceiling'])
        settings=ProviderSettings(provider=JUDGE_PROVIDER,model=JUDGE_MODEL,token_ceiling=ledger.limit,
                                  api_key=os.environ['GEMINI_API_KEY'])
        try:
            provider=_make_judge_provider(settings,ledger)
            for document in plan['documents']:
                key=document['document_id']
                def append(kind,row):
                    with (root/f'{key}-{kind}.jsonl').open('a') as handle:
                        handle.write(row.model_dump_json()+'\n');handle.flush();os.fsync(handle.fileno())
                    if kind=='calls':
                        status['budget_accounted_tokens']=ledger.used
                        write_json(status_path,status)
                        print(f'{key}: {row.stage} task {row.task_index}: {row.status}; {row.budget_tokens} accounted tokens; cumulative {ledger.used}',flush=True)
                chunks=chunk_pages(extract_pages(Path(document['path']).read_bytes()))
                context=PipelineContext(provider=provider,efficient=True,token_ceiling=ledger.limit,
                    recovery_signature=digest({'document':document,'implementation':snapshot}),
                    call_recorder=lambda row:append('calls',row),stage_recorder=lambda row:append('checkpoints',row),
                    provider_names={'coverage_analyzer':JUDGE_PROVIDER},
                    agent_setups={'coverage_analyzer':AgentSetup(agent='coverage_analyzer',role='Independent quality judge')},
                    sanitize=lambda text:text.replace(settings.api_key,'[redacted]'))
                catalog=extract_coverage_catalog(context,chunks,document_hash=document['sha256'],
                    evaluator_version=EVALUATOR_VERSION,
                    catalog_id=f"{document['sha256'][:16]}-{hashlib.sha256(EVALUATOR_VERSION.encode()).hexdigest()[:12]}",
                    created_at=datetime.now(UTC))
                write_json(root/f'{key}-catalog.draft.json',{'units':[unit.model_dump(mode='json') for unit in catalog.units]})
                status['documents'].append({'document_id':key,'units':len(catalog.units),'status':'awaiting_human_review'})
                write_json(status_path,status)
                print(f'{key}: draft saved with {len(catalog.units)} source coverage units.',flush=True)
            status['state']='awaiting_human_review'
        except Exception as error:
            status['state']='failed'
            status['error_type']=type(error).__name__
            status['error']=str(error).replace(settings.api_key,'[redacted]')[:1000]
            raise
        finally:
            status['budget_accounted_tokens']=ledger.used
            status['remaining_total_budget_tokens']=plan['user_approved_total_budget_tokens']-ledger.used
            status['updated_at']=datetime.now(UTC).isoformat()
            write_json(status_path,status)


if __name__=='__main__':
    main()
