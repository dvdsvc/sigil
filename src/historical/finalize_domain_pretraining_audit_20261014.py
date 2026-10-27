import json,hashlib,statistics,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'outputs/domain_pretraining_ab_20261014';RUN=OUT/'server_snapshot';ARMS=['direct_sft','cpt_then_sft']
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
server=json.loads((RUN/'comparison.json').read_text());cfg=json.loads((RUN/'config.json').read_text());assert server['complete'] and json.loads((RUN/'status.json').read_text())['stage']=='complete'
assert hashlib.sha256((OUT/'server_files/scripts/train_domain_pretraining_ab_20261014.py').read_bytes()).hexdigest()==cfg['runner_sha256']
for n,h in cfg['data_manifest']['files_sha256'].items():assert hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h
inits=[json.loads((RUN/a/'INITIALIZATION.json').read_text()) for a in ARMS];assert inits[0]==inits[1] and inits[0]['fresh_common_base_sha256']==cfg['base_weight_sha256'] and inits[0]['no_prior_task_adapter_loaded']
cpt=[r for r in read(RUN/'cpt_then_sft/training_steps.jsonl') if r['phase']=='CPT'];schedule=read(RUN/'cpt_schedule.jsonl');assert len(cpt)==len(schedule)==1464 and len({r['id'] for r in cpt})==1464 and [r['step'] for r in cpt]==list(range(1,1465))
for a,b in zip(schedule,cpt):
 for k in ['id','text_sha256','target_tokens']:assert a[k]==b[k]
 assert b['input_tokens']==a['target_tokens']+1
assert sum(r['target_tokens'] for r in cpt)==2860163
sf=read(RUN/'sft_schedule.jsonl');assert len(sf)==72
for arm in ARMS:
 rows=[r for r in read(RUN/arm/'training_steps.jsonl') if r['phase']=='SFT'];assert len(rows)==72 and [r['step'] for r in rows]==list(range(1,73))
 for a,b in zip(sf,rows):
  for k in ['id','text_sha256','input_tokens']:assert a[k]==b[k]
  assert a['answer_tokens']==b['target_tokens']
 assert sum(r['target_tokens'] for r in rows)==22659
preds={a:read(RUN/a/'full_document_predictions.jsonl') for a in ARMS};ev=read(OUT/'eval_full19.jsonl');assert len(ev)==19
for arm,rows in preds.items():
 assert len(rows)==19 and [r['id'] for r in rows]==[r['id'] for r in ev]
 for a,b in zip(ev,rows):
  for k in ['paper','text','text_sha256','known_reference']:assert a[k]==b[k]
  assert hashlib.sha256(b['text'].encode()).hexdigest()==b['text_sha256'] and not b['input_truncated']
  raw=b['raw_output'].strip()
  if raw.startswith('```') and raw.endswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
  try:
   x=json.loads(raw);valid=isinstance(x,dict) and set(x)=={'entity_ids'} and isinstance(x['entity_ids'],list) and all(isinstance(n,str) for n in x['entity_ids']);names=list(dict.fromkeys(x['entity_ids'])) if valid else []
  except Exception:valid=False;names=[]
  assert valid==b['json_usable'] and names==b['prediction']['entity_ids']
  g=set(sum(b['known_reference'].values(),[]));q=set(names);assert len(g)==b['known_reference_count'] and len(g&q)==b['known_recovered']
  assert not b['output_limit_reached'] or b['output_tokens']==4096
for a,b in zip(preds[ARMS[0]],preds[ARMS[1]]):assert a['input_tokens']==b['input_tokens']
bad={r['id'] for rs in preds.values() for r in rs if not r['json_usable'] or r['output_limit_reached']}
def metric(rows):
 g=sum(r['known_reference_count'] for r in rows);t=sum(r['known_recovered'] for r in rows);q=sum(len(set(r['prediction']['entity_ids'])) for r in rows);mac=[r['known_recovered']/r['known_reference_count'] for r in rows if r['known_reference_count']]
 return {'n':len(rows),'json_usable':sum(r['json_usable'] for r in rows),'output_limit_reached':sum(r['output_limit_reached'] for r in rows),'known_recovered':t,'known_reference_count':g,'known_micro_recall':t/g if g else None,'mean_per_article_known_recall':statistics.mean(mac) if mac else None,'articles_with_known_references':len(mac),'predicted_ids':q,'additional_outputs_unreviewed':q-t,'nonliteral_outputs':sum(len(r['nonliteral_outputs']) for r in rows),'inference_seconds':sum(r['seconds'] for r in rows)}
result={'complete':True,'independently_recomputed':True,'same_original_base_and_fresh_LoRA_hash_verified':True,'same72_full_SFT_updates_and_labels_verified':True,'CPT1464_updates2860163_target_tokens_verified':True,'all19_full_inputs_identical_each_arm_no_truncation':True,'full19':{a:metric(rs) for a,rs in preds.items()},'joint_success':{a:metric([r for r in rs if r['id'] not in bad]) for a,rs in preds.items()},'jointly_excluded_failed_ids':sorted(bad),'CPT_extra_compute_not_matched':True,'weak_known_inventory_incomplete_not_true_precision_F1':True,'19_articles_previously_inspected_not_new_blind_test':True,'no_test_based_checkpoint_selection':True}
for subset in ['full19','joint_success']:
 for arm in ARMS:
  for k in ['n','json_usable','output_limit_reached','known_recovered','known_reference_count','known_micro_recall','mean_per_article_known_recall','predicted_ids']:assert result[subset][arm][k]==server[subset][arm][k]
per=[]
for a,b in zip(preds['direct_sft'],preds['cpt_then_sft']):
 g=set(sum(a['known_reference'].values(),[]));q0=set(a['prediction']['entity_ids']);q1=set(b['prediction']['entity_ids'])
 per.append({'id':a['id'],'paper':a['paper'],'known_reference_count':len(g),'direct_json_usable':a['json_usable'],'CPT_json_usable':b['json_usable'],'direct_output_limit':a['output_limit_reached'],'CPT_output_limit':b['output_limit_reached'],'direct_recovered':len(g&q0),'CPT_recovered':len(g&q1),'recovered_delta_CPT_minus_direct':len(g&q1)-len(g&q0),'direct_predicted_ids':len(q0),'CPT_predicted_ids':len(q1),'direct_seconds':a['seconds'],'CPT_seconds':b['seconds'],'joint_success':a['id'] not in bad,'gained_known_ids':sorted((q1-q0)&g),'lost_known_ids':sorted((q0-q1)&g)})
result['articles_known_recovery_improved']=sum(p['recovered_delta_CPT_minus_direct']>0 for p in per);result['articles_known_recovery_declined']=sum(p['recovered_delta_CPT_minus_direct']<0 for p in per);result['articles_known_recovery_unchanged']=sum(p['recovered_delta_CPT_minus_direct']==0 for p in per)
result['CPT_validation']={n:json.loads((RUN/'cpt_then_sft'/f'{n}_CPT_validation.json').read_text()) for n in ['before','after']};v=result['CPT_validation'];assert v['before']['target_tokens']==v['after']['target_tokens']==54960 and v['before']['sequences']==v['after']['sequences']==32
result['training_minutes']={'CPT':json.loads((RUN/'cpt_then_sft/CPT_TRAINING_SUMMARY.json').read_text())['training_seconds']/60,**{a+'_SFT':json.loads((RUN/a/'SFT_TRAINING_SUMMARY.json').read_text())['training_seconds']/60 for a in ARMS}}
save('INDEPENDENT_COMPARISON.json',result);save('PER_ARTICLE_COMPARISON.json',per)
with (OUT/'PER_ARTICLE_COMPARISON.csv').open('w') as f:
 keys=[k for k in per[0] if k not in ['gained_known_ids','lost_known_ids']];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:p[k] for k in keys} for p in per)
print(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({'changed_articles':[p for p in per if p['recovered_delta_CPT_minus_direct']]},ensure_ascii=False,indent=2))
