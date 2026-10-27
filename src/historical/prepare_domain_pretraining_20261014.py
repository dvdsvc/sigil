"""Prepare source-isolated raw text CPT and identical real-fulltext ID SFT."""
import argparse, json, hashlib, re, unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from project_docx_with_math import project_docx, as_text

parser=argparse.ArgumentParser()
parser.add_argument('--legacy-root',type=Path,required=True)
parser.add_argument('--source-docx-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT=args.legacy_root
OUT=args.output
OUT.mkdir(parents=True,exist_ok=False)
SOURCE=args.source_docx_root
def read(p): return [json.loads(s) for s in p.read_text().splitlines()]
def sha(b): return hashlib.sha256(b).hexdigest()
def norm(s): return re.sub(r'[^\w]','',unicodedata.normalize('NFKC',s).lower())
def save(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write(n,rs): (OUT/n).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
groups=read(ROOT/'work/v4_projection/all_grouped_sources.jsonl')
audit=read(ROOT/'work/v4_projection/projection_audit.jsonl')
heldout=[r for r in groups if r['split']!='train']
evalrows=read(ROOT/'outputs/B_v21_union_training_20261014/evaluation/B_v21_union/full_documents_predictions.jsonl')
assert len(evalrows)==19
known=defaultdict(list)
for r in groups:
 for alias in [r.get('stem',''),r.get('title',''),r.get('doi','')]+[Path(c['path']).parent.name for c in r['docx_candidates']]:
  if alias:known[norm(alias)].append(r)
forbidden_aliases={norm(s) for r in heldout for s in [r.get('stem',''),r.get('title',''),r.get('doi','')]+[Path(c['path']).parent.name for c in r['docx_candidates']] if s}
forbidden_aliases|={norm(r['paper']) for r in evalrows}
forbidden_sha={r['text_sha256'] for r in evalrows}
forbidden_normalized={sha(norm(r['text']).encode()) for r in evalrows}
for r in audit:
 if norm(r['paper']) in forbidden_aliases:
  forbidden_sha.add(r['corrected_text_sha256'])
  forbidden_sha.add(r['source_docx_sha256'])
forbidden_groups={r['group_id'] for r in heldout}
records=[]
for path in sorted(SOURCE.rglob('full.docx')):
 rec={'paper':path.parent.name,'source_docx':str(path),'source_docx_sha256':sha(path.read_bytes())}
 try:
  blocks=project_docx(path);text=as_text(blocks)
  h=sha(text.encode());hn=sha(norm(text).encode());rec.update(text_sha256=h,normalized_text_sha256=hn,characters=len(text),tables=sum(b['type']=='table' for b in blocks),replacement_characters=text.count('\ufffd'))
  aliases={norm(path.parent.name)}
  aliases|={norm(s.strip()) for s in text.splitlines()[:6] if len(s.strip())>=20}
  for m in re.finditer(r'10\.\d{4,9}/[^\s<>"\[\]]+',text[:1800],re.I):aliases.add(norm(m.group().rstrip('.,;，。')))
  matched=[g for alias in aliases for g in known.get(alias,[])]
  gids={g['group_id'] for g in matched};rec['known_source_groups']=sorted(gids)
  rec['group_id']=sorted(gids)[0] if gids else 'text:'+hn
  reason=None
  if aliases&forbidden_aliases or gids&forbidden_groups or h in forbidden_sha or hn in forbidden_normalized or rec['source_docx_sha256'] in forbidden_sha:reason='known_validation_or_test_source'
  elif len(text)<2000:reason='too_short_for_full_article_corpus'
  elif rec['replacement_characters']/len(text)>.01:reason='replacement_character_ratio_over_one_percent'
  rec['excluded_reason']=reason
  if not reason:rec['text']=text
 except Exception as e:rec['excluded_reason']='projection_failed';rec['error']=str(e)
 records.append(rec)
eligible=[];seen={}
for r in records:
 if r['excluded_reason']:continue
 keys=[r['group_id'],r['normalized_text_sha256'],r['source_docx_sha256']]
 duplicate=next((seen[k] for k in keys if k in seen),None)
 if duplicate:r['excluded_reason']='duplicate_source';r['duplicate_of']=duplicate;continue
 for k in keys:seen[k]=r['paper']
 eligible.append(r)
assert len(eligible)>=144,len(eligible)
sft=read(ROOT/'outputs/sample_id_v4_full/model_and_evidence/run/train_full_documents.jsonl')
assert len(sft)==64 and not {r['text_sha256'] for r in sft}&{r['text_sha256'] for r in evalrows}
sfthashes={sha(norm(r['text']).encode()) for r in sft};sftg={r['group_id'] for r in sft}
ordered=sorted(eligible,key=lambda r:sha(('cpt-20261014:'+r['group_id']).encode()))
val=[r for r in ordered if r['normalized_text_sha256'] not in sfthashes and r['group_id'] not in sftg][:8]
assert len(val)==8
valg={r['group_id'] for r in val};train=[r for r in ordered if r['group_id'] not in valg][:128]
assert len(train)==128 and not {r['group_id'] for r in train}&valg
def plain(rs,split):
 return [{'id':'CPT_'+r['text_sha256'][:16],'paper':r['paper'],'group_id':r['group_id'],'text':r['text'],'text_sha256':r['text_sha256'],'source_docx_sha256':r['source_docx_sha256'],'characters':r['characters'],'tables':r['tables'],'split':split} for r in rs]
write('cpt_train_articles.jsonl',plain(train,'train'));write('cpt_validation_articles.jsonl',plain(val,'validation'))
union=[]
for r in sft:
 names=list(dict.fromkeys(sum(r['labels'].values(),[])))
 assert all(n in r['text'] for n in names)
 names.sort(key=lambda n:r['text'].find(n))
 union.append({k:r[k] for k in ['id','paper','group_id','text','text_sha256']}|{'labels':{'entity_ids':names},'split':'train','complete_document':True,'weak_labels_not_exhaustive':True})
assert sum(not r['labels']['entity_ids'] for r in union)==4
write('sft_full_train.jsonl',union)
evalclean=[]
for r in evalrows:
 ref=r['known_reference'];removed=[]
 if r['paper']=='erlian7':
  removed=[str(i) for i in range(1,14)];ref={'sample_ids':ref['sample_ids'],'spot_ids':[n for n in ref['spot_ids'] if n not in removed]}
 evalclean.append({k:r[k] for k in ['id','paper','text','text_sha256']}|{'known_reference':ref,'original_known_reference':r['known_reference'],'explicitly_removed_heating_steps':removed,'reference_status':'Incomplete known inventory; empty is UNKNOWN. Previously inspected development/regression articles, not new blind test.'})
assert sum(len(set(sum(r['known_reference'].values(),[]))) for r in evalclean)==210
write('eval_full19.jsonl',evalclean)
assert not {r['text_sha256'] for r in train+val}&{r['text_sha256'] for r in evalclean}
save('SOURCE_AUDIT.json',{'source_directory':str(SOURCE),'original_docx_count':len(records),'eligible_distinct_sources':len(eligible),'exclusion_counts':dict(Counter(r['excluded_reason'] for r in records if r['excluded_reason'])),'records':[{k:v for k,v in r.items() if k!='text'} for r in records],'source_alias_screening':'Known grouped splits, filename/title/leading DOI aliases, exact projected and normalized full-text hashes, DOCX hashes. Heuristic aliases are not exhaustive source identity proof.'})
save('DATA_MANIFEST.json',{'original_docx_count':len(records),'cpt_training_articles':128,'cpt_validation_articles':8,'cpt_training_characters':sum(r['characters'] for r in train),'cpt_validation_characters':sum(r['characters'] for r in val),'sft_training_articles':64,'sft_full_negative_articles':4,'eval_full_articles':19,'eval_known_reference_total':210,'CPT_train_and_validation_source_overlap':0,'CPT_validation_and_SFT_exact_normalized_text_overlap':0,'CPT_train_and_known_eval_exact_text_overlap':0,'data_selection':'Fixed SHA256 source order;8 independent language-model validation articles; first128 remaining eligible sources. One CPT epoch, not chosen on extraction test scores.','projection':'Original DOCX body paragraphs, tables and Office Math text retained; no new OCR, no aggressive cleanup, no synthetic contexts.','evaluation':'Previously inspected19 complete articles; regression comparison, not a new blind benchmark.','training_route':'Same untouched original Qwen3.5-0.8B and identical fresh LoRA init. direct_sft versus cpt_then_sft. Identical complete64-article SFT schedule and prompt both arms.','runner_limits':'CPT sequence2048 tokens; one epoch. Complete SFT articles never truncated.','files_sha256':{n:sha((OUT/n).read_bytes()) for n in ['cpt_train_articles.jsonl','cpt_validation_articles.jsonl','sft_full_train.jsonl','eval_full19.jsonl']}})
print(json.dumps(json.loads((OUT/'DATA_MANIFEST.json').read_text()),ensure_ascii=False,indent=2))
