"""Fresh common-origin LoRA: raw-text CPT + identical whole-document SFT vs SFT."""
import argparse, json, hashlib, random, time, gc, traceback, statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
ROOT=a.workspace;DATA=a.data;RUN=ROOT/'runs/domain_pretraining_ab_20261014_01';RUN.mkdir(exist_ok=True)
BASE=ROOT/'models/Qwen3.5-0.8B';SEED=20261014;CPT_SEQLEN=2048
BASE_SHA='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
SYSTEM=json.loads((ROOT/'runs/B_v21_union_20261014_01/config.json').read_text())['system']
MODULES=['down_proj','gate_proj','q_proj','v_proj','k_proj','up_proj','o_proj']
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 p=RUN/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def status(**kw):save('status.json',kw)
def log(n,x):
 with (RUN/n).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps(x,ensure_ascii=False),flush=True)
assert sha(BASE/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA
manifest=json.loads((DATA/'DATA_MANIFEST.json').read_text())
for n,h in manifest['files_sha256'].items():assert sha(DATA/n)==h
tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
status(stage='encoding')

def encode_cpt(rs):
 result=[]
 for r in rs:
  assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
  ids=tok(r['text'],add_special_tokens=False)['input_ids']+[tok.eos_token_id]
  for start in range(0,len(ids),CPT_SEQLEN-1):
   target=ids[start:start+CPT_SEQLEN-1]
   seq=[ids[start-1] if start else tok.eos_token_id]+target
   result.append({'id':r['id']+f'_{start:07d}','article_id':r['id'],'paper':r['paper'],'group_id':r['group_id'],'text_sha256':r['text_sha256'],'token_start':start,'token_end':start+len(target),'ids':seq,'target_tokens':len(target)})
  assert sum(x['target_tokens'] for x in result if x['article_id']==r['id'])==len(ids)
 return result
cpt=encode_cpt(read(DATA/'cpt_train_articles.jsonl'));v=encode_cpt(read(DATA/'cpt_validation_articles.jsonl'))
val=[]
for aid in sorted({r['article_id'] for r in v}):
 rs=[r for r in v if r['article_id']==aid];indices=sorted({round(i*(len(rs)-1)/3) for i in range(4)});val.extend(rs[i] for i in indices)
random.Random(SEED).shuffle(cpt)
sftrows=read(DATA/'sft_full_train.jsonl');order=[r for r in sftrows if r['labels']['entity_ids']]+[r for r in sftrows if not r['labels']['entity_ids']]*3;random.Random(SEED).shuffle(order)
assert len(order)==72
def encode_sft(r):
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 text=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 inp=tok(text,add_special_tokens=False)['input_ids'];ans=tok(json.dumps(r['labels'],ensure_ascii=False)+tok.eos_token,add_special_tokens=False)['input_ids'];assert len(inp)<=64512 and len(ans)<=4096
 return {'row':r,'input':inp,'answer':ans}
sft=[encode_sft(r) for r in order];ev=read(DATA/'eval_full19.jsonl');assert len(ev)==19
config={'experiment':'domain_pretraining_ab_20261014_01','base_weight_sha256':BASE_SHA,'initialization':'Both arms start from untouched original base and identical freshly initialized LoRA; never inherit B/v4 task adapter.','system':SYSTEM,'seed':SEED,'lora':{'r':8,'alpha':16,'dropout':.05,'target_modules':MODULES},'CPT':{'articles':128,'epochs':1,'sequence_length':CPT_SEQLEN,'optimizer_updates':len(cpt),'target_tokens':sum(r['target_tokens'] for r in cpt),'learning_rate':2e-5,'warmup':32,'loss':'Causal next-token loss on every raw-text target token including EOS; first token prefixed by EOS, subsequent chunks carry prior token. No chat template, no ID labels, no cross-document packing, every article token covered exactly once as target.','validation_articles':8,'validation_sequences':len(val),'validation_target_tokens':sum(r['target_tokens'] for r in val),'validation_selection':'Up to4 evenly spaced whole sequences per article, fixed before training.'},'SFT':{'articles':64,'unique_positive_articles':60,'reviewed_negative_articles':4,'updates_each_arm':72,'negative_repetitions':3,'positive_repetitions':1,'learning_rate':5e-5,'warmup':5,'loss':'Answer-only cross entropy; identical complete-document input order, labels and dropout seed reset each arm.','input_truncation':False,'max_actual_input_tokens':max(len(r['input']) for r in sft)},'optimizer':'AdamW, weight_decay=0, fresh optimizer for SFT in both arms','precision':'bf16','batch':1,'gradient_accumulation':1,'evaluation':'Same19 previously inspected complete articles, identical union prompt, greedy/no thinking,4096 output tokens; incomplete labels. No test-dependent checkpoint choice.','checkpoint_selection':'Fixed final CPT epoch and fixed final72-update SFT; no best-on-test selection.','data_manifest':manifest,'runner_sha256':sha(Path(__file__))}
save('config.json',config)
(RUN/'cpt_schedule.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='ids'},ensure_ascii=False)+'\n' for r in cpt))
(RUN/'sft_schedule.jsonl').write_text(''.join(json.dumps({'step':i,'id':r['row']['id'],'text_sha256':r['row']['text_sha256'],'input_tokens':len(r['input']),'answer_tokens':len(r['answer'])},ensure_ascii=False)+'\n' for i,r in enumerate(sft,1)))
save('CPT_VALIDATION_MANIFEST.json',[{k:v for k,v in r.items() if k!='ids'} for r in val])
status(stage='prepared',cpt_steps=len(cpt),cpt_target_tokens=config['CPT']['target_tokens'],sft_steps_each_arm=72)
print(json.dumps({'prepared':True,'CPT_steps':len(cpt),'CPT_target_tokens':config['CPT']['target_tokens'],'CPT_validation_sequences':len(val),'SFT_max_input_tokens':config['SFT']['max_actual_input_tokens']}),flush=True)
if a.prepare_only:raise SystemExit(0)
assert torch.cuda.is_available()
assert not any((RUN/arm/'training_steps.jsonl').exists() for arm in ['cpt_then_sft','direct_sft']),'Existing run must not be silently restarted.'

def seed():random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def fresh():
 seed();base=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
 seed();model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=MODULES,bias='none'))
 model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();return model
def init_hash(m):
 h=hashlib.sha256()
 for n,t in sorted(m.named_parameters(),key=lambda x:x[0]):
  if t.requires_grad:h.update(n.encode());h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def checkpoint(model,folder,arm,phase,step):
 d=RUN/folder;model.save_pretrained(d);tok.save_pretrained(d);state={'folder':folder,'arm':arm,'phase':phase,'step':step,'sha256':sha(d/'adapter_model.safetensors'),'base_sha256':BASE_SHA};save(folder+'/TRAINING_STATE.json',state);return state
def lm_loss(model,e):
 ids=torch.tensor([e['ids']],device='cuda');y=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=0)
 return F.cross_entropy(y.logits[0,:-1,:].float(),ids[0,1:])
def id_loss(model,e):
 ids=torch.tensor([e['input']+e['answer']],device='cuda');y=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(e['answer'])+1)
 return F.cross_entropy(y.logits[0,:-1,:].float(),torch.tensor(e['answer'],device='cuda'))
def validation(model,label):
 model.eval();num=den=0
 with torch.inference_mode():
  for i,e in enumerate(val,1):
   status(stage='CPT_validation',label=label,completed=i-1,total=len(val));l=lm_loss(model,e);num+=float(l)*e['target_tokens'];den+=e['target_tokens']
 x={'target_token_weighted_loss':num/den,'target_tokens':den,'sequences':len(val),'article_count':8,'same_fixed_sequences_before_after_CPT':True,'not_extraction_accuracy':True};save('cpt_then_sft/'+label+'.json',x);model.train();return x
def train_phase(model,arm,phase,es,lr0,warm):
 seed();model.train();params=[q for q in model.parameters() if q.requires_grad];opt=torch.optim.AdamW(params,lr=lr0,weight_decay=0.);torch.cuda.reset_peak_memory_stats();start=time.monotonic();tokens=0
 for i,e in enumerate(es,1):
  ts=time.monotonic();opt.zero_grad(set_to_none=True);lr=lr0*min(1.,i/warm)*(1.-.5*(i-1)/len(es))
  for g in opt.param_groups:g['lr']=lr
  loss=lm_loss(model,e) if phase=='CPT' else id_loss(model,e);assert torch.isfinite(loss);loss.backward();n=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(n);opt.step();torch.cuda.synchronize();tokens+=e['target_tokens'] if phase=='CPT' else len(e['answer'])
  rec={'arm':arm,'phase':phase,'step':i,'total_steps':len(es),'id':e['id'] if phase=='CPT' else e['row']['id'],'text_sha256':e['text_sha256'] if phase=='CPT' else e['row']['text_sha256'],'loss':float(loss.detach()),'lr':lr,'target_tokens':e['target_tokens'] if phase=='CPT' else len(e['answer']),'input_tokens':len(e['ids']) if phase=='CPT' else len(e['input']),'step_seconds':time.monotonic()-ts,'elapsed_seconds':time.monotonic()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3};log(arm+'/training_steps.jsonl',rec);status(stage='training',**rec);del loss
  if phase=='CPT' and i%256==0:checkpoint(model,f'{arm}/checkpoints/cpt_step_{i:04d}',arm,phase,i)
 result={'complete':True,'arm':arm,'phase':phase,'steps':len(es),'target_tokens':tokens,'training_seconds':time.monotonic()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'trainable_parameters':sum(q.numel() for q in params)};save(arm+'/'+phase+'_TRAINING_SUMMARY.json',result);del opt;gc.collect();torch.cuda.empty_cache();return result

def parse(raw):
 s=raw.strip()
 if s.startswith('```') and s.endswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
 try:
  x=json.loads(s)
  if isinstance(x,dict) and set(x)=={'entity_ids'} and isinstance(x['entity_ids'],list) and all(isinstance(n,str) for n in x['entity_ids']):return list(dict.fromkeys(x['entity_ids'])),True
 except Exception:pass
 return [],False
def infer(model,r):
 prompt=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok(prompt,add_special_tokens=False)['input_ids'];assert len(ids)<=64512;x=torch.tensor([ids],device='cuda');start=time.monotonic()
 with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=4096,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
 new=y[0,len(ids):];raw=tok.decode(new,skip_special_tokens=True).strip();names,valid=parse(raw);g=set(sum(r['known_reference'].values(),[]));q=set(names)
 return {**r,'raw_output':raw,'prediction':{'entity_ids':names},'json_usable':valid,'output_limit_reached':len(new)>=4096 and int(new[-1])!=tok.eos_token_id,'input_tokens':len(ids),'input_truncated':False,'output_tokens':len(new),'seconds':time.monotonic()-start,'known_recovered':len(g&q),'known_reference_count':len(g),'additional_outputs_unreviewed':sorted(q-g),'missing_known_reference':sorted(g-q),'nonliteral_outputs':[n for n in names if n not in r['text']]}
def metrics(rows):
 hit=sum(r['known_recovered'] for r in rows);g=sum(r['known_reference_count'] for r in rows)
 return {'n':len(rows),'json_usable':sum(r['json_usable'] for r in rows),'output_limit_reached':sum(r['output_limit_reached'] for r in rows),'known_recovered':hit,'known_reference_count':g,'known_micro_recall':hit/g if g else None,'mean_per_article_known_recall':statistics.mean([r['known_recovered']/r['known_reference_count'] for r in rows if r['known_reference_count']]),'predicted_ids':sum(len(r['prediction']['entity_ids']) for r in rows),'no_true_full_precision_f1_claim':True}

model=None;initial=None
try:
 for arm in ['cpt_then_sft','direct_sft']:
  (RUN/arm).mkdir(exist_ok=True);status(stage='loading_original_base',arm=arm);model=fresh();h=init_hash(model)
  if initial is None:initial=h
  assert h==initial;save(arm+'/INITIALIZATION.json',{'fresh_common_base_sha256':BASE_SHA,'fresh_LoRA_tensor_sha256':h,'no_prior_task_adapter_loaded':True})
  if arm=='cpt_then_sft':
   validation(model,'before_CPT_validation');train_phase(model,arm,'CPT',cpt,2e-5,32);checkpoint(model,arm+'/cpt_adapter',arm,'CPT_final',len(cpt));validation(model,'after_CPT_validation')
  train_phase(model,arm,'SFT',sft,5e-5,5);checkpoint(model,arm+'/adapter',arm,'SFT_final',72)
  del model;model=None;gc.collect();torch.cuda.empty_cache()
 save('TRAINING_COMPLETION.json',{'complete':True,'fresh_LoRA_tensor_sha256':initial,'identical_SFT_schedule_both_arms':True,'both_SFT_updates':72,'CPT_updates':len(cpt),'CPT_target_tokens':config['CPT']['target_tokens']});status(stage='training_complete_evaluation_pending')
 preds={}
 for arm in ['direct_sft','cpt_then_sft']:
  status(stage='loading_evaluation_model',arm=arm);base=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'});model=PeftModel.from_pretrained(base,RUN/arm/'adapter',local_files_only=True);model.eval();preds[arm]=[]
  for i,r in enumerate(ev):
   status(stage='evaluating_complete_articles',arm=arm,completed=i,total=19,paper=r['paper']);result=infer(model,r);result['arm']=arm;log(arm+'/full_document_predictions.jsonl',result);preds[arm].append(result)
  save(arm+'/FULL_DOCUMENT_METRICS.json',metrics(preds[arm]));del model,base;model=None;gc.collect();torch.cuda.empty_cache()
 bad={r['id'] for rs in preds.values() for r in rs if not r['json_usable'] or r['output_limit_reached']}
 result={'complete':True,'same_original_base_and_fresh_LoRA_initialization':True,'CPT_then_SFT_vs_same_SFT_only':True,'full19':{arm:metrics(rs) for arm,rs in preds.items()},'joint_success':{arm:metrics([r for r in rs if r['id'] not in bad]) for arm,rs in preds.items()},'jointly_excluded_failed_ids':sorted(bad),'old_reference_scope_incomplete_not_blind_new_test':True,'evaluation_not_used_to_select_training_checkpoint':True};save('comparison.json',result);status(stage='complete');print(json.dumps({'complete':True,'comparison':result}),flush=True)
except Exception:
 save('failure.json',{'traceback':traceback.format_exc()})
 if model is not None:
  try:checkpoint(model,'interrupted_adapter',locals().get('arm','unknown'),'interrupted',locals().get('i',0))
  except Exception:pass
 status(stage='failed');raise
