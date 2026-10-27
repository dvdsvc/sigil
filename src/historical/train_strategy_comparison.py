"""Paired continuing-SFT: union then typed vs typed then typed, same input order."""
import argparse, collections, gc, hashlib, json, math, random, sys, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

p=argparse.ArgumentParser()
p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--run-name',required=True)
args=p.parse_args()
ROOT=args.workspace;DATA=args.data;RUN=ROOT/'runs'/args.run_name
sys.path.insert(0,str(ROOT/'src'))
from id_common_v4 import SYSTEM,parse_output,read_jsonl,write_jsonl
UNION_SYSTEM='''Extract geological study sample identifiers and individual grain/spot/analysis identifiers explicitly present in the input, including tables. Copy names exactly and return their deduplicated union in order of first appearance. Do not classify the names. Do not infer a sample name from a spot prefix. Exclude database record indices, ages, isotope ratios, instruments, references, figures, table numbers, coordinates, counts and analytical standards. Return only {"entity_ids": [...]}; use an empty list when none are present. 识别输入中实际出现的样品、颗粒、测点或分析名称，合并为一个列表，不分类。原样复制，按首次出现去重。不推断未出现的名称，不自动续号。排除数据库记录索引、年龄、比值、仪器、引文、坐标、数量和标样。仅输出entity_ids字段的JSON对象，不解释。'''
SEED=20261013
INITIAL=ROOT/'adapters/v4'
MODEL=ROOT/'models/Qwen3.5-0.8B'
EXPECTED='3e44212de831c06793e70c3cda98cefccc97381f2ae4fbe7f33c9946e14c5587'
assert torch.cuda.is_available()
assert hashlib.sha256((INITIAL/'adapter_model.safetensors').read_bytes()).hexdigest()==EXPECTED
RUN.mkdir(parents=True,exist_ok=False)
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def status(**kw):save(RUN/'status.json',kw)
def log(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    print(json.dumps(obj,ensure_ascii=False),flush=True)
tok=AutoTokenizer.from_pretrained(INITIAL,local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
def prefix(text,system):return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def encode(row,kind):
    system=UNION_SYSTEM if kind=='union' else SYSTEM
    labels={'entity_ids':list(dict.fromkeys(row['labels']['sample_ids']+row['labels']['spot_ids']))} if kind=='union' else row['labels']
    if kind=='union':labels['entity_ids'].sort(key=lambda s:row['text'].find(s))
    a=tok(prefix(row['text'],system),add_special_tokens=False)['input_ids']
    b=tok(json.dumps(labels,ensure_ascii=False)+tok.eos_token,add_special_tokens=False)['input_ids']
    assert len(a)<=64512 and len(b)<=4096,(row['id'],len(a),len(b))
    return {'row':row,'input':a,'answer':b,'kind':kind}
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
def metrics(rows):
    res={'n':len(rows),'json_valid':sum(r['json_valid'] for r in rows),'output_limit_reached':sum(r['output_limit_reached'] for r in rows),
         'exact_match':sum(r['json_valid'] and all(set(r['prediction'][k])==set(r['labels'][k]) for k in r['labels']) for r in rows),
         'nonliteral_predictions':sum(len(r['nonliteral_predictions']) for r in rows),
         'empty_reference_cases':sum(not any(r['labels'].values()) for r in rows),
         'correct_empty_cases':sum(not any(r['labels'].values()) and r['json_valid'] and not any(r['prediction'].values()) for r in rows),
         'cross_list_ids':sum(len(set(r['prediction']['sample_ids'])&set(r['prediction']['spot_ids'])) for r in rows)}
    for kind in ['sample_ids','spot_ids','typed_all','untyped_union']:
        tp=fp=fn=0
        for r in rows:
            if kind=='typed_all':
                g={(k,x) for k in r['labels'] for x in r['labels'][k]};q={(k,x) for k in r['prediction'] for x in r['prediction'][k]}
            elif kind=='untyped_union':g=set(sum(r['labels'].values(),[]));q=set(sum(r['prediction'].values(),[]))
            else:g=set(r['labels'][kind]);q=set(r['prediction'][kind])
            tp+=len(g&q);fp+=len(q-g);fn+=len(g-q)
        res[kind]=prf(tp,fp,fn)
    recovered=correct=0
    for r in rows:
        g=set(sum(r['labels'].values(),[]));q=set(sum(r['prediction'].values(),[]))
        for name in g&q:
            recovered+=1
            correct+=({k for k in r['labels'] if name in r['labels'][k]}=={k for k in r['prediction'] if name in r['prediction'][k]})
    res['type_accuracy_on_recovered_reference_ids']={'correct':correct,'recovered':recovered,'accuracy':correct/recovered if recovered else None}
    return res

train=read_jsonl(DATA/'train.jsonl')
eval_sets={name:read_jsonl(DATA/(name+'.jsonl')) for name in ['diagnostic','heldout_tables','full_documents','full_negatives']}
assert len(train)==88
assert not {r['group_id'] for r in train}&{r['group_id'] for rs in eval_sets.values() for r in rs}
assert all(hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256'] for r in train)
encoded={kind:{r['id']:encode(r,kind) for r in train} for kind in ['union','typed']}
orders=[]
for phase in [0,1]:
    order=list(train);random.Random(SEED+phase).shuffle(order);orders.append(order)
config={'seed':SEED,'initial_adapter_sha256':EXPECTED,'initial_adapter':str(INITIAL),'model':str(MODEL),
        'system_typed':SYSTEM,'system_union':UNION_SYSTEM,'strategy_A':['union','typed'],'strategy_B':['typed','typed'],
        'updates_per_phase':len(train),'updates_per_arm':len(train)*2,'learning_rate':7e-5,'warmup_steps':5,
        'gradient_accumulation':1,'micro_batch':1,'precision':'bf16','optimizer':'AdamW weight_decay=0',
        'same_order_per_phase':True,'same_final_inference_prompt':True,'same_generation':'greedy, no thinking, no retries',
        'final_output_budget':4096,'diagnostic_output_budget':512,'input_truncation':False,
        'checkpoint_selection':'predeclared final checkpoint only','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.jsonl')}}
save(RUN/'config.json',config)
write_jsonl(RUN/'scheduled_inputs.jsonl',[{'phase':i+1,'index':j+1,'id':r['id'],'group_id':r['group_id'],'paper':r['paper']} for i,o in enumerate(orders) for j,r in enumerate(o)])
def loss_for(model,e):
    ids=torch.tensor([e['input']+e['answer']],device='cuda')
    out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(e['answer'])+1)
    return F.cross_entropy(out.logits[0,:-1,:].float(),torch.tensor(e['answer'],device='cuda'))
def infer(model,row,budget):
    x=tok(prefix(row['text'],SYSTEM),add_special_tokens=False,return_tensors='pt').to('cuda')
    n=x['input_ids'].shape[-1];assert n<=64512
    torch.cuda.synchronize();start=time.monotonic()
    with torch.inference_mode():y=model.generate(**x,max_new_tokens=budget,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
    new=y[0,n:];raw=tok.decode(new,skip_special_tokens=True).strip();pred,valid=parse_output(raw)
    return {**row,'raw_output':raw,'prediction':pred,'json_valid':valid,'input_tokens':n,'output_tokens':len(new),
            'output_budget':budget,'input_truncated':False,
            'output_limit_reached':len(new)>=budget and int(new[-1])!=tok.eos_token_id,
            'inference_seconds':time.monotonic()-start,
            'nonliteral_predictions':[{'type':k,'id':v} for k,vs in pred.items() for v in vs if v not in row['text']]}

try:
    all_summaries={}
    for arm,kinds in [('A_staged',['union','typed']),('B_direct',['typed','typed'])]:
        target=RUN/arm;target.mkdir()
        random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        status(arm=arm,stage='loading_model')
        model=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
        model=PeftModel.from_pretrained(model,INITIAL,local_files_only=True,is_trainable=True)
        model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
        params=[q for q in model.parameters() if q.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=7e-5,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();step=0
        in_tokens=out_tokens=0;steps=[]
        for phase,kind in enumerate(kinds):
            for row in orders[phase]:
                step+=1;e=encoded[kind][row['id']]
                optimizer.zero_grad(set_to_none=True)
                lr=7e-5*min(1.,step/5)*(1.-.5*(step-1)/(len(train)*2))
                for g in optimizer.param_groups:g['lr']=lr
                loss=loss_for(model,e);assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
                optimizer.step();torch.cuda.synchronize()
                in_tokens+=len(e['input']);out_tokens+=len(e['answer'])
                rec={'arm':arm,'phase':phase+1,'kind':kind,'step':step,'total_steps':len(train)*2,'id':row['id'],
                     'input_tokens':len(e['input']),'answer_tokens':len(e['answer']),'loss':float(loss.detach()),'lr':lr,
                     'elapsed_seconds':time.monotonic()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
                steps.append(rec);log(target/'training_steps.jsonl',rec);status(stage='training',**rec);del loss
            if phase==0:
                model.save_pretrained(target/'phase1_adapter');tok.save_pretrained(target/'phase1_adapter')
        seconds=time.monotonic()-start
        model.save_pretrained(target/'adapter');tok.save_pretrained(target/'adapter')
        training={'complete':True,'arm':arm,'steps':step,'training_seconds_including_phase1_save_and_first_shape_compile':seconds,
                  'peak_memory_gib':max(r['peak_memory_gib'] for r in steps),'input_tokens':in_tokens,'answer_tokens':out_tokens,
                  'trainable_parameters':sum(q.numel() for q in params),'gpu':torch.cuda.get_device_name(0),
                  'checkpoint_sha256':hashlib.sha256((target/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
                  'phase1_checkpoint_sha256':hashlib.sha256((target/'phase1_adapter/adapter_model.safetensors').read_bytes()).hexdigest()}
        save(target/'training_summary.json',training)
        model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
        arm_metrics={};reload_examples=[]
        for name,rows in eval_sets.items():
            predictions=[]
            for i,row in enumerate(rows):
                status(arm=arm,stage='evaluation',dataset=name,completed=i,total=len(rows),paper=row['paper'])
                item=infer(model,row,512 if name=='diagnostic' else 4096)
                if name=='full_documents':
                    item['reference_comparison']={k:{'reference_count':len(set(row['labels'][k])),
                        'recovered_count':len(set(row['labels'][k])&set(item['prediction'][k])),
                        'missing_reference':sorted(set(row['labels'][k])-set(item['prediction'][k])),
                        'unreferenced_outputs':sorted(set(item['prediction'][k])-set(row['labels'][k]))} for k in row['labels']}
                with (target/(name+'_predictions.jsonl')).open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                predictions.append(item)
                if i%10==0 or name=='full_documents':print(json.dumps({'arm':arm,'dataset':name,'completed':i+1,'total':len(rows),'paper':row['paper'],'json_valid':item['json_valid'],'output_tokens':item['output_tokens'],'seconds':item['inference_seconds']},ensure_ascii=False),flush=True)
            if name=='full_documents':
                m={'n':len(rows),'json_valid':sum(r['json_valid'] for r in predictions),
                   'scope':'Incomplete known references: report known-ID recovery and unreferenced candidates, no aggregate full precision/F1.',
                   'rows':[{k:r[k] for k in ['paper','json_valid','input_tokens','output_tokens','output_limit_reached','inference_seconds','reference_comparison','nonliteral_predictions']} for r in predictions]}
            else:
                m=metrics(predictions)
                m['scope']='Existing development fragments with weak labels, not a new blind benchmark.' if name=='diagnostic' else 'Database-column supervision on source-held-out rendered tables, two variants per source/method.' if name=='heldout_tables' else 'Two source-reviewed full real negative articles.'
                if name=='heldout_tables':m['by_variant']={v:metrics([r for r in predictions if r['variant']==v]) for v in ['clean','noisy']}
            save(target/(name+'_metrics.json'),m);arm_metrics[name]=m
            if name in ['diagnostic','heldout_tables']:reload_examples+=predictions[:2]
        del model,optimizer,params;gc.collect();torch.cuda.empty_cache()
        status(arm=arm,stage='reload_verification')
        model=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
        model=PeftModel.from_pretrained(model,target/'adapter',local_files_only=True);model.eval()
        checks=[{'id':r['id'],'same_raw':infer(model,r,r['output_budget'])['raw_output']==r['raw_output']} for r in reload_examples]
        save(target/'reload_verification.json',{'all_consistent':all(c['same_raw'] for c in checks),'checks':checks});assert all(c['same_raw'] for c in checks)
        all_summaries[arm]={'training':training,'evaluation':arm_metrics,'reload_consistent':True}
        save(target/'summary.json',all_summaries[arm]);del model;gc.collect();torch.cuda.empty_cache()
        save(RUN/'comparison_progress.json',all_summaries)
    save(RUN/'comparison.json',{'complete':True,'arms':all_summaries,'config':config})
    status(stage='complete',arms=list(all_summaries));print(json.dumps({'complete':True,'run':str(RUN)}),flush=True)
except Exception:
    save(RUN/'failure.json',{'traceback':traceback.format_exc()});status(stage='failed');raise
