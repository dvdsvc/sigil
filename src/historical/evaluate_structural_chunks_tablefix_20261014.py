"""Lossless structure-aware full-corpus evaluation; no ID-driven chunk selection."""
import argparse, json, hashlib, re, time, statistics, traceback, gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--prepare-only', action='store_true')
p.add_argument('--run-name', default='B_v21_structure_chunks_20261014_02')
a = p.parse_args()
ROOT = a.workspace
OLD = ROOT / 'runs/B_v21_union_20261014_01'
OUT = ROOT / 'runs' / a.run_name
OUT.mkdir(exist_ok=True)
BASE = ROOT / 'models/Qwen3.5-0.8B'
tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
cfg = json.loads((OLD / 'config.json').read_text())
SYSTEM = cfg['system']
TARGET, MAXBODY = 2048, 3072

def save(name, x):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

def count(s):
    return len(tok(s, add_special_tokens=False)['input_ids'])

def heading(s):
    x = s.strip()
    if not x or len(x) > 140 or x.count('|') > 1:
        return False
    if re.match(r'^#{1,6}\s', x):
        return True
    if re.match(r'^(?:\d{1,2}(?:\.\d{1,2}){0,3})[.、\s]+\D', x) and len(x) < 85:
        return True
    if re.match(r'^(?:第[一二三四五六七八九十0-9]+[章节]|摘要|参考文献|致谢|结论|引言|Abstract|Introduction|References|Acknowledg|Conclusions|Discussion|Results|Methods|Materials)\b', x, re.I):
        return True
    return len(x) <= 65 and len(re.findall('[A-Z]', x)) > 6 and x.upper() == x and not re.search(r'[.。;；!?！？]', x)

def tableline(s):
    return s.count('|') >= 3

def atomic_units(text):
    """Contiguous original offsets: parsed lines are paragraphs; pipe rows form tables."""
    lines = []
    pos = 0
    for line in text.splitlines(keepends=True):
        lines.append((pos, pos + len(line), line))
        pos += len(line)
    units, i = [], 0
    while i < len(lines):
        st, en, s = lines[i]
        if tableline(s) or re.match(r'^\s*\[TABLE\b', s):
            # Parsed tables may insert a merged sample-description row with no pipes.
            # Keep captions, marker, header, description and data rows in one block.
            while units:
                prior = text[units[-1]['start']:units[-1]['end']].strip()
                if re.match(r'^(?:Table|TABLE|表)\s*\d', prior):
                    st = units.pop()['start']
                else:
                    break
            j = i + 1
            while j < len(lines):
                line = lines[j][2]
                if tableline(line) or re.match(r'^\s*\[TABLE\b',line):
                    j += 1
                    continue
                # Preserve an intervening merged row when followed by table data.
                if len(line) <= 500 and j+1 < len(lines) and tableline(lines[j+1][2]):
                    j += 1
                    continue
                break
            units.append({'start': st, 'end': lines[j-1][1], 'kind': 'table'})
            i = j
        else:
            units.append({'start': st, 'end': en, 'kind': 'heading' if heading(s) else 'paragraph'})
            i += 1
    assert ''.join(text[u['start']:u['end']] for u in units) == text
    return units

def split_long(u, text):
    body = text[u['start']:u['end']]
    if count(body) <= MAXBODY:
        return [u]
    if u['kind'] == 'table':
        pieces = body.splitlines(keepends=True)
    else:
        # Only oversize paragraphs are split, at actual sentence boundaries.
        pieces = re.split(r'(?<=[。！？!?])|(?<=\.)\s+(?=[A-Z])', body)
        # The latter separator consumes spaces; reconstruct by offsets instead.
        cuts = [m.end() for m in re.finditer(r'[。！？!?]|\.\s+(?=[A-Z])', body)]
        cuts = sorted(set([0] + cuts + [len(body)]))
        pieces = [body[x:y] for x,y in zip(cuts,cuts[1:])]
    output, start, end = [], u['start'], u['start']
    for piece in pieces:
        if end > start and count(text[start:end] + piece) > TARGET:
            output.append({'start':start,'end':end,'kind':u['kind']+'_part'})
            start = end
        # A corrupted giant sentence/row has no useful semantic boundary.
        # Preserve all characters with explicit emergency whitespace/character cuts.
        if count(piece) > MAXBODY:
            if end > start:
                output.append({'start':start,'end':end,'kind':u['kind']+'_part'})
            base = end
            while base < end + len(piece):
                lo, hi = base+1, end+len(piece)
                while lo < hi:
                    mid = (lo+hi+1)//2
                    if count(text[base:mid]) <= TARGET: lo=mid
                    else: hi=mid-1
                output.append({'start':base,'end':lo,'kind':u['kind']+'_emergency'})
                base=lo
            end += len(piece)
            start=end
        else:
            end += len(piece)
    if end > start: output.append({'start':start,'end':end,'kind':u['kind']+'_part'})
    assert ''.join(text[x['start']:x['end']] for x in output) == body
    return output

def segment(r):
    text = r['text']
    units = atomic_units(text)
    expanded, section = [], ''
    for u in units:
        if u['kind']=='heading': section=text[u['start']:u['end']].strip()
        table_context=''
        if u['kind']=='table':
            # Repeat only literal caption/header material when a large table splits.
            header=''
            for line in text[u['start']:u['end']].splitlines(keepends=True)[:6]:
                if count(header+line)>512:break
                header+=line
            table_context=header
        for v in split_long(u,text):
            expanded.append({**v,'section':section,'table_context':table_context if v['start']>u['start'] else ''})
    chunks, current = [], []
    def flush():
        if not current:return
        st,en=current[0]['start'],current[-1]['end']
        section=current[0]['section'];prefix=''
        if section and not text[st:en].lstrip().startswith(section): prefix=section+'\n'
        tc=current[0]['table_context']
        if tc: prefix += tc
        # No extra summaries or invented context; body is an exact source slice.
        payload=prefix+text[st:en]
        chunks.append({'id':r['id']+f'_CHUNK_{len(chunks):04d}','article_id':r['id'],'paper':r['paper'],'start_char':st,'end_char':en,'body':text[st:en],'context_prefix':prefix,'text':payload,'body_sha256':hashlib.sha256(text[st:en].encode()).hexdigest(),'kinds':[x['kind'] for x in current],'section_heading':section,'body_tokens':count(text[st:en]),'text_tokens':count(payload)})
        current.clear()
    for u in expanded:
        different_section=current and u['section']!=current[0]['section']
        table_change=current and (u['kind'].startswith('table') != current[-1]['kind'].startswith('table'))
        too_big=current and count(text[current[0]['start']:u['end']])>TARGET
        if different_section or table_change or too_big:flush()
        current.append(u)
    flush()
    assert ''.join(c['body'] for c in chunks)==text
    assert all(c['body_tokens']<=MAXBODY for c in chunks)
    assert all(x['end_char']==y['start_char'] for x,y in zip(chunks,chunks[1:]))
    return chunks

manifest=json.loads((OLD/'evaluation/CASE_MANIFEST.json').read_text())
ids=set(manifest['full_ids'])
articles=[r for r in read(ROOT/'outputs/full_document_evaluation_correction/data/new_full_documents.jsonl') if r['id'] in ids]
assert len(articles)==19
chunks=sum((segment(r) for r in articles),[])
frozen={'article_count':19,'chunk_count':len(chunks),'article_ids':[r['id'] for r in articles],'target_body_tokens':TARGET,'max_body_tokens':MAXBODY,'median_body_tokens':statistics.median(c['body_tokens'] for c in chunks),'actual_max_body_tokens':max(c['body_tokens'] for c in chunks),'all_article_characters_covered_exactly_once_in_bodies':True,'all_sections_including_references_and_unrelated_text_included':True,'known_reference_not_used_to_choose_chunks':True,'no_training':True,'same_prompt_as_whole_document_eval':True,'output_budget_per_chunk':4096,'extra_inference_calls_and_total_output_budget_are_not_equal_to_whole_article':True,'semantic_boundaries':'Existing headings, paragraph lines and table rows; oversized paragraphs split at sentence endings, emergency cuts explicitly recorded. Heuristic structure, not perfect semantic parsing.','literal_heading_and_large_table_headers_repeated_as_context':True,'table_boundary_fix':'Captions, TABLE marker, column headers and merged sample-description rows stay with data; first up to six literal lines are repeated for long-table parts, capped at512 tokens.','emergency_cut_units':sum(sum('emergency' in k for k in c['kinds']) for c in chunks),'base_sha256':cfg['base_weight_sha256'],'adapter_sha256':json.loads((OLD/'training_summary.json').read_text())['final_adapter_sha256'],'system':SYSTEM,'do_sample':False,'enable_thinking':False}
save('CASE_MANIFEST.json',frozen)
(OUT/'chunks.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in chunks))
save('status.json',{'stage':'prepared','chunks':len(chunks),'articles':19})
print(json.dumps({'prepared':True,'chunks':len(chunks),'median_body_tokens':frozen['median_body_tokens'],'max_body_tokens':frozen['actual_max_body_tokens'],'emergency_cut_units':frozen['emergency_cut_units']}),flush=True)
if a.prepare_only:raise SystemExit(0)

def parse(raw):
    s=raw.strip()
    if s.startswith('```') and s.endswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
    try:
        x=json.loads(s)
        if isinstance(x,dict) and set(x)=={'entity_ids'} and isinstance(x['entity_ids'],list) and all(isinstance(n,str) for n in x['entity_ids']):return list(dict.fromkeys(x['entity_ids'])),True
        if isinstance(x,dict) and set(x)=={'sample_ids','spot_ids'} and all(isinstance(v,list) and all(isinstance(n,str) for n in v) for v in x.values()):return list(dict.fromkeys(x['sample_ids']+x['spot_ids'])),True
    except (ValueError,IndexError):pass
    return [],False

def summarize(rs):
    g=sum(r['known_reference_count'] for r in rs);hit=sum(r['known_recovered'] for r in rs)
    return {'n':len(rs),'known_recovered':hit,'known_reference_count':g,'known_micro_recall':hit/g if g else None,'mean_per_article_known_recall':statistics.mean([r['known_recovered']/r['known_reference_count'] for r in rs if r['known_reference_count']]),'predicted_ids':sum(len(r['prediction']['entity_ids']) for r in rs),'papers_with_invalid_or_limited_chunks':sum(not r['all_chunks_usable_and_unlimited'] for r in rs)}

try:
    completed=read(OUT/'chunk_predictions.jsonl') if (OUT/'chunk_predictions.jsonl').exists() else []
    assert [r['id'] for r in completed]==[c['id'] for c in chunks[:len(completed)]]
    adapter=OLD/'adapter'
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==frozen['adapter_sha256']
    cache_path=ROOT/'runs/B_v21_structure_chunks_20261014_01/chunk_predictions.jsonl'
    cache_rows=read(cache_path) if cache_path.exists() else []
    cache_manifest=json.loads((cache_path.parent/'CASE_MANIFEST.json').read_text()) if cache_rows else {}
    assert not cache_rows or (cache_manifest['adapter_sha256']==frozen['adapter_sha256'] and cache_manifest['system']==SYSTEM and cache_manifest['output_budget_per_chunk']==4096)
    cache={r['text']:r for r in cache_rows if r['json_usable'] and not r['output_limit_reached']}
    frozen['reused_identical_input_count_available']=sum(c['text'] in cache for c in chunks)
    save('CASE_MANIFEST.json',frozen)
    save('status.json',{'stage':'loading_model','completed':len(completed),'total':len(chunks)})
    model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
    model=PeftModel.from_pretrained(model,adapter,local_files_only=True);model.eval()
    for i,c in enumerate(chunks[len(completed):],len(completed)):
        save('status.json',{'stage':'evaluating','completed':i,'total':len(chunks),'paper':c['paper'],'chunk_id':c['id']})
        prompt=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':c['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inp=tok(prompt,add_special_tokens=False)['input_ids'];assert len(inp)<=4096
        if c['text'] in cache:
            old=cache[c['text']]
            assert old['input_tokens']==len(inp)
            rec={**c,**{k:old[k] for k in ['raw_output','prediction','json_usable','output_limit_reached','input_tokens','output_tokens','nonliteral_outputs']},'seconds':0.0,'cached_original_inference_seconds':old['seconds'],'reused_identical_input_from_chunk':old['id']}
            with (OUT/'chunk_predictions.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            completed.append(rec)
            print(json.dumps({'completed':i+1,'total':len(chunks),'reused_identical_input':True}),flush=True)
            continue
        x=torch.tensor([inp],device='cuda');start=time.monotonic()
        with torch.inference_mode(): y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=4096,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
        new=y[0,len(inp):];raw=tok.decode(new,skip_special_tokens=True).strip();names,valid=parse(raw);limit=len(new)>=4096 and int(new[-1])!=tok.eos_token_id
        rec={**c,'raw_output':raw,'prediction':{'entity_ids':names},'json_usable':valid,'output_limit_reached':limit,'input_tokens':len(inp),'output_tokens':len(new),'seconds':time.monotonic()-start,'nonliteral_outputs':[n for n in names if n not in c['text']]}
        with (OUT/'chunk_predictions.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        completed.append(rec)
        print(json.dumps({'completed':i+1,'total':len(chunks),'paper':c['paper'],'json_usable':valid,'output_limit_reached':limit,'predicted_ids':len(names),'seconds':rec['seconds']},ensure_ascii=False),flush=True)
    merged=[]
    whole=read(OLD/'evaluation/B_v21_union/full_documents_predictions.jsonl')
    parent=read(OLD/'evaluation/B_parent/full_documents_predictions.jsonl')
    oldbad={r['id'] for r in whole+parent if not r['json_usable'] or r['output_limit_reached']}
    currentwholebad={r['id'] for r in whole if not r['json_usable'] or r['output_limit_reached']}
    for arow in articles:
        cr=[r for r in completed if r['article_id']==arow['id']];names=list(dict.fromkeys(n for r in cr for n in r['prediction']['entity_ids']));gold=set(sum(arow['known_reference'].values(),[]));pred=set(names)
        merged.append({**arow,'prediction':{'entity_ids':names},'known_recovered':len(gold&pred),'known_reference_count':len(gold),'missing_known_reference':sorted(gold-pred),'additional_outputs_unreviewed':sorted(pred-gold),'chunks':len(cr),'all_chunks_usable_and_unlimited':all(r['json_usable'] and not r['output_limit_reached'] for r in cr),'invalid_chunks':sum(not r['json_usable'] for r in cr),'limited_chunks':sum(r['output_limit_reached'] for r in cr),'seconds':sum(r['seconds'] for r in cr)})
    (OUT/'article_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
    # Convert whole-document output only for common summary fields.
    for r in whole:r['all_chunks_usable_and_unlimited']=r['json_usable'] and not r['output_limit_reached']
    newbad={r['id'] for r in merged if not r['all_chunks_usable_and_unlimited']};joint=currentwholebad|newbad
    result={'complete':True,'same_final_adapter':True,'full19':{'whole':summarize(whole),'structured_chunks':summarize(merged)},'fixed_prior_common17':{'whole':summarize([r for r in whole if r['id'] not in oldbad]),'structured_chunks':summarize([r for r in merged if r['id'] not in oldbad])},'joint_success_only':{'whole':summarize([r for r in whole if r['id'] not in joint]),'structured_chunks':summarize([r for r in merged if r['id'] not in joint])},'failed_article_ids':sorted(joint),'chunk_json_usable':sum(r['json_usable'] for r in completed),'chunk_output_limit_reached':sum(r['output_limit_reached'] for r in completed),'chunk_count':len(completed),'evaluation_seconds':sum(r['seconds'] for r in completed),'identical_inputs_reused':sum('reused_identical_input_from_chunk' in r for r in completed),'reused_inputs_original_inference_seconds':sum(r.get('cached_original_inference_seconds',0) for r in completed),'known_references_incomplete_no_true_precision_f1':True,'extra_calls_and_output_budget_confounded_with_chunking':True,'data_not_used_for_training':True}
    save('comparison.json',result);save('status.json',{'stage':'complete','chunks':len(chunks),'articles':19});print(json.dumps({'complete':True,'comparison':result}),flush=True)
except Exception:
    save('failure.json',{'traceback':traceback.format_exc()});save('status.json',{'stage':'failed'});raise
