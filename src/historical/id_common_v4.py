import json, re

SYSTEM_V3 = '''Extract geological study sample identifiers and individual grain/spot/analysis identifiers explicitly present in the complete document, including its tables. Copy each identifier exactly, deduplicate each list in order of first appearance. Use context and table headers to distinguish the two types. Do not infer a sample identifier from a spot prefix. Exclude ages, isotope ratios, instruments, reference numbers, figures, tables and analytical standards. Return only {"sample_ids": [...], "spot_ids": [...]}; use an empty list when a type is absent. 全文联合识别：样品编号放入sample_ids，颗粒、测点或单次分析编号放入spot_ids。利用正文和表格上下文区分，保留原文拼写，按首次出现去重。不得从测点前缀推断未出现的样品编号。排除年龄、仪器、引文、图表号和分析标样。仅输出这两个字段的JSON对象，不解释，不输出年龄或父子关系。'''

def matches(text, ids):
    found=[]
    for x in ids:
        pat=re.compile(r'(?<![A-Za-z0-9_.–—\-])'+re.escape(x)+r'(?![A-Za-z0-9_–—\-]|\.\d)')
        for m in pat.finditer(text):
            if re.fullmatch(r'[\d.]+',x):
                end=text.find('\n',m.end()); end=len(text) if end<0 else end
                line=text[text.rfind('\n',0,m.start())+1:end]
                if not line.lstrip().startswith(x+'\t') and not re.search(r'sample|grain|spot|样品|测点|颗粒',text[max(0,m.start()-35):m.start()],re.I):continue
            found.append((m.start(),x))
    found.sort();return list(dict.fromkeys(x for _,x in found))

def parse_output(raw):
    try:
        obj=json.loads(raw)
        assert isinstance(obj,dict) and set(obj)=={'sample_ids','spot_ids'}
        assert all(isinstance(obj[k],list) and all(isinstance(x,str) for x in obj[k]) for k in obj)
        return obj,True
    except (ValueError,TypeError,AssertionError):return {'sample_ids':[],'spot_ids':[]},False

def read_jsonl(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def write_jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

SYSTEM = SYSTEM_V3 + '\nPrefer the complete identifier literally written in this document when a grain is also referred to with an abbreviated number; do not repeat the abbreviation if its full identifier is available. Never construct a full identifier that is not written. Headers Sample No., Sample numbers and 样品编号 identify samples, even when their names contain several hyphens. Exclude geographic coordinates, figure-panel letters, bibliographic years, counts, heating durations and instrument IDs. A number is a spot identifier only when the text explicitly names it as a point/grain identifier or places it in an analysis identifier column; proximity to the word grain/sample is insufficient. 本文有完整测点编号时，优先原文完整写法；只有简写时保留简写，不拼接未出现的前缀。样品表头下的编号仍是样品，多段连字符不能作为测点判据。排除坐标、图中字母、引文年份、数量、加热时间、仪器编号。数值只有位于编号字段或明确的测点编号列表时才算编号。'
