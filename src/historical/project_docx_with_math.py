"""Versioned text projection preserving ordinary and Office Math text.

Does not overwrite the v3 training inputs. This is a plain-text projection,
not an OCR reader or a complete renderer of Word mathematical layout.
"""
import argparse,json,zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+NS['w']+'}';M='{'+NS['m']+'}'

def paragraph_text(p):
    parts=[]
    for node in p.iter():
        if node.tag in (W+'t',M+'t'):parts.append(node.text or '')
        elif node.tag==W+'tab':parts.append('\t')
        elif node.tag in (W+'br',W+'cr'):parts.append('\n')
    return ''.join(parts)

def project_docx(path):
    with zipfile.ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
    blocks=[]
    for node in root.find('w:body',NS):
        if node.tag==W+'p':blocks.append({'type':'paragraph','text':paragraph_text(node)})
        elif node.tag==W+'tbl':
            rows=[['\n'.join(paragraph_text(p) for p in cell.findall('w:p',NS)) for cell in row.findall('w:tc',NS)] for row in node.findall('w:tr',NS)]
            blocks.append({'type':'table','rows':rows})
    return blocks

def as_text(blocks):
    lines=[]
    for i,b in enumerate(blocks):
        if b['type']=='paragraph':lines.append(b['text'])
        else:lines.append(f'[TABLE block={i}]');lines.extend(' | '.join(row) for row in b['rows'])
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--docx',required=True);p.add_argument('--output-directory',required=True);a=p.parse_args()
    out=Path(a.output_directory);out.mkdir(parents=True,exist_ok=False);blocks=project_docx(a.docx)
    (out/'full.txt').write_text(as_text(blocks),encoding='utf-8')
    (out/'blocks.json').write_text(json.dumps({'source_docx':str(Path(a.docx).resolve()),'version':2,'projection':'w:t plus m:t in body order; plain text math, no image OCR','blocks':blocks},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
