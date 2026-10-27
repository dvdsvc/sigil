"""Original strict parser and conservative prefix recovery; no publication code."""
import json
import re

def parse(raw):
 s=raw.strip()
 if s.startswith('```') and s.endswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
 try:
  x=json.loads(s)
  if isinstance(x,dict) and set(x)=={'entity_ids'} and isinstance(x['entity_ids'],list) and all(isinstance(n,str) for n in x['entity_ids']):return list(dict.fromkeys(x['entity_ids'])),True
 except Exception:pass
 return [],False

def prefix(raw):
 m=re.match(r'^\s*\{\s*"entity_ids"\s*:\s*\[',raw)
 if not m:return []
 dec=json.JSONDecoder();pos=m.end();v=[]
 while pos<len(raw):
  while pos<len(raw) and raw[pos].isspace():pos+=1
  try:x,end=dec.raw_decode(raw,pos)
  except ValueError:break
  if not isinstance(x,str):break
  v.append(x);pos=end
  while pos<len(raw) and raw[pos].isspace():pos+=1
  if pos<len(raw) and raw[pos]==',':pos+=1
  else:break
 return v
