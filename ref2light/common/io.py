from __future__ import annotations
import json, os
import datetime
from pathlib import Path
import torch

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write_json(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

def append_jsonl(p,d):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')

def atomic_save(p,d):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');torch.save(d,tmp);os.replace(tmp,p)





def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
