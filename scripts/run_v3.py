"""Resumable CPU workflow. Run with pythonw.exe on Windows to survive console closure."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs/v3'; LOG.mkdir(parents=True,exist_ok=True)
PY=str(Path(sys.executable).with_name('python.exe')) if os.name=='nt' else sys.executable
STAGES=[('data',['scripts/prepare_v3.py']),
        ('features',['scripts/extract_v3.py','--roles','train,cal,dev']),
        ('train',['scripts/train_v3.py']),
        ('sealed_features',['scripts/extract_v3.py','--roles','test']),
        ('sealed_eval',['scripts/eval_v3.py'])]


def mark(**data):
    data['updated_at']=datetime.now(timezone.utc).isoformat()
    tmp=LOG/'state.tmp.json'; tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(LOG/'state.json')


def main():
    completed=[]
    if (LOG/'state.json').exists():
        completed=json.loads((LOG/'state.json').read_text(encoding='utf-8')).get('completed',[])
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',
             TOKENIZERS_PARALLELISM='false',HF_HUB_DOWNLOAD_TIMEOUT='60',HF_HUB_ETAG_TIMEOUT='30')
    flags=0x00000008|0x00000200|0x08000000 if os.name=='nt' else 0
    for name,args in STAGES:
        if name in completed: continue
        for attempt in range(1,4):
            mark(stage=name,status='running',completed=completed,attempt=attempt)
            with (LOG/f'{name}.log').open('ab') as output:
                proc=subprocess.Popen([PY,'-X','utf8','-u',*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                      stdout=output,stderr=subprocess.STDOUT,creationflags=flags)
                rc=proc.wait()
            if rc==0: break
            if attempt==3:
                mark(stage=name,status='failed',return_code=rc,completed=completed)
                return
            time.sleep(10)
        completed.append(name)
        mark(stage=name,status='complete',completed=completed)
    mark(stage='all',status='complete',completed=completed)


if __name__=='__main__':
    main()
