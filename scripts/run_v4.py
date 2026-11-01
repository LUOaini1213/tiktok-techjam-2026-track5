"""Checkpointed serial CPU workflow. Nonzero child status stops all later stages."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.v4_common import REPORT,write_json,read_json,preserve_v3


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepared',action='store_true'); a=p.parse_args()
    preserve_v3(); logs=ROOT/'logs/v4'; logs.mkdir(exist_ok=True,parents=True)
    stages=[] if a.prepared else [('acquire',['scripts/prepare_v4.py'])]
    stages += [('extract',['scripts/extract_v4.py']),('fit_lock',['scripts/train_v4.py']),
               ('sealed_features',['scripts/extract_v4.py','--roles','test']),
               ('evaluate',['scripts/eval_v4.py']),('preflight',['scripts/verify_v4.py']),
               ('pytest',['-m','pytest','tests/','-q'])]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
    completed=[]
    for name,args in stages:
        if a.prepared and name=='extract' and not read_json(ROOT/'data/v4/manifest.json')['complete']:
            raise RuntimeError('Acquisition must be complete before --prepared run')
        state=dict(stage=name,completed=completed,status='running',updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(REPORT/'run_state.json',state); print(f'STAGE {name}',flush=True)
        child_env=dict(env)
        if name!='acquire': child_env['HF_HUB_OFFLINE']='1'
        with (logs/f'{name}.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-X','utf8','-u',*args],cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            state.update(status='failed',exit_code=proc.returncode); write_json(REPORT/'run_state.json',state)
            raise RuntimeError(f'{name} failed ({proc.returncode}); see logs/v4/{name}.log')
        completed.append(name)
    write_json(REPORT/'run_state.json',dict(status='complete',completed=completed,updated_utc=datetime.now(timezone.utc).isoformat()))
    preserve_v3(); print('All V4 local stages complete; Git push is a separate verified release step',flush=True)


if __name__=='__main__':
    # Process-scoped only: long CPU work should survive automatic screen-off.
    # Explicit sleep/lid/user power actions still take precedence; no power plan is changed.
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    finally:
        if sys.platform=='win32': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
