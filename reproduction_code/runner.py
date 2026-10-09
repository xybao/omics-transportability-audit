"""Explicit-input v3 runner. Changed outputs/environment invalidate cache.
Dry-run has no filesystem writes. No glob-based prediction loading.
"""
from pathlib import Path
import os,sys,json,subprocess,importlib.metadata as im,platform
from common import ROOT,TASKS,sha

def environment():
    return dict(python=sys.version,platform=platform.platform(),packages=dict(sorted((d.metadata['Name'],d.version) for d in im.distributions() if d.metadata['Name'])))
def key(inputs,scripts,cmd):
    import hashlib
    files={str(p.relative_to(ROOT)):sha(p) for p in inputs+scripts}
    return hashlib.sha256(json.dumps({'files':files,'cmd':cmd,'env':environment(),'threads':1,'PYTHONHASHSEED':'0'},sort_keys=True).encode()).hexdigest()
def cache_valid(state,k,outputs):
    return bool(state and state.get('key')==k and all(p.exists() and state.get('outputs',{}).get(str(p.relative_to(ROOT)))==sha(p) for p in outputs))
def definitions():
    scripts=[ROOT/p for p in ('analyse.py','common.py','pilot/posthoc3.py','pilot/bench3.py','pilot/bench.py')]
    stages=[]
    for t in TASKS:
        inputs=[ROOT/'inputs'/f'{n}_{t}.{ext}' for n,ext in [('preds3','parquet'),('subject_components','csv')]]
        outputs=[ROOT/'results'/f'{n}_{t}.{ext}' for n,ext in [('audit','csv'),('metrics_seed','csv'),('calibrated','parquet'),('decisions','parquet'),('conformal','parquet'),('gap','csv'),('heterogeneity','csv'),('adapt','csv')]]
        stages.append((f'core_{t}',[sys.executable,'analyse.py',t,'2000'],inputs,scripts,outputs))
        inputs=[ROOT/'results'/f'{n}_{t}.parquet' for n in ('calibrated','decisions','conformal')]+[ROOT/'inputs'/f'subject_components_{t}.csv']
        outputs=[ROOT/'results'/f'{n}_{t}.csv' for n in ('threshold','scenario','sets','budget_draws','budget_allocations','budget_exclusions','budget')]
        stages.append((f'secondary_{t}',[sys.executable,'secondary.py',t],inputs,[ROOT/'secondary.py',ROOT/'common.py',ROOT/'pilot/posthoc3.py'],outputs))
    return stages

def main():
    dry='--dry' in sys.argv;force='--force' in sys.argv
    only=sys.argv[sys.argv.index('--only')+1].split(',') if '--only' in sys.argv else None
    if not dry:
        (ROOT/'provenance').mkdir(exist_ok=True)
        (ROOT/'provenance/environment.json').write_text(json.dumps(environment(),indent=2),encoding='utf8')
    for name,cmd,inputs,scripts,outputs in definitions():
        if only and name not in only:continue
        f=ROOT/'provenance'/f'{name}.json';state=json.loads(f.read_text()) if f.exists() else None
        missing=[p for p in inputs+scripts if not p.exists()]
        if missing:
            if dry:print(name,'WOULD_WAIT_FOR_INPUT');continue
            raise FileNotFoundError(str(missing))
        k=key(inputs,scripts,cmd)
        if not force and cache_valid(state,k,outputs):print(name,'CACHED_VERIFIED');continue
        if dry:print(name,'WOULD_RUN');continue
        print(name,'RUNNING',flush=True)
        env=os.environ.copy();env.update({n:'1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')});env.update(PYTHONHASHSEED='0',PYTHONIOENCODING='utf-8')
        with open(ROOT/'provenance'/f'{name}.log','w',encoding='utf8') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        record={'key':k,'cmd':cmd,'outputs':{str(p.relative_to(ROOT)):sha(p) for p in outputs}}
        f.write_text(json.dumps(record,indent=2),encoding='utf8')
        print(name,'DONE',flush=True)
if __name__=='__main__':main()
