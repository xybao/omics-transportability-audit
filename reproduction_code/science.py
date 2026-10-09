"""Portable analysis entry. Explicit approved inputs; new output directory only.
No access to previous versions, manuscript files, or sealed validation data.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True,type=Path);ap.add_argument('--output-dir',required=True,type=Path);ap.add_argument('--rscript',required=True);ap.add_argument('--r-library',type=Path)
    a=ap.parse_args();src=Path(__file__).resolve().parent;inp=a.input_dir.resolve();out=a.output_dir.resolve()
    if out.exists():raise SystemExit('Output directory must not exist; choose a fresh directory')
    missing=[n for n in ('GSE89403.metadata.txt','GSE73408.metadata.txt') if not (src/'source_evidence_v6'/n).is_file()]
    if missing:raise SystemExit('Obtain the auxiliary GEO metadata listed in SOURCE_RECORDS.md before scientific evaluation: '+', '.join(missing))
    records=json.loads((src/'ACCEPTED_INPUTS.json').read_text(encoding='utf8'))
    # Validate against a shipped, fixed accepted manifest BEFORE creating outputs.
    for r in records:
        p=(inp/r['path']).resolve()
        if not p.is_relative_to(inp) or not p.is_file() or digest(p)!=r['sha256']:raise SystemExit('Unaccepted input: '+r['path'])
    out.mkdir(parents=True)
    for n in ('analyse.py','secondary.py','common.py','runner.py','aggregate.py','meta_analysis.R','supplementary_analysis.py','cohort_table.py','scenario_annotations.py','clinical_summary.py','clinical_evidence.json','exploratory_analysis.py'):
        shutil.copyfile(src/n,out/n)
    shutil.copytree(src/'pilot',out/'pilot',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(src/'source_evidence_v6',out/'source_evidence_v6')
    shutil.copyfile(src/'cohort_source_metadata.json',out/'cohort_source_metadata.json')
    for r in records:
        dest=out/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(inp/r['path'],dest)
    (out/'results').mkdir();(out/'provenance').mkdir()
    env=os.environ.copy();env.update(RSCRIPT=a.rscript,PYTHONIOENCODING='utf-8',PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if a.r_library:env['R_LIBS_USER']=str(a.r_library.resolve())
    for n in ('runner.py','aggregate.py','supplementary_analysis.py','cohort_table.py','scenario_annotations.py','clinical_summary.py','exploratory_analysis.py'):
        print('PORTABLE',n,flush=True);subprocess.run([sys.executable,str(out/n)],cwd=out,env=env,check=True)
if __name__=='__main__':main()
