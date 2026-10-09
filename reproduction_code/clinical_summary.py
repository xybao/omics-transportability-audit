"""Summarize source metadata and clinical composition for the registered cohorts."""
import json
import numpy as np
import pandas as pd
from common import *

def main():
    evidence=json.loads((ROOT/'clinical_evidence.json').read_text(encoding='utf8'))
    rows=[];composition=[]
    md=pd.read_parquet(ROOT/'data_processed/tb/task_od_meta.parquet')
    cm=pd.read_csv(ROOT/'inputs/subject_components_tb_od.csv')
    md=cm.merge(md,on='sample_id',suffixes=('','_original'),validate='one_to_one')
    eligible=set(pd.read_csv(ROOT/'results/gap_tb_od.csv').held_out)
    assert eligible==set(evidence)
    for h,u in md.groupby('cohort_group'):
        cases=u[u.y==1];controls=u[u.y==0]
        flags={}
        for col,name in [('PneumoniaStatus','pneumonia'),('SarcoidosisStatus','sarcoidosis'),('LungCancerStatus','lung_cancer')]:
            flags[name]=controls[col].fillna('').str.lower().str.startswith(('positive','active','inactive','non-active'))
        # Sarcoidosis has descriptive active/non-active values, but 'Negative' must not count.
        f=pd.DataFrame(flags,index=controls.index);cnt=f.sum(axis=1)
        for name in f:composition.append(dict(cohort=h,role='OOD' if h in eligible else 'source_only',category=name,count=int(f[name].sum()),basis='positive/descriptive curated category; categories may overlap'))
        for name,num in [('unclassified_OD',int((cnt==0).sum())),('overlapping_categories',int((cnt>1).sum()))]:composition.append(dict(cohort=h,role='OOD' if h in eligible else 'source_only',category=name,count=num,basis='computed across named curated diagnosis fields; not proof of absence of other disease'))
        if h not in eligible:continue
        rows.append(dict(cohort=h,**evidence[h],case_records=len(cases),OD_records=len(controls),components=u.component_id.nunique(),case_culture_field=json.dumps(cases.CultureStatus.fillna('unknown').value_counts().to_dict()),OD_lungdx_field=json.dumps(controls.LungDxStatus.fillna('unknown').value_counts().to_dict()),local_selection='Exact registered task_od_meta.parquet sample IDs joined 1:1 to subject_components_tb_od.csv; PTB/OD labels retained; no new exclusions',evidence_status='Primary Methods read; remaining record-level gaps explicitly unresolved',review_date='2026-09-25'))
    pd.DataFrame(rows).to_csv(ROOT/'results/clinical_source_audit.csv',index=False)
    pd.DataFrame(composition).to_csv(ROOT/'results/OD_composition.csv',index=False)
    d=pd.read_csv(ROOT/'results/cohort_definitions.csv')
    d=d.rename(columns={'raw_to_task_exclusions':'export_records_not_in_task'})
    for task in TASKS:
        path=ROOT/'data_processed/crc4_meta.parquet' if task=='crc4' else ROOT/'data_processed/tb'/f'task_{task[3:]}_meta.parquet'
        x=pd.read_parquet(path);c=pd.read_csv(ROOT/'inputs'/f'subject_components_{task}.csv')
        x=c.merge(x,on='sample_id',suffixes=('','_original'),validate='one_to_one')
        for h,u in x.groupby('cohort_group'):
            raw=u['age' if task=='crc4' else 'Age'];v=pd.to_numeric(raw,errors='coerce');v=v.where(np.isfinite(v))
            ix=(d.task==task)&(d.cohort==h)
            n=int(v.notna().sum());missing=int(raw.isna().sum());bad=int(raw.notna().sum()-n)
            d.loc[ix,'age_curated']=f'numeric n={n}; min/median/max='+('/'.join(f'{z:.1f}' for z in [v.min(),v.median(),v.max()]) if n else 'unknown')+f'; missing={missing}; nonnumeric={bad}; source units not harmonized'
            d.loc[ix,'age_numeric_n']=n;d.loc[ix,'age_missing_n']=missing;d.loc[ix,'age_nonnumeric_n']=bad
            if task=='tb_od':
                c0=pd.DataFrame(composition);c0=c0[c0.cohort==h]
                d.loc[ix,'other_disease_curated']='; '.join(f'{r.category}={r.count}' for r in c0.itertuples())+'; see S17/S18 for source conflicts'
    d.to_csv(ROOT/'results/cohort_definitions.csv',index=False)
    print('Clinical source audit: eight OOD cohorts; labels and inclusion unchanged')
if __name__=='__main__':main()
