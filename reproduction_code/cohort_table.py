"""Traceable cohort definitions from registered curated metadata; unknown != negative."""
import json
import pandas as pd,numpy as np
from common import *
UNKNOWN='Not reported in available curated fields; original-source confirmation required'
def vals(d,c):
    if c not in d:return UNKNOWN
    a=sorted(set(str(x) for x in d[c].dropna() if str(x).strip() not in ('','NA','nan','None')))
    return '; '.join(a) if a else UNKNOWN
def main():
    source=json.loads((ROOT/'cohort_source_metadata.json').read_text(encoding='utf8'));rows=[];overlaps=[];selection=[]
    raw=pd.read_parquet(ROOT/'data_processed/tb/tb_meta.parquet')
    maps={}
    for task in TASKS:
        cm=pd.read_csv(ROOT/'inputs'/f'subject_components_{task}.csv');maps[task]=cm
        if task=='crc4':md=pd.read_parquet(ROOT/'data_processed/crc4_meta.parquet')
        else:md=pd.read_parquet(ROOT/'data_processed/tb'/f'task_{task[3:]}_meta.parquet')
        md=cm.merge(md,on='sample_id',suffixes=('','_original'),validate='one_to_one')
        eligible=set(pd.read_csv(ROOT/'results'/f'gap_{task}.csv').held_out)
        for h,u in md.groupby('cohort_group'):
            studies=sorted(u.study.unique()) if task!='crc4' else sorted(u.study_name.unique())
            sr=[source.get(s,{}) for s in studies]
            def field(k):return ' | '.join(s+': '+str(v.get(k) or UNKNOWN) for s,v in zip(studies,sr))
            original=raw[raw.study.isin(studies)] if task!='crc4' else None
            r=dict(task=task,cohort=h,original_studies='; '.join(studies),role='held_out_and_source' if h in eligible else 'source_only',eligibility_reason='both task labels have at least 10 records' if h in eligible else 'at least one task label has fewer than 10 records; source pool only',records=len(u),components=u.component_id.nunique(),case_records=int(u.y.sum()),control_records=int((1-u.y).sum()),repeat_records=len(u)-u.component_id.nunique(),country_curated=vals(u,'country'),country_status='curated value, not contact address; not independently country-validated for this paper',specimen=vals(u,'body_site' if task=='crc4' else 'Tissue'),platform=vals(u,'sequencing_platform') if task=='crc4' else field('Platform'),age_curated=vals(u,'age' if task=='crc4' else 'Age'),HIV_curated=UNKNOWN if task=='crc4' else vals(u,'HIVStatus'),sample_time=vals(u,'MeasurementTime'),treatment=vals(u,'Treatment'),diagnostic_basis_curated=UNKNOWN if task=='crc4' else field('DiagnosisMethod'),LTBI_tests_curated='; '.join(c+': '+vals(u[u.y==0],c) for c in ('TST','QFT_GIT')) if task=='tb_ltbi' else 'Not applicable',other_disease_curated='; '.join(c+': '+vals(u[u.y==0],c) for c in ('PneumoniaStatus','SarcoidosisStatus','LungCancerStatus','LungDxStatus')) if task=='tb_od' else 'Not applicable',recruitment_and_reference_standard='Not independently adjudicated; no consecutive-suspect recruitment assumption',source_url='; '.join('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+s for s in studies) if task!='crc4' else 'https://bioconductor.org/packages/curatedMetagenomicData/; PMID: '+vals(u,'PMID'),curated_study_notes=field('Notes') if task!='crc4' else vals(u,'study_condition'),original_export_records=len(original) if original is not None else np.nan,raw_to_task_exclusions=len(original)-len(u) if original is not None else np.nan,evidence='registered task metadata + curatedTBData DataSummary; clinical codes not reinterpreted')
            if task=='crc4':r['sample_time']=UNKNOWN;r['treatment']=UNKNOWN
            rows.append(r)
        if task!='crc4':
            # Describe actual stable CAL_C record choice across seeds without new quantiles.
            d=pd.read_parquet(ROOT/'inputs'/f'preds3_{task}.parquet');d=d[(d.method=='lasso')&(d.prep=='global')&(d.split=='CAL_C')]
            lookup=md.set_index('sample')
            for (h,s),v in d.groupby(['held_out','seed']):
                v=v.assign(component=lookup.component_id.loc[v['sample']].to_numpy())
                for component,w in v.groupby('component'):
                    if len(w)<2:continue
                    chosen=w.sort_values('sample').iloc[0]['sample'];info=lookup.loc[chosen]
                    selection.append(dict(task=task,held_out=h,seed=s,component_id=component,n_candidate_records=len(w),selected_study=info.study,selected_time=str(info.get('MeasurementTime','')),candidate_studies=';'.join(sorted(lookup.loc[w['sample'],'study'].unique())),rule='minimum_sample_index',interpretation='deterministic ordering can favor a platform; no random-measurement exchangeability claim'))
    for i,a in enumerate(TASKS):
        for b in TASKS[i+1:]:
            if 'crc4' in (a,b):continue
            common=set(maps[a].sample_id)&set(maps[b].sample_id)
            overlaps.append(dict(task_a=a,task_b=b,shared_sample_ids=len(common),interpretation='lower-bound documented shared records, not count of all shared people'))
    pd.DataFrame(rows).to_csv(ROOT/'results/cohort_definitions.csv',index=False)
    pd.DataFrame(overlaps).to_csv(ROOT/'results/cross_task_overlap.csv',index=False)
    pd.DataFrame(selection).to_csv(ROOT/'results/conformal_record_selection.csv',index=False)
    print('Cohort metadata rows:',len(rows),'; unknown clinical definitions explicitly retained')
if __name__=='__main__':main()
