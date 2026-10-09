"""Calculate fixed-prediction exploratory decompositions and cohort influence summaries."""
import json,csv
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
from common import *

def geo_meta(path):
    rows=list(csv.reader(path.open(encoding='utf8'),delimiter='\t'));d={};chars=[]
    for r in rows:
        if not r:continue
        if r[0]=='!Sample_characteristics_ch1':chars.append(r[1:])
        elif r[0].startswith('!Sample_'):d[r[0][8:]]=r[1:]
    out=pd.DataFrame(d)
    for arr in chars:
        for i,s in enumerate(arr):
            if ': ' in s:
                k,v=s.split(': ',1);out.loc[i,'characteristic:'+k]=v
    return out

def main():
    R=ROOT/'results';md=pd.read_parquet(ROOT/'data_processed/tb/task_od_meta.parquet')
    cm=pd.read_csv(ROOT/'inputs/subject_components_tb_od.csv')
    meta=cm.merge(md,on='sample_id',suffixes=('','_source'),validate='one_to_one')
    lookup=meta.set_index('sample')
    dec=pd.read_parquet(R/'decisions_tb_od.parquet')
    cf=pd.read_parquet(R/'conformal_tb_od.parquet')
    d=dec[(dec.held_out=='GSE39939')&(dec.split=='OOD')&(dec.working==.9)&(dec.y==1)]
    c=cf[(cf.held_out=='GSE39939')&(cf.method=='lasso')&(cf.y==1)]
    for v in (d,c):
        assert v.groupby('sample').seed.nunique().eq(10).all()
        assert not v.duplicated(['sample','seed']).any()
    one=d.groupby('sample').positive.mean().to_frame('sensitivity').join(c.groupby('sample').covered.mean().rename('case_coverage'))
    one=one.join(lookup[['CultureStatus','component_id']]);assert len(one)==52 and one.component_id.nunique()==52
    groups=[]
    for label,v in list(one.groupby('CultureStatus'))+[('pooled inherited labels',one)]:
        groups.append(dict(task='tb_od',cohort='GSE39939',method='lasso',diagnostic_group=label,n_records=len(v),n_subject_components=v.component_id.nunique(),n_seeds=10,sensitivity=v.sensitivity.mean(),case_class_coverage=v.case_coverage.mean(),working=.90,interpretation='Exploratory fixed-program descriptive point estimates; seed repetitions are not independent subjects; no subgroup interval or test; no adjudication of true disease'))
    g=pd.DataFrame(groups);assert g.n_subject_components.tolist()==[25,27,52]
    assert np.allclose(g.sensitivity,[.764,7.2/27,26.3/52])
    assert np.allclose(g.case_class_coverage,[.832,11.2/27,32/52])
    g.to_csv(R/'exploratory_diagnostic_groups.csv',index=False)
    # Sensitivity of descriptive summaries to withholding one TARGET estimate only.
    summaries=[]
    def summarize(frame,keys,value,estimand):
        for key,u in frame.groupby(keys):
            if not isinstance(key,tuple):key=(key,)
            for scope,v in [('all_targets',u),('omit_GSE89403_target_only',u[u.held_out!='GSE89403'])]:
                summaries.append(dict(estimand=estimand,**dict(zip(keys,key)),scope=scope,metric=value,n_cohorts=v.held_out.nunique(),median=v[value].median(),minimum=v[value].min(),maximum=v[value].max(),interpretation='Target-summary influence only; other folds retain their original source training, which may include GSE89403. Not source exclusion, relabelling or retraining.'))
    summarize(pd.read_csv(R/'gap_tb_od.csv'),['method','layer'],'delta','E1_matched_log_loss')
    summarize(pd.read_csv(R/'adapt_tb_od.csv'),['method','layer','adaptation'],'delta','E3_adaptation')
    summarize(pd.read_csv(R/'budget_tb_od.csv'),['method','layer','k'],'delta','E4_label_budget')
    th=pd.read_csv(R/'threshold_tb_od.csv');th=th[(th.split=='OOD')&(th.working==.9)].assign(method='lasso')
    for value in ['sens','spec']:summarize(th,['method','working'],value,'E5_fixed_threshold')
    sets=pd.read_csv(R/'sets_tb_od.csv')
    for value in ['covered','coverage_class1']:summarize(sets,['method'],value,'E6_sets')
    pd.DataFrame(summaries).to_csv(R/'target_summary_influence.csv',index=False)
    # Optional illustrative ranking/threshold pair, on the same held-out cohort.
    cal=pd.read_parquet(R/'calibrated_tb_od.parquet');v=cal[(cal.held_out=='GSE42830')&(cal.method=='lasso')&(cal.split=='OOD')]
    ids=np.sort(v['sample'].unique());y=v.groupby('sample').y.first().reindex(ids).to_numpy();p=v.pivot(index='sample',columns='seed',values='pp').reindex(ids).to_numpy()
    assert p.shape==(57,10)
    W=draws(lookup.component_id.loc[ids].to_numpy(),2000,('v6','GSE42830','AUROC_fixed_seed_mean'))
    vals=[]
    for j in range(10):
        order=np.argsort(p[:,j],kind='stable');pp=p[order,j];yy=y[order];w=W[:,order];starts=np.r_[0,np.flatnonzero(np.diff(pp))+1]
        pos=np.add.reduceat(w*yy,starts,axis=1);neg=np.add.reduceat(w*(1-yy),starts,axis=1);den=pos.sum(1)*neg.sum(1)
        vals.append(np.divide((pos*(np.cumsum(neg,axis=1)-neg/2)).sum(1),den,out=np.full(len(W),np.nan),where=den>0))
    boot=np.mean(vals,axis=0);row=th[th.held_out=='GSE42830'].iloc[0]
    pd.DataFrame([dict(cohort='GSE42830',method='lasso',case_records=int(y.sum()),control_records=int((1-y).sum()),AUROC=np.mean([roc_auc_score(y,p[:,j]) for j in range(10)]),AUROC_lo=np.nanquantile(boot,.025),AUROC_hi=np.nanquantile(boot,.975),AUROC_n_valid=int(np.isfinite(boot).sum()),sensitivity=row.sens,sensitivity_lo=row.sens_lo,sensitivity_hi=row.sens_hi,working=.9,interpretation='Exploratory illustrative comparison of distinct metrics; mean of ten fixed-program AUROCs; 2000 subject-component draws. Small selected cohort; no claim of clinical adequacy or isolated calibration failure.')]).to_csv(R/'ranking_threshold_example.csv',index=False)
    # Evidence-level metadata distinctions, without replacing inherited analysis labels.
    s1=pd.read_csv(R/'cohort_definitions.csv').rename(columns={'age_numeric_n':'n_records_with_numeric_age','age_missing_n':'n_records_missing_age','age_nonnumeric_n':'n_records_unparseable_age'})
    s1['age_unit']='unverified';s1['age_unit_basis']='No verified unit in registered record-level metadata';s1['age_summary_scope']='Raw field summary by measurement record; not harmonized or suitable for cross-cohort comparison'
    s1.to_csv(R/'cohort_definitions.csv',index=False)
    s17=pd.read_csv(R/'clinical_source_audit.csv').drop(columns=['evidence_status'],errors='ignore')
    s17['study_standard_evidence']='Primary Methods read; source-standard summary only'
    s17['current_record_mapping']='Registered task identifiers checked; individual original clinical adjudication not fully resolved'
    s17.loc[s17.cohort.eq('GSE39939'),'current_record_mapping']='All 52 current case records mapped to curated culture fields (25 confirmed/27 negative); not independent clinical adjudication'
    s17.loc[s17.cohort.eq('GSE89403'),'study_standard_evidence']='Study-level TB treatment cohort verified; OD task identity requires sample metadata, not the word control in the paper'
    s18=pd.read_csv(R/'OD_composition.csv').rename(columns={'count':'curated_field_count'})
    s18['primary_study_group_count']=np.nan;s18['record_mapping_verified']='Not verified';s18['count_interpretation']='Non-additive evidence levels; zero curated count does not establish disease absence'
    k=(s18.cohort=='GSE73408')&(s18.category=='pneumonia');s18.loc[k,'primary_study_group_count']=39
    sourcepath=ROOT/'source_evidence_v6/GSE89403.metadata.txt';od=meta[(meta.study=='GSE89403')&(meta.y==0)].copy();assert len(od)==17
    if sourcepath.exists():
        gm=geo_meta(sourcepath);od=od.merge(gm,left_on='sample_id',right_on='geo_accession',how='left',validate='one_to_one')
    mapped=[]
    for _,r in od.iterrows():
        fields={str(k):str(val) for k,val in r.items() if str(k).startswith('characteristic:') and pd.notna(val)}
        group=r.get('characteristic:disease state','Unavailable')
        mapped.append(dict(sample_id=r.sample_id,public_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+r.sample_id,original_title=r.get('title','Unavailable'),original_group=group,original_time=r.get('characteristic:time','Unavailable'),curated_TBStatus=r.TBStatus,current_y=int(r.y),curated_time=r.MeasurementTime,curated_LungDxStatus=r.LungDxStatus,evidence_state='Public sample-ID to source group verified; individual diagnosis/TB exclusion unresolved' if group=='Lung Dx Controls' else 'Public sample-ID to MTP Controls verified; MTP clinical definition and OD task eligibility unresolved',source_metadata_available=bool(fields)))
    pd.DataFrame(mapped).to_csv(R/'GSE89403_record_mapping.csv',index=False)
    assert pd.DataFrame(mapped).original_group.value_counts().to_dict()=={'Lung Dx Controls':9,'MTP Controls':8}
    assert set(pd.DataFrame(mapped).original_time)=={'DX'}
    ix=s17.cohort.eq('GSE89403')
    s17.loc[ix,'current_record_mapping']='17/17 GSM IDs mapped to original GEO characteristics: 9 Lung Dx Controls and 8 MTP Controls; all time DX; inherited OD/y=0 retained'
    s17.loc[ix,'control_definition']='Original GEO disease-state group verified (9 Lung Dx Controls, 8 MTP Controls). Exact MTP disease/control definition and individual TB exclusion not established. This cohort is not a clinically verified uniform disease-control comparison.'
    s17.loc[ix,'unresolved']='Clinical meaning and OD eligibility of MTP Controls; individual alternative diagnoses and TB exclusion. Retained as inherited-label benchmark with target-summary influence analysis; not a clinical exemplar.'
    gm=geo_meta(ROOT/'source_evidence_v6/GSE73408.metadata.txt')
    pneu=meta[(meta.study=='GSE73408')&(meta.y==0)].merge(gm,left_on='sample_id',right_on='geo_accession',validate='one_to_one')
    assert len(pneu)==39 and pneu['characteristic:clinical group'].eq('PNA').all() and pneu.PneumoniaStatus.eq('Negative').all()
    assert meta.y.eq(meta.TBStatus.eq('PTB').astype(int)).all()
    pd.DataFrame(dict(sample_id=pneu.sample_id,original_clinical_group=pneu['characteristic:clinical group'],curated_PneumoniaStatus=pneu.PneumoniaStatus,curated_TBStatus=pneu.TBStatus,current_y=pneu.y,source_url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73408/matrix/GSE73408_series_matrix.txt.gz',decision='39/39 original PNA groups mapped; curated description conflicts; OD labels unchanged')).to_csv(R/'GSE73408_record_mapping.csv',index=False)
    s18.loc[k,'record_mapping_verified']='Yes: 39/39 public GSM IDs map to original clinical group PNA; clinical diagnosis itself not re-adjudicated'
    ix=s17.cohort.eq('GSE73408');s17.loc[ix,'current_record_mapping']='39/39 current OD GSM IDs map to original GEO clinical group PNA; Table 1 reports 39 pneumonia participants'
    s17.loc[ix,'control_definition']='Original PNA grouping is verified for all 39 current OD samples. The curated PneumoniaStatus=Negative conflicts with that descriptive grouping; fixed OD/y=0 labels remain unchanged.'
    s17.loc[ix,'unresolved']='Clinical diagnosis not independently re-adjudicated; curated descriptive field discrepancy remains, with explicit original mapping. Current outcome labels and inclusion do not use this field.'
    s17.to_csv(R/'clinical_source_audit.csv',index=False);s18.to_csv(R/'OD_composition.csv',index=False)
    print(g[['diagnostic_group','n_subject_components','sensitivity','case_class_coverage']].to_string(index=False))
if __name__=='__main__':main()
