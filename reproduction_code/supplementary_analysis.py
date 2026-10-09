"""Calculate supplementary confidence intervals, label summaries and boundary annotations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import pandas as pd,numpy as np
from scipy.stats import beta
from common import *

def exact(k,n):
    return (0. if k==0 else beta.ppf(.025,k,n-k+1),1. if k==n else beta.ppf(.975,k+1,n-k)) if n else (np.nan,np.nan)

def main():
    r=ROOT/'results';audit=[];single=[];hets=[];labels=[];exclusions=[]
    for task in TASKS:
        md=pd.read_csv(ROOT/'inputs'/f'subject_components_{task}.csv').set_index('sample');cm=md.component_id
        th=pd.read_csv(r/f'threshold_{task}.csv')
        for nm in ('sens','spec'):
            th[nm+'_bootstrap_degenerate']=np.isclose(th[nm+'_lo'],th[nm+'_hi'],atol=1e-12)
            for row in th[th[nm+'_bootstrap_degenerate']].to_dict('records'):
                audit.append(dict(task=task,held_out=row['held_out'],split=row['split'],method='lasso',working=row['working'],metric=nm,value=row[nm],issue='empirical_bootstrap_degenerate_not_population_certainty'))
        th.to_csv(r/f'threshold_{task}.csv',index=False)
        # Distinct supplemental estimand: seed 0, smallest sample ID per subject,
        # chosen without reference to outcome/prediction. No fractional counts.
        dec=pd.read_parquet(r/f'decisions_{task}.parquet')
        for (h,split,working),u in dec[dec.seed==0].groupby(['held_out','split','working']):
            u=u.assign(component=cm.loc[u['sample']].to_numpy()).sort_values('sample').drop_duplicates('component')
            for y,nm in [(1,'sens'),(0,'spec')]:
                v=u[u.y==y];n=len(v);k=int((v.positive==y).sum());lo,hi=exact(k,n)
                single.append(dict(task=task,held_out=h,split=split,working=working,metric=nm,seed=0,n_subject=n,n_success=k,estimate=k/n if n else np.nan,lo=lo,hi=hi,estimand='seed0_one_stably_selected_record_per_component',interval='Clopper_Pearson_95pct',status='post_review_supplement_not_prespecified_primary'))
        cf=pd.read_parquet(r/f'conformal_{task}.parquet');sets=pd.read_csv(r/f'sets_{task}.csv')
        for (h,m),u in cf.groupby(['held_out','method']):
            e=u.groupby('sample').agg(y=('y','first'),covered=('covered','mean'))
            ix=(sets.held_out==h)&(sets.method==m)
            sets.loc[ix,'n_records']=len(e)
            for y in (0,1):
                v=e[e.y==y];W=draws(cm.loc[v.index].to_numpy(),2000,(task,h,'sets_class',y))
                prefix=f'coverage_class{y}'
                for k,val in interval(W@v.covered.to_numpy()/W.sum(1)).items():sets.loc[ix,prefix+'_'+k]=val
                sets.loc[ix,f'n_class{y}_records']=len(v)
                sets.loc[ix,f'n_class{y}_subjects']=cm.loc[v.index].nunique()
            u0=u[u.seed==0].assign(component=lambda z:cm.loc[z['sample']].to_numpy()).sort_values('sample').drop_duplicates('component')
            for y in ('all',0,1):
                v=u0 if y=='all' else u0[u0.y==y];n=len(v);k=int(v.covered.sum());lo,hi=exact(k,n)
                single.append(dict(task=task,held_out=h,split='OOD',method=m,metric=f'coverage_{y}',seed=0,n_subject=n,n_success=k,estimate=k/n if n else np.nan,lo=lo,hi=hi,estimand='seed0_one_stably_selected_record_per_component',interval='Clopper_Pearson_95pct',status='post_review_supplement_not_prespecified_primary'))
        for nm in ('covered','empty','single','both','coverage_class0','coverage_class1'):
            sets[nm+'_bootstrap_degenerate']=np.isclose(sets[nm+'_lo'],sets[nm+'_hi'],atol=1e-12)
            for row in sets[sets[nm+'_bootstrap_degenerate']].to_dict('records'):
                audit.append(dict(task=task,held_out=row['held_out'],split='OOD',method=row['method'],metric=nm,value=row[nm],issue='empirical_bootstrap_degenerate_not_population_certainty'))
        sets.to_csv(r/f'sets_{task}.csv',index=False)
        het=pd.read_csv(r/f'heterogeneity_{task}.csv')
        for h,u in md.groupby('cohort_group'):
            ix=het.held_out==h
            het.loc[ix,'case_records']=int(u.y.sum());het.loc[ix,'control_records']=int((1-u.y).sum())
            het.loc[ix,'case_components']=u[u.y==1].component_id.nunique();het.loc[ix,'control_components']=u[u.y==0].component_id.nunique()
        het['included_in_meta']=np.isfinite(het.cal_int)&np.isfinite(het.se)&(het.se>0)
        het['se_definition']='SD_of_valid_subject_cluster_bootstrap_intercepts'
        het['meta_parameters']='DL/REML/PM; metafor rma.uni; vi=se^2; test=knha; Q-profile tau interval'
        hets.append(het)
        b=pd.read_csv(r/f'budget_draws_{task}.csv')
        for (h,m,k),u in b.groupby(['held_out','method','k']):
            labels.append(dict(task=task,held_out=h,method=m,k=k,n_allocations=len(u),min_cases=u.n_label_cases.min(),q025_cases=u.n_label_cases.quantile(.025),median_cases=u.n_label_cases.median(),q975_cases=u.n_label_cases.quantile(.975),max_cases=u.n_label_cases.max(),single_class_fraction=1-u.success.mean(),fallback='retain_unupdated_ensemble',eligible_cohorts=b.held_out.nunique()))
        exclusions.append(pd.read_csv(r/f'budget_exclusions_{task}.csv'))
    for name,df in [('boundary_audit',pd.DataFrame(audit)),('single_program_intervals',pd.DataFrame(single)),('E2_public_inputs',pd.concat(hets)),('budget_label_composition',pd.DataFrame(labels)),('budget_exclusions_all',pd.concat(exclusions))]:df.to_csv(r/f'{name}.csv',index=False)
    assert ((pd.DataFrame(single).hi-pd.DataFrame(single).lo)>0).all()
    print('V4 supplemental audit:',len(audit),'degenerate empirical intervals explicitly marked')
if __name__=='__main__':main()
