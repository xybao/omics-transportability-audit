"""Prespecified v3 implementations for threshold, sets and fixed-budget updates.
No row bootstrap, no selection on residual evaluation labels for layer A.
"""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys
import numpy as np,pandas as pd
from common import *
sys.path.insert(0,str(ROOT/'pilot'))
import posthoc3 as P

def main(task,B=2000,repeats=200):
    out=ROOT/'results';cm=pd.read_csv(ROOT/'inputs'/f'subject_components_{task}.csv').set_index('sample').component_id
    dec=pd.read_parquet(out/f'decisions_{task}.parquet');th=[];sc=[]
    for (h,split,t),v in dec.groupby(['held_out','split','working']):
        ids=np.sort(v['sample'].unique());y=v.groupby('sample').y.first().reindex(ids).to_numpy()
        p=v.pivot(index='sample',columns='seed',values='positive').reindex(ids).to_numpy();mask=np.isfinite(p);p=np.nan_to_num(p)
        W=draws(cm.loc[ids].to_numpy(),B,(task,h,split,'threshold'))
        estimates=[];reps=[]
        for k in (1,0):
            keep=(y==k)[:,None];good=p if k else mask.astype(float)-p
            pt=((good*keep).sum(0)/(mask*keep).sum(0)).mean()
            bs=((W@(good*keep))/(W@(mask*keep).astype(float))).mean(1)
            estimates.append(pt);reps.append(bs)
        se,sp=estimates;sb,pb=reps
        base=dict(task=task,held_out=h,split=split,method='lasso',working=t,estimand='mean_10_seed_fixed_source_threshold',n_records=len(y),n_subject=cm.loc[ids].nunique(),n_case=int(y.sum()),sens=se,spec=sp)
        for nm,rep in [('sens',sb),('spec',pb)]:base.update({nm+'_'+k:val for k,val in interval(rep).items()})
        th.append(base)
        for prev in (.01,.05,.10,.20):
            values={'TP':1000*prev*se,'FN':1000*prev*(1-se),'FP':1000*(1-prev)*(1-sp),'TN':1000*(1-prev)*sp}
            values['PPV']=values['TP']/(values['TP']+values['FP']) if values['TP']+values['FP'] else np.nan
            values['risk_after_negative']=values['FN']/(values['FN']+values['TN']) if values['FN']+values['TN'] else np.nan
            boots={'TP':1000*prev*sb,'FN':1000*prev*(1-sb),'FP':1000*(1-prev)*(1-pb),'TN':1000*(1-prev)*pb}
            boots['PPV']=boots['TP']/(boots['TP']+boots['FP']);boots['risk_after_negative']=boots['FN']/(boots['FN']+boots['TN'])
            for name,a in boots.items():values.update({name+'_'+k:z for k,z in interval(a).items()})
            sc.append(dict(task=task,held_out=h,split=split,working=t,assumed_prevalence=prev,**values))
    pd.DataFrame(th).to_csv(out/f'threshold_{task}.csv',index=False)
    pd.DataFrame(sc).to_csv(out/f'scenario_{task}.csv',index=False)
    cf=pd.read_parquet(out/f'conformal_{task}.parquet');sets=[]
    for (h,m),v in cf.groupby(['held_out','method']):
        e=v.groupby('sample').agg(y=('y','first'),covered=('covered','mean'),empty=('empty','mean'),single=('single','mean'),both=('both','mean'))
        W=draws(cm.loc[e.index].to_numpy(),B,(task,h,'sets'));r=dict(task=task,held_out=h,method=m,n_subject=cm.loc[e.index].nunique())
        for col in ('covered','empty','single','both'):
            r[col]=e[col].mean();r.update({col+'_'+k:z for k,z in interval((W@e[col].to_numpy())/W.sum(1)).items()})
        for k in (0,1):r[f'coverage_class{k}']=e.loc[e.y==k,'covered'].mean()
        assert abs(r['empty']+r['single']+r['both']-1)<1e-12
        sets.append(r)
    pd.DataFrame(sets).to_csv(out/f'sets_{task}.csv',index=False)
    c=pd.read_parquet(out/f'calibrated_{task}.parquet');budget=[];alloc=[];excluded=[]
    for (h,m),u in c[c.split=='OOD'].groupby(['held_out','method']):
        e=u.groupby('sample').agg(y=('y','first'),p=('pp','mean'),n_seed=('seed','nunique'))
        assert (e.n_seed==10).all()
        comps=cm.loc[e.index].to_numpy();subs=np.sort(np.unique(comps));ixof={s:np.where(comps==s)[0] for s in subs}
        if len(subs)<60:
            excluded.append(dict(task=task,held_out=h,method=m,n_subject=len(subs),reason='fewer_than_60_subject_components'));continue
        rng=np.random.default_rng(seed(task,h,'shared_label_budget'))
        y=e.y.to_numpy();p=e.p.to_numpy()
        for rep in range(repeats):
            order=rng.permutation(subs);ev=np.concatenate([ixof[s] for s in order[40:]])
            for k in (10,20,40):
                lab=np.concatenate([ixof[s] for s in order[:k]])
                subject_y=np.array([y[ixof[s][0]] for s in order[:k]])
                weights=np.concatenate([np.full(len(ixof[s]),1/len(ixof[s])) for s in order[:k]])
                success=len(np.unique(subject_y))==2
                if success:
                    a,_=P.fit_logistic(P.logit(p[lab]),y[lab],w=weights,slope=False);q=P.sigm(a+P.logit(p[ev]))
                else:q=p[ev]
                dl=loss(y[ev],q)-loss(y[ev],p[ev]);both=len(np.unique(y[ev]))==2
                budget.append(dict(task=task,held_out=h,method=m,repeat=rep,k=k,rng_seed=seed(task,h,'shared_label_budget'),success=success,n_label_cases=int(subject_y.sum()),n_eval_subject=len(order[40:]),eval_two_classes=both,d_nll_A=float(dl.mean()),d_nll_Bw=float(sum(.5*dl[y[ev]==z].mean() for z in (0,1))) if both else np.nan))
                if m=='lasso':alloc.append(dict(task=task,held_out=h,repeat=rep,k=k,label_components=';'.join(map(str,order[:k])),eval_components=';'.join(map(str,order[40:]))))
    bb=pd.DataFrame(budget);bb.to_csv(out/f'budget_draws_{task}.csv',index=False)
    pd.DataFrame(alloc).to_csv(out/f'budget_allocations_{task}.csv',index=False)
    pd.DataFrame(excluded).to_csv(out/f'budget_exclusions_{task}.csv',index=False)
    summary=[]
    for (h,m,k),v in bb.groupby(['held_out','method','k']):
        for layer in ('A','Bw'):
            a=v[f'd_nll_{layer}'].dropna().to_numpy()
            summary.append(dict(task=task,held_out=h,method=m,k=k,layer=layer,estimand='allocation_mean_fixed_OOD_ensemble',delta=float(a.mean()),allocation_lo=float(np.quantile(a,.025)),allocation_hi=float(np.quantile(a,.975)),mc_se=float(a.std(ddof=1)/np.sqrt(len(a))),n_requested=repeats,n_evaluable=len(a),fallback_rate=1-v.success.mean()))
    pd.DataFrame(summary).to_csv(out/f'budget_{task}.csv',index=False)
    print(task,'thresholds, conformity sets and shared allocations complete',flush=True)
if __name__=='__main__':main(sys.argv[1])
