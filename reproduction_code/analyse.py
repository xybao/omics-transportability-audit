"""Evaluate registered saved predictions with matched models and calibrators."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[n]='1'
import sys, json, zlib
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from common import *
sys.path.insert(0,str(ROOT/'pilot'))
import posthoc3 as P
import bench3 as B3

def main(task,B=2000):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    md=pd.read_csv(ROOT/'inputs'/f'subject_components_{task}.csv').sort_values('sample')
    d=pd.read_parquet(ROOT/'inputs'/f'preds3_{task}.parquet')
    d=d[(d.prep=='global') & d.method.isin(MODELS)].copy()
    assert set(d.method)==set(MODELS)
    cm=md.set_index('sample').component_id
    assert not d.duplicated(['held_out','method','seed','split','sample']).any()
    checks=[]; unit=[]; cal=[]; conf=[]; thresholds=[]
    # Audit every saved partition against a reconstructed full split, including TRAIN.
    for (h,s),u in d.groupby(['held_out','seed']):
        y=md.y.to_numpy();g=md.cohort_group.to_numpy();comp=md.component_id.to_numpy()
        pool=np.where(g!=h)[0];ood=np.where(g==h)[0]
        sp=B3.source_split(y,g,pool,np.random.default_rng(1000*s+zlib.crc32(h.encode())%997),comp=comp)
        sp['OOD']=ood
        for m,v in u.groupby('method'):
            for part in ('CAL_P','CAL_C','ID','OOD'):
                assert set(v[v.split==part]['sample'])==set(sp[part]),(task,h,s,m,part)
            union=[set(comp[ix]) for ix in sp.values()]
            assert all(not (a&b) for i,a in enumerate(union) for b in union[i+1:])
            checks.append(dict(task=task,held_out=h,seed=int(s),method=m,train_n=len(sp['TRAIN']),status='PASS'))
    for (h,m,s),u in d.groupby(['held_out','method','seed']):
        cp=u[u.split=='CAL_P']; cc=u[u.split=='CAL_C']
        assert cp.y.nunique()==2 and len(cp)>=10
        a,b=P.fit_logistic(P.logit(cp.p.to_numpy()),cp.y.to_numpy())
        transform=lambda x:P.sigm(a+b*P.logit(x))
        pcp=transform(cp.p.to_numpy());pcc=transform(cc.p.to_numpy());pi=cp.y.mean()
        ident=f'{task}|{h}|{m}|global|seed={s}|source_platt'
        for split in ('ID','OOD'):
            t=u[u.split==split].copy(); y=t.y.to_numpy();p=transform(t.p.to_numpy())
            assert len(set(y))==2
            t['pp']=p;t['pp_B']=P.shift_prior(p,pi,.5);t['pi_cal']=pi;t['program_id']=ident
            cal.append(t)
            for layer,pp in [('A',p),('B',t.pp_B.to_numpy())]:
                w=np.ones(len(y)) if layer=='A' else np.where(y==1,.5/y.mean(),.5/(1-y.mean()))
                met=P.wmetrics(y,pp,w)
                unit.append(dict(task=task,held_out=h,method=m,seed=s,split=split,layer=layer,program_id=ident,n=len(t),n_subject=cm.loc[t['sample']].nunique(),**met))
            if m=='lasso':
                for working in (.9,.95):
                    pos=np.sort(pcp[cp.y.to_numpy()==1]);threshold=pos[int(np.floor((1-working)*len(pos)+1e-9))]
                    for ix,row in enumerate(t.itertuples()):
                        thresholds.append(dict(task=task,held_out=h,seed=s,split=split,sample=row.sample,y=row.y,working=working,threshold=threshold,positive=int(p[ix]>=threshold),program_id=ident))
            if split=='OOD':
                # One preselected measurement per CAL_C component for conformity quantiles.
                cct=cc.assign(component=cm.loc[cc['sample']].to_numpy()).copy()
                cct['pp']=pcc;cct=cct.sort_values('sample').drop_duplicates('component')
                q=[P.conformal_quantile(1-np.where(k==1,cct.pp.to_numpy(),1-cct.pp.to_numpy())[cct.y.to_numpy()==k]) for k in (0,1)]
                sets=np.column_stack([p<=q[0],1-p<=q[1]])
                for i,row in enumerate(t.itertuples()):
                    conf.append(dict(task=task,held_out=h,method=m,seed=s,sample=row.sample,y=row.y,covered=int(sets[i,row.y]),empty=int(sets[i].sum()==0),single=int(sets[i].sum()==1),both=int(sets[i].sum()==2),n_cal0=int((cct.y==0).sum()),n_cal1=int((cct.y==1).sum())))
    pd.DataFrame(checks).to_csv(out/f'audit_{task}.csv',index=False)
    pd.DataFrame(unit).to_csv(out/f'metrics_seed_{task}.csv',index=False)
    c=pd.concat(cal);c.to_parquet(out/f'calibrated_{task}.parquet',index=False)
    pd.DataFrame(thresholds).to_parquet(out/f'decisions_{task}.parquet',index=False)
    pd.DataFrame(conf).to_parquet(out/f'conformal_{task}.parquet',index=False)
    gap=[];het=[];adapt=[]
    for (h,m),u in c.groupby(['held_out','method']):
        boot={}; point={}
        for split,v in u.groupby('split'):
            ids=np.sort(v['sample'].unique()); comps=cm.loc[ids].to_numpy()
            W=draws(comps,B,(task,h,'evaluation',split))
            yy=v.groupby('sample').y.first().reindex(ids).to_numpy()
            for layer,col in [('A','pp'),('B','pp_B')]:
                pv=v.pivot(index='sample',columns='seed',values=col).reindex(ids).to_numpy()
                observed=np.isfinite(pv);ls=loss(yy[:,None],pv);ls=np.nan_to_num(ls)
                if layer=='A':
                    vals=(W@ls)/(W@observed.astype(float))
                    pt=ls.sum(0)/observed.sum(0)
                else:
                    vals=np.zeros((B,pv.shape[1]));pt=np.zeros(pv.shape[1])
                    for k in (0,1):
                        keep=(yy==k)[:,None]
                        vals+=.5*(W@(ls*keep))/(W@(observed*keep).astype(float))
                        pt+=.5*(ls*keep).sum(0)/(observed*keep).sum(0)
                boot[split,layer]=np.mean(vals,axis=1);point[split,layer]=pt.mean()
                if split=='OOD':
                    assert np.isfinite(pv).all() and pv.shape[1]==10
                    p=pv.mean(1);weights=W.copy()
                    if layer=='B':
                        for k in (0,1):
                            ix=yy==k;weights[:,ix]*=(.5/weights[:,ix].sum(1))[:,None]
                    est=offset_boot(yy,p,weights);w0=np.ones((1,len(yy)))
                    if layer=='B':
                        for k in (0,1):w0[:,yy==k]*=.5/(yy==k).sum()
                    het.append(dict(task=task,held_out=h,method=m,layer=layer,estimand='OOD_complete_10seed_ensemble',n=len(yy),n_subject=len(set(comps)),cal_int=offset_boot(yy,p,w0)[0],**interval(est)))
        for layer in ('A','B'):
            gap.append(dict(task=task,held_out=h,method=m,layer=layer,estimand='mean_of_10_matched_seed_differences',nll_ID=point['ID',layer],nll_OOD=point['OOD',layer],delta=point['OOD',layer]-point['ID',layer],**interval(boot['OOD',layer]-boot['ID',layer])))
        vv=u[u.split=='OOD'];ids=np.sort(vv['sample'].unique());yy=vv.groupby('sample').y.first().reindex(ids).to_numpy()
        W=draws(cm.loc[ids].to_numpy(),B,(task,h,'adaptation'))
        changes={'em':[],'bbse':[]};fallback=0
        for s,v in vv.groupby('seed'):
            v=v.set_index('sample').loc[ids];p=v.pp.to_numpy();pi=v.pi_cal.iloc[0]
            raw=d[(d.held_out==h)&(d.method==m)&(d.seed==s)&(d.split=='CAL_P')]
            a,b=P.fit_logistic(P.logit(raw.p.to_numpy()),raw.y.to_numpy());pc=P.sigm(a+b*P.logit(raw.p.to_numpy()))
            pib=P.bbse_prior(pc,raw.y.to_numpy(),p)
            fallback+=int(np.isnan(pib))
            for name,pit in [('em',P.em_prior(p,pi)),('bbse',pib)]:
                q=p if np.isnan(pit) else P.shift_prior(p,pi,pit)
                changes[name].append(loss(yy,q)-loss(yy,p))
        for name,L in changes.items():
            dl=np.mean(L,axis=0)
            for layer in ('A','Bw'):
                if layer=='A':val=dl.mean();reps=(W@dl)/W.sum(1)
                else:val=sum(.5*dl[yy==k].mean() for k in (0,1));reps=sum(.5*(W[:,yy==k]@dl[yy==k])/W[:,yy==k].sum(1) for k in (0,1))
                adapt.append(dict(task=task,held_out=h,method=m,adaptation=name,layer=layer,delta=val,bbse_fallback_seeds=fallback if name=='bbse' else 0,**interval(reps)))
        print(task,h,m,flush=True)
    for name,rows in [('gap',gap),('heterogeneity',het),('adapt',adapt)]:pd.DataFrame(rows).to_csv(out/f'{name}_{task}.csv',index=False)

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 2000)
