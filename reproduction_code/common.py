from pathlib import Path
import hashlib, json
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize_scalar, brentq
ROOT=Path(__file__).resolve().parent
TASKS=('crc4','tb_ltbi','tb_ctrl','tb_od')
MODELS=('lasso','rf','gbdt','mlp')
def seed(*key): return int.from_bytes(hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).digest()[:4],'little')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def loss(y,p):
    p=np.clip(p,1e-4,1-1e-4)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))
def draws(comp,B,key):
    un,inv=np.unique(comp,return_inverse=True)
    counts=np.random.default_rng(seed(*key)).multinomial(len(un),np.full(len(un),1/len(un)),size=B)
    return counts[:,inv].astype(float)
def interval(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    if not len(a): return dict(lo=np.nan,hi=np.nan,se=np.nan,n_boot=0)
    return dict(lo=float(np.quantile(a,.025)),hi=float(np.quantile(a,.975)),se=float(np.std(a,ddof=1)),n_boot=len(a))
def tau2(y,v,method):
    y=np.asarray(y,float); v=np.asarray(v,float); k=len(y)
    if k<2 or not np.isfinite(v).all() or np.any(v<=0): raise ValueError('Invalid random effects inputs')
    w=1/v; mu=np.average(y,weights=w); q=np.sum(w*(y-mu)**2)
    if method=='DL': return max(0,(q-k+1)/(w.sum()-(w*w).sum()/w.sum()))
    def Q(t):
        w=1/(v+t); return np.sum(w*(y-np.average(y,weights=w))**2)-(k-1)
    upper=max(1,float(np.var(y,ddof=1))*10)
    if method=='PM':
        if Q(0)<=0:return 0.
        while Q(upper)>0: upper*=2
        return brentq(Q,0,upper,xtol=1e-12)
    def nll(t):
        w=1/(v+t); mu=np.average(y,weights=w)
        return .5*(np.log(v+t).sum()+np.log(w.sum())+np.sum(w*(y-mu)**2))
    opt=minimize_scalar(nll,bounds=(0,upper),method='bounded',options={'xatol':1e-12})
    return 0. if nll(0)<=opt.fun else float(opt.x)
def offset_boot(y,p,weights):
    z=np.log(np.clip(p,1e-4,1-1e-4)/np.clip(1-p,1e-4,1-1e-4))
    a=np.zeros(weights.shape[0])
    for _ in range(70):
        q=expit(z[None,:]+a[:,None]); grad=np.sum(weights*(y[None,:]-q),axis=1)
        hess=np.sum(weights*q*(1-q),axis=1)+1e-12
        step=np.clip(grad/hess,-1,1); a=np.clip(a+step,-10,10)
        if np.max(abs(step))<1e-8:break
    cases=weights@y;total=weights.sum(axis=1)
    a[(cases<=0)|(cases>=total)|~np.isfinite(total)]=np.nan
    return a
