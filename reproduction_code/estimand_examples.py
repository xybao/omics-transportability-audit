"""Exact synthetic examples; no patient data or empirical adaptation claim."""
import sys,json
import numpy as np
from common import ROOT,loss
sys.path.insert(0,str(ROOT/'pilot'))
import posthoc3 as P
def main():
    single=float(loss(np.array([1,1]),np.array([.1,.9])).mean())
    ensemble=float(loss(np.array([1,1]),np.array([.5,.5])).mean())
    p=np.linspace(.01,.99,99)
    err=float(np.max(abs(P.shift_prior(P.shift_prior(p,.5,.2),.2,.8)-P.shift_prior(p,.5,.8))))
    example=dict(mean_single_member_loss=single,complete_ensemble_loss=ensemble,apparent_target_minus_source=ensemble-single,prior_composition_max_error=err,conformal_n20_alpha0p1=float(P.conformal_quantile(np.arange(1,21),.1)))
    assert ensemble<single and err<1e-12 and example['conformal_n20_alpha0p1']==19
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/estimand_examples.json').write_text(json.dumps(example,indent=2),encoding='utf8')
if __name__=='__main__':main()
