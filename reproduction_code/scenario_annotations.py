"""Propagate empirical-boundary flags to every scenario output without changing values."""
import pandas as pd,numpy as np,json
from common import *
METRICS=('TP','FN','FP','TN','PPV','risk_after_negative')
def main():
    summaries=[]
    for t in TASKS:
        p=ROOT/'results'/f'scenario_{t}.csv';d=pd.read_csv(p)
        th=pd.read_csv(ROOT/'results'/f'threshold_{t}.csv')
        keys=['task','held_out','split','working']
        d=d.drop(columns=[c for c in d if c.startswith('source_')],errors='ignore').merge(th[keys+['sens_bootstrap_degenerate','spec_bootstrap_degenerate']].rename(columns={'sens_bootstrap_degenerate':'source_sens_degenerate','spec_bootstrap_degenerate':'source_spec_degenerate'}),on=keys,validate='many_to_one')
        for metric in METRICS:
            d[metric+'_bootstrap_degenerate']=np.isclose(d[metric+'_lo'],d[metric+'_hi'],rtol=0,atol=1e-12)
            d[metric+'_interval_status']=np.where(d[metric+'_bootstrap_degenerate'],'NE—degenerate empirical interval','conditional empirical interval')
            summaries.append(dict(task=t,metric=metric,degenerate_metric_rows=int(d[metric+'_bootstrap_degenerate'].sum())))
        d['interpretation']='Zero-width intervals do not establish zero population uncertainty or zero missed-case risk.'
        d.to_csv(p,index=False)
    totals=pd.DataFrame(summaries).groupby('metric').degenerate_metric_rows.sum().to_dict()
    assert totals=={'TP':16,'FN':16,'FP':4,'TN':4,'PPV':4,'risk_after_negative':16},totals
    print('Scenario boundary propagation:',totals)
if __name__=='__main__':main()
