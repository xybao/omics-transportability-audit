"""Synthetic demonstration only; no clinical inputs or study reproduction claim."""
import argparse,json
from pathlib import Path
import numpy as np
from common import tau2,loss
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    r=dict(data='synthetic only',tau_squared_REML=tau2(np.array([-1,0,1]),np.array([.1,.1,.1]),'REML'),mean_individual_logloss=float(loss(np.array([1,1]),np.array([.1,.9])).mean()),ensemble_logloss=float(loss(np.array([1]),np.array([.5]))[0]))
    assert abs(r['tau_squared_REML']-.9)<1e-7
    (a.output_dir/'demo.json').write_text(json.dumps(r,indent=2));print(r)
if __name__=='__main__':main()
