import os,sys,json,subprocess
import pandas as pd,numpy as np
from common import *
def main():
    r=ROOT/'results'
    d=pd.concat([pd.read_csv(r/f'heterogeneity_{t}.csv') for t in TASKS],ignore_index=True)
    d.to_csv(r/'heterogeneity_all.csv',index=False)
    rex=os.environ.get('RSCRIPT',r'C:/Program Files/R/R-4.5.2/bin/Rscript.exe')
    with open(ROOT/'provenance/metafor.log','w',encoding='utf8') as log:subprocess.run([rex,'--vanilla',str(ROOT/'meta_analysis.R'),str(ROOT)],check=True,stdout=log,stderr=subprocess.STDOUT)
if __name__=='__main__':main()
