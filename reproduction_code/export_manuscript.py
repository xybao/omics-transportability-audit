"""Export manuscript tables and figures from the completed scientific analysis.
No model fitting or changes to numerical estimands are performed here.
"""
import os
os.environ['MPLBACKEND']='Agg'
import json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from pathlib import Path
import argparse
from common import TASKS, MODELS
ROOT=None
OUT=FIG=TABLE=None
LABEL={'crc4':'CRC vs controls','tb_ltbi':'TB vs LTBI','tb_ctrl':'TB vs healthy controls','tb_od':'TB vs other diseases'}
COLOR={'lasso':'#2166ac','rf':'#1b9e77','gbdt':'#d95f02','mlp':'#7570b3'}
def read(name):return pd.concat([pd.read_csv(ROOT/'results'/f'{name}_{t}.csv') for t in TASKS],ignore_index=True)
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
    plt.close(fig)
def table(d,name):
    d.to_csv(TABLE/(name+'.csv'),index=False)
    lines=['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    lines += ['| '+' | '.join(str(x).replace('|','/').replace('\r', '').replace('\n', '<br>') for x in row)+' |' for row in d.round(4).itertuples(index=False,name=None)]
    (TABLE/(name+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf8')
def main():
    global ROOT, OUT, FIG, TABLE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--omit-source-record-tables',action='store_true',help='Omit Tables S20 and S21, whose source-record inputs are not redistributed in the public package.')
    a=ap.parse_args();ROOT=a.analysis_dir.resolve();OUT=a.output_dir.resolve()
    if OUT.exists():raise SystemExit('Choose a fresh output directory')
    FIG=OUT/'figures';TABLE=OUT/'tables'
    FIG.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'pred_audit_v3'})
    gaps=read('gap');mets=read('metrics_seed');adapt=read('adapt');budget=read('budget');th=read('threshold');sets=read('sets');sc=read('scenario')
    meta=pd.read_csv(ROOT/'results/meta_metafor.csv');counts=[]
    cohort_counts=pd.read_csv(ROOT/'results/cohort_definitions.csv')
    for t in TASKS:
        md=cohort_counts[cohort_counts.task==t]
        counts.append(dict(task=t,records=int(md.records.sum()),subject_components=int(md.components.sum()),heldout_groups=int(md.role.eq('held_out_and_source').sum()),source_only_groups=int(md.role.eq('source_only').sum())))
    table(pd.DataFrame(counts).assign(task=lambda d:d.task.map({'crc4': 'Colorectal cancer versus selected controls', 'tb_ltbi': 'TB-labelled cases versus latent tuberculosis infection', 'tb_ctrl': 'TB-labelled cases versus healthy controls', 'tb_od': 'TB-labelled cases versus other diseases'})).rename(columns={'task':'Medical task','records':'Measurement records','subject_components':'Subject components','heldout_groups':'OOD groups','source_only_groups':'Source-only groups'}),'Table1_tasks');table(pd.read_csv(ROOT/'results/cohort_definitions.csv'),'TableS1_cohorts')
    summary=gaps.groupby(['task','method','layer']).agg(median_delta=('delta','median'),min_delta=('delta','min'),max_delta=('delta','max'),n_cohorts=('held_out','nunique')).reset_index()
    summary['ci_above_zero']=[int(((gaps.task==r.task)&(gaps.method==r.method)&(gaps.layer==r.layer)&(gaps.lo>0)).sum()) for r in summary.itertuples()]
    table(summary,'Table2_matched_gaps');table(gaps,'TableS2_matched_gaps_all');table(meta,'Table3_heterogeneity')
    table(mets.groupby(['task','held_out','method','split','layer'])[['auroc','nll','brier','cal_int','cal_slope']].mean().reset_index(),'TableS3_seed_mean_metrics')
    table(adapt,'TableS4_adaptation');table(budget,'TableS5_label_budget');table(th,'TableS6_thresholds');table(sc,'TableS7_scenarios');table(sets,'TableS8_prediction_sets')
    for source,name in [('E2_public_inputs','TableS10_E2_cohort_inputs'),('boundary_audit','TableS11_boundary_audit'),('single_program_intervals','TableS12_single_program_intervals'),('budget_label_composition','TableS13_label_composition'),('budget_exclusions_all','TableS14_budget_exclusions'),('cross_task_overlap','TableS15_cross_task_overlap'),('conformal_record_selection','TableS16_conformal_record_selection')]:table(pd.read_csv(ROOT/'results'/f'{source}.csv'),name)
    # 1: matched gap, one panel per task, each point a cohort; no pseudo-independent seed bars.
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,t in zip(axes.flat,TASKS):
        u=gaps[(gaps.task==t)&(gaps.layer=='A')]
        for j,m in enumerate(MODELS):
            v=u[u.method==m].sort_values('held_out');ys=j+np.linspace(-.26,.26,len(v))
            ax.errorbar(v.delta,ys,xerr=[np.maximum(0,v.delta-v.lo),np.maximum(0,v.hi-v.delta)],fmt='o',ms=3,color=COLOR[m],alpha=.75,capsize=2)
        ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(4),yticklabels=[m.upper() for m in MODELS],xlabel='OOD − ID log-loss (layer A)',title=LABEL[t])
    fig.tight_layout();save(fig,'Figure1_matched_transport')
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ax,t in zip(axes.flat,TASKS):
        u=meta[(meta.task==t)&(meta.layer=='B')&(meta.estimator=='REML')].set_index('method').loc[list(MODELS)]
        ax.errorbar(u.tau,range(4),xerr=[u.tau-u.tau_lo,u.tau_hi-u.tau],fmt='o',capsize=3,color='#2166ac')
        ax.set(yticks=range(4),yticklabels=[m.upper() for m in MODELS],xlabel='Between-cohort SD, tau (log-odds)',title=LABEL[t],xlim=(0,3.5))
    fig.tight_layout();save(fig,'Figure2_heterogeneity')
    fig,axes=plt.subplots(2,2,figsize=(13,11))
    for ax,t in zip(axes.flat,TASKS):
        u=adapt[(adapt.task==t)&(adapt.layer=='A')&(adapt.method=='lasso')];hs=sorted(u.held_out.unique())
        for offset,nm,col in [(-.13,'em','#d95f02'),(.13,'bbse','#2166ac')]:
            v=u[u.adaptation==nm].set_index('held_out').loc[hs]
            ax.errorbar(v.delta,np.arange(len(hs))+offset,xerr=[np.maximum(0,v.delta-v.lo),np.maximum(0,v.hi-v.delta)],fmt='o',ms=3,capsize=2,color=col,label=nm.upper())
        ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(len(hs)),yticklabels=hs,xlabel='Adapted − source-calibrated log-loss',title=LABEL[t]);ax.tick_params(axis='y',labelsize=8);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'FigureS2_adaptation')
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ax,t in zip(axes.flat,TASKS):
        u=budget[(budget.task==t)&(budget.layer=='A')&(budget.method=='lasso')]
        for h,v in u.groupby('held_out'):ax.plot(v.k,v.delta,'o-',lw=.8,ms=3,alpha=.4,color='#2166ac')
        v=u.groupby('k').delta.median();ax.plot(v.index,v.values,'o-',lw=2.5,color='black',label='Median of cohort means')
        ax.axhline(0,color='black',lw=.8);ax.set(title=LABEL[t]+f' (n={u.held_out.nunique()}; max fallback={u.fallback_rate.max():.1%})',xlabel='Labelled participant components',ylabel='Update − no update log-loss',xticks=[10,20,40]);ax.legend(fontsize=7)
    fig.tight_layout();save(fig,'FigureS3_label_budget')
    fig,axes=plt.subplots(4,2,figsize=(13,17),sharex=True,gridspec_kw={'width_ratios':[1.3,1]})
    for row,t in enumerate(TASKS):
        u=th[(th.task==t)&(th.split=='OOD')&(th.working==.9)].sort_values('held_out')
        v=th[(th.task==t)&(th.split=='ID')&(th.working==.9)].set_index('held_out').loc[u.held_out]
        assert len(u)==len(v) and set(u.method)=={'lasso'} and set(v.method)=={'lasso'}
        ys=np.arange(len(u));ax,bx=axes[row]
        for j,(src,dst) in enumerate(zip(v.sens,u.sens)):
            ax.plot([src,dst],[j-.10,j+.10],color='#999999',lw=.9,zorder=1)
        for data,offset,col,label,fill in [(v,-.10,'#555555','Source evaluation (ID)','white'),(u,.10,'#2166ac','Target evaluation (OOD)','#2166ac')]:
            ax.errorbar(data.sens,ys+offset,xerr=[np.maximum(0,data.sens-data.sens_lo),np.maximum(0,data.sens_hi-data.sens)],fmt='o',ms=4,mfc=fill,mec=col,color=col,capsize=2,label=label)
            flag=data.sens_bootstrap_degenerate.to_numpy(dtype=bool)
            ax.scatter(data.sens.to_numpy()[flag],(ys+offset)[flag],marker='x',color='black',s=45,zorder=5)
        bx.errorbar(u.spec,ys,xerr=[np.maximum(0,u.spec-u.spec_lo),np.maximum(0,u.spec_hi-u.spec)],fmt='o',ms=4,color='#d95f02',capsize=2,label='Target specificity')
        flag=u.spec_bootstrap_degenerate.to_numpy(dtype=bool)
        bx.scatter(u.spec.to_numpy()[flag],ys[flag],marker='x',color='black',s=45,zorder=5)
        ax.axvline(.9,ls=':',lw=.9,color='gray',label='Source selection target (0.90)')
        ax.set(yticks=ys,yticklabels=u.held_out,title=LABEL[t],xlabel='Sensitivity',xlim=(-.02,1.02))
        bx.set(yticks=ys,yticklabels=u.held_out,title='Target specificity',xlabel='Specificity',xlim=(-.02,1.02))
        for panel in (ax,bx):panel.tick_params(axis='y',labelsize=8);panel.invert_yaxis()
        if row==0:ax.legend(fontsize=8,loc='lower left')
    fig.tight_layout();save(fig,'Figure3_thresholds')
    fig,axes=plt.subplots(4,2,figsize=(15,20))
    for row,t in enumerate(TASKS):
        u=sets[(sets.task==t)&(sets.method=='lasso')].sort_values('held_out');ys=np.arange(len(u));left=np.zeros(len(u))
        ax,bx=axes[row]
        for nm,col in [('empty','#d95f02'),('single','#2166ac'),('both','#bdbdbd')]:
            ax.barh(ys,u[nm],left=left,label=nm,color=col);left+=u[nm].to_numpy()
        ax.set(yticks=ys,yticklabels=u.held_out,xlim=(0,1),title=LABEL[t],xlabel='Mean fraction of prediction sets')
        for offset,nm,col,label in [(-.2,'covered','#333333','Overall'),(0,'coverage_class0','#2166ac','Controls'),(.2,'coverage_class1','#d95f02','Cases')]:
            bx.errorbar(u[nm],ys+offset,xerr=[np.maximum(0,u[nm]-u[nm+'_lo']),np.maximum(0,u[nm+'_hi']-u[nm])],fmt='o',ms=3,capsize=2,color=col,label=label)
            flags=u[nm+'_bootstrap_degenerate'].astype(bool);bx.scatter(u.loc[flags,nm],(ys+offset)[flags],marker='x',color='black',s=45)
        labels=[f"{r.held_out} (case/control subjects {int(r.n_class1_subjects)}/{int(r.n_class0_subjects)})" for r in u.itertuples()]
        bx.set(yticks=ys,yticklabels=labels,xlim=(-.02,1.02),xlabel='Empirical coverage; crosses mark point intervals',title='Overall and class-conditional coverage')
        bx.axvline(.9,color='gray',ls=':');ax.tick_params(axis='y',labelsize=7);bx.tick_params(axis='y',labelsize=7)
    axes[0,0].legend(loc='lower left',bbox_to_anchor=(0,1.12),ncol=3,fontsize=9)
    axes[0,1].legend(loc='lower left',bbox_to_anchor=(0,1.12),ncol=3,fontsize=9)
    fig.tight_layout();save(fig,'Figure4_prediction_sets')
    # Tables and JSON are the sole numeric sources for the manuscript builder.
    data={'counts':counts,'gap_A':summary[summary.layer=='A'].to_dict('records'),'tau_B_REML':meta[(meta.layer=='B')&(meta.estimator=='REML')].to_dict('records'),
          'label_lasso':budget[(budget.method=='lasso')&(budget.layer=='A')].groupby(['task','k']).delta.median().reset_index().to_dict('records'),
          'threshold_lasso':th[(th.split=='OOD')&(th.working==.9)].groupby('task').agg(min_sens=('sens','min'),max_sens=('sens','max'),min_spec=('spec','min'),max_spec=('spec','max')).reset_index().to_dict('records')}
    (OUT/'paper_numbers.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    for source,name in [('clinical_source_audit','TableS17_clinical_sources'),('OD_composition','TableS18_curated_categories_and_source_evidence'),('exploratory_diagnostic_groups','TableS19_diagnostic_groups'),('GSE89403_record_mapping','TableS20_GSE89403_record_mapping'),('GSE73408_record_mapping','TableS21_GSE73408_record_mapping'),('target_summary_influence','TableS22_target_summary_influence'),('ranking_threshold_example','TableS23_ranking_threshold_example')]:
        if a.omit_source_record_tables and source in {'GSE89403_record_mapping','GSE73408_record_mapping'}:
            print('Not redistributed: '+name+'; see SOURCE_RECORDS.md')
            continue
        table(pd.read_csv(ROOT/'results'/f'{source}.csv'),name)
    view=sc[['task','held_out','split','working','assumed_prevalence']].copy()
    for metric in ['TP','FN','FP','TN','PPV','risk_after_negative']:
        cells=[]
        for _,r in sc.iterrows():
            lo,hi=r[metric+'_lo'],r[metric+'_hi'];deg=bool(r[metric+'_bootstrap_degenerate']);places=3
            while not deg and round(lo,places)==round(hi,places) and places<12:places+=1
            cells.append(f'{r[metric]:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]'+(' NE' if deg else ''))
        view[metric]=cells
    table(view,'TableS7_scenarios_reader')
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(FIG/'FigureS1_cohort_forest.pdf',metadata={'CreationDate':None,'ModDate':None}) as pdf:
        for task in TASKS:
            d=read('gap');d=d[(d.task==task)&(d.layer=='A')]
            fig,axes=plt.subplots(2,2,figsize=(12,9))
            for ax,m in zip(axes.flat,MODELS):
                u=d[d.method==m].sort_values('held_out');y=np.arange(len(u))
                ax.errorbar(u.delta,y,xerr=[np.maximum(0,u.delta-u.lo),np.maximum(0,u.hi-u.delta)],fmt='o',ms=3,capsize=2)
                ax.set(yticks=y,yticklabels=u.held_out,title=LABEL[task]+' / '+m.upper(),xlabel='Matched OOD minus ID log-loss (layer A)')
                ax.tick_params(axis='y',labelsize=7);ax.axvline(0,color='gray',lw=.7)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print('Exported four main figures, three supplementary figures and numerical tables')
if __name__=='__main__':main()
