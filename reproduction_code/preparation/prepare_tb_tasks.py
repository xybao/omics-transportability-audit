"""Reconstruct TB task matrices; requires provider-derived exports obtained separately."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


import argparse
from pathlib import Path
_parser=argparse.ArgumentParser(description="Build TB task matrices from permitted registered expression exports")
_parser.add_argument("--input-dir",required=True)
_parser.add_argument("--output-dir",required=True)
_args=_parser.parse_args()
D=str(Path(_args.input_dir).resolve())
OUT=Path(_args.output_dir).resolve();OUT.mkdir(parents=True,exist_ok=True)

DROP_STUDY = {"GSE34608": "PBMC + Agilent 探针号未映射", "GSE81746": "n=8，探针号混杂",
              "GSETornheim": "特征为 6 万转录本、取值异常，且仅训练池用", "GSE107731": "n=6，地区缺失"}
COUNT_LIKE = {"GSE101705": "cpm", "GSE152218": "cpm", "GSE84076": "log1p"}   # 其余已是 log2 尺度
# Berry 2010 伦敦系列并成一个组：GSE19435 的 12 个对照与 GSE19444 的 12 个对照是同一批芯片（谱相关>0.98），
# GSE19435 的患者来自 GSE19439 所在的第一批伦敦队列，GSE107991 是 GSE19444 同一批人的 RNA-seq 重测。
GROUP = {"GSE19444": "UK_Berry_London", "GSE107991": "UK_Berry_London", "GSE19435": "UK_Berry_London", "GSE19439": "UK_Berry_London",
         "GSE19442": "ZA_Berry", "GSE107992": "ZA_Berry"}
COUNTRY_FIX = {"GSE42825": "UK", "GSE42826": "UK(+FR)", "GSE42830": "UK", "GSE83456": "UK", "GSEBruno": "India",
               "GSE19435": "UK", "GSE19439": "UK", "GSE19444": "UK", "GSE107991": "UK", "GSE29536": "UK", "GSE107994": "UK"}

meta = pd.read_parquet(f"{D}/tb_meta.parquet")
meta = meta[meta.in_expr.astype(str).str.lower() == "true"].copy()
meta = meta[~meta.study.isin(DROP_STUDY)]
n0 = len(meta)

# ---------- 样本级过滤 ----------
t = meta.MeasurementTime.fillna("NA").astype(str)
keep = pd.Series(True, index=meta.index)
keep &= ~((meta.study == "GSE89403") & (t != "Baseline"))
keep &= ~((meta.study == "GSE107994") & (t != "Baseline"))
keep &= ~((meta.study == "GSE19435") & (t != "0_months"))
keep &= ~((meta.study == "GSE56153") & ~t.isin(["0 weeks", "NA"]))
keep &= ~((meta.study == "GSE40553") & ~meta.Treatment.fillna("").isin(["Pre-treatment", "Untreated"]))
if "Outlier" in meta: keep &= ~(meta.Outlier.fillna("No").astype(str) == "Yes")
keep &= ~(meta.HIVStatus.fillna("NA").astype(str).str.lower() == "positive")          # 主分析只要 HIV 阴性/未报告
keep &= meta.TBStatus.isin(["PTB", "LTBI", "Control", "OD"])
keep &= ~((meta.study == "GSE19435") & (meta.TBStatus == "Control"))   # 与 GSE19444 的对照重复
meta = meta[keep].copy()
# 每个患者只留一个样本（有 PatientID 的研究）
has_pid = meta.PatientID.notna()
meta = pd.concat([meta[~has_pid], meta[has_pid].drop_duplicates(["study", "PatientID"])])
meta["cohort_group"] = meta.study.map(GROUP).fillna(meta.study)
meta["country"] = meta.study.map(COUNTRY_FIX).fillna(meta.GeographicalRegion).replace({"London": "UK", "South India": "India"})
print(f"样本：导出 {n0} → 清洗后 {len(meta)}")
print(meta.groupby(["study", "TBStatus"]).size().unstack(fill_value=0).to_string())

# ---------- 读表达、统一到 log2 尺度 ----------
expr = {}
for s in meta.study.unique():
    e = pd.read_parquet(f"{D}/expr_{s}.parquet").set_index("gene")
    e = e.loc[:, [c for c in e.columns if c in set(meta.sample_id[meta.study == s])]].astype(float)
    if COUNT_LIKE.get(s) == "cpm": e = np.log2(e / e.sum(0) * 1e6 + 1)
    elif COUNT_LIKE.get(s) == "log1p": e = np.log2(e + 1)
    e = e[~e.index.duplicated()]
    e = e[~e.index.str.match(r"^\d+$|^\d+-[A-Z][a-z]{2}$|^[A-Z][a-z]{2}-\d+$", na=False)]   # Excel 把基因名改成日期/序号的残留
    expr[s] = e

# ---------- 任务矩阵：基因交集 ----------
ONLY = os.environ.get("ONLY_TASKS", "ltbi,ctrl,od").split(",")
for task, neg in (("ltbi", "LTBI"), ("ctrl", "Control"), ("od", "OD")):
    if task not in ONLY: continue
    mt = meta[meta.TBStatus.isin(["PTB", neg])]
    cnt = mt.groupby("study").TBStatus.value_counts().unstack(fill_value=0)
    ok = cnt.index[(cnt.get("PTB", 0) > 0) & (cnt.get(neg, 0) > 0)]
    if task in ("ctrl", "od"): ok = [s for s in ok if s != "GSEBruno"] + []          # GSEBruno 仅 7515 个基因，会拖垮交集 → 单独处理
    mt = mt[mt.study.isin(ok)]
    genes = None
    for s in ok: genes = set(expr[s].index) if genes is None else genes & set(expr[s].index)
    genes = sorted(genes)
    X = pd.concat([expr[s].loc[genes, mt.sample_id[mt.study == s]].T for s in ok])
    X = X.dropna(axis=1)
    X.index.name = "sample_id"
    mt = mt.set_index("sample_id").loc[X.index].reset_index()
    X.reset_index().to_parquet(OUT/f"task_{task}_X.parquet"); mt.to_parquet(OUT/f"task_{task}_meta.parquet")
    print(f"\n[task {task}] 研究 {len(ok)} 个 / 队列组 {mt.cohort_group.nunique()} 个 / 国家 {mt.country.nunique()} 个 / 样本 {len(mt)} / 基因交集 {X.shape[1]}")
    print(mt.groupby(["cohort_group", "TBStatus"]).size().unstack(fill_value=0).to_string())

