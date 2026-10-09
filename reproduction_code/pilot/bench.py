"""Scientific model, split or calibration functions used by the evaluation pipeline."""
import os, sys, time, zlib, warnings
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import lightgbm as lgb; HAS_LGB = True
except Exception:
    HAS_LGB = False


# ---------------------------------------------------------------- 任务
def load_crc(version=""):
    """version="" → MetaPhlAn 3（curatedMetagenomicData）；version="4" → MetaPhlAn 4（Piccinno 2025 Zenodo，同一批开发样本）。"""
    X = pd.read_parquet(f"{ROOT}/data_processed/crc{version}_relabund.parquet").set_index("sample_id")
    md = pd.read_parquet(f"{ROOT}/data_processed/crc{version}_meta.parquet").set_index("sample_id").loc[X.index]
    keep = (md.study_name.astype(str) != "ThomasAM_2019_c").values      # 与 YachidaS_2019 重复的 80 个样本
    X, md = X[keep], md[keep]
    y = (md.study_condition.astype(str) == "CRC").astype(int).values
    g = md.study_name.astype(str).values
    raw = X.values.astype(float)
    return dict(name=f"crc{version}", X=np.log10(raw / 100 + 1e-6), present=raw > 0, y=y, g=g, ids=X.index.values)


def rank_int(M):
    """样本内秩 → 正态分位数（逐样本，不用任何队列信息 → inductive，专治跨平台尺度差异）。"""
    from scipy.stats import norm, rankdata
    R = np.apply_along_axis(rankdata, 1, M)
    return norm.ppf((R - 0.5) / M.shape[1])


def load_tb(which):
    X = pd.read_parquet(f"{ROOT}/data_processed/tb/task_{which}_X.parquet").set_index("sample_id")
    md = pd.read_parquet(f"{ROOT}/data_processed/tb/task_{which}_meta.parquet")
    md = md.rename(columns={"index": "sample_id"}) if "sample_id" not in md.columns else md
    md = md.set_index("sample_id").loc[X.index]
    y = (md.TBStatus.astype(str) == "PTB").astype(int).values
    g = md.cohort_group.astype(str).values
    M = X.values.astype(float)
    return dict(name=f"tb_{which}", X=M, Xrank=rank_int(M), present=None, y=y, g=g, batch=md.study.astype(str).values, ids=X.index.values, top_var=2000)


TASKS = {"crc": load_crc, "crc4": lambda: load_crc("4"), "tb_ltbi": lambda: load_tb("ltbi"), "tb_ctrl": lambda: load_tb("ctrl"), "tb_od": lambda: load_tb("od")}


# ---------------------------------------------------------------- 指标
def ece_equal_mass(y, p, n_bins=10):
    order = np.argsort(p); bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                bacc=balanced_accuracy_score(y, p > 0.5), ece=ece_equal_mass(y, p),
                brier=brier_score_loss(y, p), nll=log_loss(y, p), base_rate=float(y.mean()),
                mean_pred=float(p.mean()))


# ---------------------------------------------------------------- 方法
def fit_lasso(Xtr, ytr, gtr, seed):
    Cs = np.logspace(-2.5, 0.5, 8); ng = len(np.unique(gtr))
    folds = list(GroupKFold(n_splits=min(5, ng)).split(Xtr, ytr, gtr))
    score = np.zeros(len(Cs))
    for i, C in enumerate(Cs):
        for a, b in folds:
            if len(np.unique(ytr[b])) < 2: continue
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000, random_state=seed).fit(Xtr[a], ytr[a])
            score[i] += roc_auc_score(ytr[b], m.decision_function(Xtr[b]))
    return LogisticRegression(penalty="l1", solver="liblinear", C=Cs[score.argmax()], max_iter=2000, random_state=seed).fit(Xtr, ytr)


def fit_rf(Xtr, ytr, gtr, seed):
    return RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=1, random_state=seed).fit(Xtr, ytr)


def fit_gbdt(Xtr, ytr, gtr, seed):
    if HAS_LGB:
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, max_depth=3, min_child_samples=10,
                                  colsample_bytree=0.2, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                                  random_state=seed, n_jobs=1, verbose=-1).fit(Xtr, ytr)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, max_features=0.2,
                                          l2_regularization=1.0, random_state=seed).fit(Xtr, ytr)


def fit_mlp(Xtr, ytr, gtr, seed):
    return MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-1, learning_rate_init=1e-3, max_iter=300,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                         random_state=seed).fit(Xtr, ytr)


class _SelectThenTabPFN:
    """训练样本内 ANOVA-F 选 k 个特征 → TabPFN。特征选择只看训练数据。"""
    def __init__(self, k, seed):
        from sklearn.feature_selection import SelectKBest, f_classif
        from tabpfn import TabPFNClassifier
        self.sel = SelectKBest(f_classif, k=k)
        self.clf = TabPFNClassifier(device=os.environ.get("TABPFN_DEVICE", "cuda"), random_state=seed)
    def fit(self, X, y):
        k = min(self.sel.k, X.shape[1]); self.sel.set_params(k=k)
        self.clf.fit(self.sel.fit_transform(X, y), y); return self
    def predict_proba(self, X):
        return self.clf.predict_proba(self.sel.transform(X))


def fit_tabpfn(Xtr, ytr, gtr, seed):
    return _SelectThenTabPFN(int(os.environ.get("TABPFN_K", "500")), seed).fit(Xtr, ytr)


METHODS = {"lasso": fit_lasso, "rf": fit_rf, "gbdt": fit_gbdt, "mlp": fit_mlp, "tabpfn": fit_tabpfn}


def zscore_by_group(M, groups):
    out = np.empty_like(M)
    for c in np.unique(groups):
        i = groups == c; mu, sd = M[i].mean(0), M[i].std(0)
        out[i] = (M[i] - mu) / np.where(sd > 0, sd, 1.0)
    return out


# ---------------------------------------------------------------- 一个评测单元
def draw_id_set(y, g, pool, n_target, prev_target, rng):
    """从训练池按队列比例分层抽 ID 测试集：规模≈n_target，病例比例≈prev_target；每队列至多抽走 40%。"""
    n_target = min(n_target, int(0.3 * len(pool)))
    n_pos = int(round(n_target * prev_target)); n_neg = n_target - n_pos
    chosen = []
    for label, k in ((1, n_pos), (0, n_neg)):
        cand = pool[y[pool] == label]
        cohorts, counts = np.unique(g[cand], return_counts=True)
        quota = np.floor(k * counts / counts.sum()).astype(int)
        quota = np.minimum(quota, (0.4 * counts).astype(int))
        for c, q in zip(cohorts, quota):
            if q > 0: chosen.append(rng.choice(cand[g[cand] == c], q, replace=False))
    return np.concatenate(chosen)


def run_unit(T, held, seed):
    X, P, y, g = T["X"], T["present"], T["y"], T["g"]
    rng = np.random.default_rng(1000 * seed + zlib.crc32(held.encode()) % 997)
    ood = np.where(g == held)[0]; pool = np.where(g != held)[0]
    idt = draw_id_set(y, g, pool, len(ood), y[ood].mean(), rng)
    tr = np.setdiff1d(pool, idt)
    preps = {}
    if P is not None:                                              # 微生物组：按训练样本的检出率筛特征
        feat = P[tr].mean(0) >= 0.05; Xf = X[:, feat]
        preps["global"] = StandardScaler().fit(Xf[tr]).transform(Xf)       # inductive
        preps["percohort"] = zscore_by_group(Xf, g)                        # transductive
    else:                                                          # 转录组：按训练样本方差取前 top_var 个基因
        k = T.get("top_var", 2000)
        feat = np.zeros(X.shape[1], bool); feat[np.argsort(-X[tr].var(0))[:k]] = True
        Xf = X[:, feat]
        preps["global"] = StandardScaler().fit(Xf[tr]).transform(Xf)       # inductive，原始 log2 尺度
        fr = np.zeros(X.shape[1], bool); fr[np.argsort(-T["Xrank"][tr].var(0))[:k]] = True
        Xr = T["Xrank"][:, fr]
        preps["rank"] = StandardScaler().fit(Xr[tr]).transform(Xr)         # inductive，样本内秩变换
        preps["percohort"] = zscore_by_group(Xf, g)                        # transductive
    rows = []
    for mname, fit in METHODS.items():
        for prep, M in preps.items():
            t0 = time.time()
            model = fit(M[tr], y[tr], g[tr], seed)
            for split, idx in (("ID", idt), ("OOD", ood)):
                p = model.predict_proba(M[idx])[:, 1]
                rows.append(dict(task=T["name"], held_out=held, seed=seed, method=mname, prep=prep, split=split,
                                 n_train=len(tr), n_test=len(idx), n_feat=int(feat.sum()),
                                 secs=round(time.time() - t0, 1), **metrics(y[idx], p)))
    return rows


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "crc"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    T = TASKS[task]()
    cohorts = [c for c in np.unique(T["g"]) if min((T["y"][T["g"] == c] == 1).sum(), (T["y"][T["g"] == c] == 0).sum()) >= 10]
    print(f"task={task} cohorts={len(cohorts)} n={len(T['y'])} seeds={n_seeds} lightgbm={HAS_LGB}", flush=True)
    jobs = [(c, s) for c in cohorts for s in range(n_seeds)]
    out = Parallel(n_jobs=min(24, os.cpu_count() - 2), verbose=5)(delayed(run_unit)(T, c, s) for c, s in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(f"{ROOT}/runs", exist_ok=True)
    path = f"{ROOT}/runs/bench_{task}_week1.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    piv = df.groupby(["method", "prep", "split"]).auroc.mean().unstack("split")
    piv["gap"] = piv["ID"] - piv["OOD"]
    print(piv.round(3).to_string()); print("saved", path)
