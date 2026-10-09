"""Scientific model, split or calibration functions used by the evaluation pipeline."""
import os, sys, zlib, time, warnings
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

ROOT = B.ROOT


class Filter(BaseEstimator, TransformerMixin):
    """输入 [basis | values] 两块等宽矩阵：用 basis 在**当前训练折**上定特征掩码，输出 values 的被选列。"""
    def __init__(self, kind="prevalence", thr=0.05, k=2000): self.kind, self.thr, self.k = kind, thr, k
    def fit(self, X, y=None):
        p = X.shape[1] // 2; basis = X[:, :p]
        if self.kind == "prevalence": self.mask_ = (basis > -5.99).mean(0) >= self.thr      # log10(丰度+1e-6) > -6 即检出
        else:
            self.mask_ = np.zeros(p, bool); self.mask_[np.argsort(-basis.var(0))[: min(self.k, p)]] = True
        return self
    def transform(self, X): p = X.shape[1] // 2; return X[:, p:][:, self.mask_]


def pipe(T, clf, scale=True):
    f = Filter("prevalence", 0.05) if T["present"] is not None else Filter("variance", k=T.get("top_var", 2000))
    steps = [("filter", f)] + ([("scale", StandardScaler())] if scale else []) + [("clf", clf)]
    return Pipeline(steps)


def fit_method(name, T, M, tr, y, g, seed, scale):
    if name == "lasso":
        base = pipe(T, LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=seed), scale)
        ng = len(np.unique(g[tr]))
        gs = GridSearchCV(base, {"clf__C": np.logspace(-2.5, 0.5, 8)}, scoring="roc_auc",
                          cv=GroupKFold(n_splits=min(5, ng)), n_jobs=1, refit=True)
        return gs.fit(M[tr], y[tr], groups=g[tr])
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=1, random_state=seed)
    elif name == "gbdt":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, max_depth=3, min_child_samples=10,
                                 colsample_bytree=0.2, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                                 random_state=seed, n_jobs=1, verbose=-1)
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-1, learning_rate_init=1e-3, max_iter=150,
                            early_stopping=False, random_state=seed)                       # 固定训练预算
    elif name == "tabpfn":
        clf = B._SelectThenTabPFN(int(os.environ.get("TABPFN_K", "500")), seed)
    else:
        raise ValueError(name)
    return pipe(T, clf, scale).fit(M[tr], y[tr])


_COMP_CACHE = {}


def components_for(task):
    """受试者连通分量（P0-3）：同人的多条记录必须整体落在同一个集合里。
    由 pilot/subject_components.py 预先生成；找不到就返回 None（退回按样本行切分，并在日志里说明）。"""
    if task in _COMP_CACHE: return _COMP_CACHE[task]
    f = f"{ROOT}/inputs/subject_components_{task}.csv"
    comp = None
    if os.path.exists(f):
        d = pd.read_csv(f).sort_values("sample")
        comp = d.component_id.values
    if comp is None: raise FileNotFoundError(f)
    if not np.array_equal(d['sample'].values, np.arange(len(d))): raise ValueError('Component index mismatch')
    if (d.groupby('component_id').y.nunique() > 1).any(): raise ValueError('Conflicting subject labels')
    if (d.groupby('component_id').cohort_group.nunique() > 1).any(): raise ValueError('Subject spans cohort groups')
    _COMP_CACHE[task] = comp
    return comp


def source_split(y, g, pool, rng, fr=(0.60, 0.15, 0.10, 0.15), comp=None):
    """源域四分。comp 给定时按**受试者连通分量**整体分配，避免同一个人的多条记录跨集合。"""
    if comp is None: raise ValueError('Subject component mapping is mandatory in v3')
    if len(comp) != len(y): raise ValueError('Component vector must match the current task rows')
    for c in np.unique(comp):
        ix = comp == c
        if len(np.unique(y[ix])) != 1 or len(np.unique(g[ix])) != 1:
            raise ValueError('Component spans labels or cohorts')
    parts = {k: [] for k in ("TRAIN", "CAL_P", "CAL_C", "ID")}
    for c in np.unique(g[pool]):
        for lab in (0, 1):
            ix = pool[(g[pool] == c) & (y[pool] == lab)]
            if comp is None:
                units = [np.array([i]) for i in ix]
            else:                                   # 按分量成组，整组分配
                order = {}
                for i in ix: order.setdefault(comp[i], []).append(i)
                units = [np.array(v) for v in order.values()]
            units = [units[j] for j in rng.permutation(len(units))]
            sizes = np.array([len(u) for u in units]); n = sizes.sum()
            if n == 0: continue
            cum = np.cumsum(sizes)
            a = int(np.searchsorted(cum, fr[0] * n)) + 1
            b = int(np.searchsorted(cum, (fr[0] + fr[1]) * n)) + 1
            d = int(np.searchsorted(cum, (fr[0] + fr[1] + fr[2]) * n)) + 1
            a, b, d = min(a, len(units)), min(b, len(units)), min(d, len(units))
            for k, sl in zip(parts, (units[:a], units[a:b], units[b:d], units[d:])):
                if sl: parts[k].append(np.concatenate(sl))
    return {k: (np.concatenate(v) if v else np.array([], int)) for k, v in parts.items()}


def zscore_train_stats(V, g, tr, ood):
    """按批次（研究×平台）标准化：源批次用其 TRAIN 样本的均值/方差；目标批次用自身全部样本（整批传导）。
    划分仍按队列组；标准化按批次——同一队列组里芯片与 RNA-seq 不混在一起。"""
    out = np.empty_like(V); trm = np.zeros(len(g), bool); trm[tr] = True
    for c in np.unique(g):
        i = g == c; ref = i & trm if (i & trm).sum() >= 5 else i
        mu, sd = V[ref].mean(0), V[ref].std(0)
        out[i] = (V[i] - mu) / np.where(sd > 0, sd, 1.0)
    return out


def run_unit(T, held, seed, methods):
    X, y, g = T["X"], T["y"], T["g"]
    rng = np.random.default_rng(1000 * seed + zlib.crc32(held.encode()) % 997)
    ood = np.where(g == held)[0]; pool = np.where(g != held)[0]
    full = os.environ.get("FULL_SOURCE") == "1"            # 标准全源训练的留一队列：不切 CAL/ID，只评留出队列
    sp = source_split(y, g, pool, rng, (1.0, 0.0, 0.0, 0.0) if full else (0.60, 0.15, 0.10, 0.15),
                      comp=components_for(T["name"])); tr = sp["TRAIN"]
    def build(prep):                                            # 逐个生成、用完即弃，单精度，控制内存
        if prep == "global": return np.hstack([X, X]).astype(np.float32), True
        if prep == "rank": return np.hstack([T["Xrank"], T["Xrank"]]).astype(np.float32), True
        return np.hstack([X, zscore_train_stats(X, T.get("batch", g), tr, ood)]).astype(np.float32), False
    preps = ["global"]  # v3 active inference; historical rank/batch paths excluded
    out = []
    for prep in preps:
        M, scale = build(prep)
        for m in methods:
            model = fit_method(m, T, M, tr, y, g, seed, scale)
            for split, idx in (("CAL_P", sp["CAL_P"]), ("CAL_C", sp["CAL_C"]), ("ID", sp["ID"]), ("OOD", ood)):
                idx = np.asarray(idx, dtype=int)
                if len(idx) == 0: continue
                out.append(pd.DataFrame(dict(held_out=held, seed=seed, method=m, prep=prep, split=split, sample=idx,
                                             cohort=g[idx], y=y[idx], p=model.predict_proba(M[idx])[:, 1],
                                             train_prev=y[tr].mean(), n_train=len(tr))))
        del M
    return pd.concat(out)


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "crc"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    methods = sys.argv[3].split(",") if len(sys.argv) > 3 else ["lasso", "rf", "gbdt", "mlp"]
    T = B.TASKS[task]()
    cohorts = [c for c in np.unique(T["g"]) if min((T["y"][T["g"] == c] == 1).sum(), (T["y"][T["g"] == c] == 0).sum()) >= 10]
    print(f"[protocol v2] task={task} held-out={len(cohorts)} seeds={n_seeds} methods={methods}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=int(os.environ.get("BENCH_JOBS", min(24, os.cpu_count() - 2))))(
        delayed(run_unit)(T, c, s, methods) for c in cohorts for s in range(n_seeds))
    df = pd.concat(res); df["task"] = task
    os.makedirs(f"{ROOT}/runs", exist_ok=True)
    path = f"{ROOT}/runs/preds3_{task}{os.environ.get('BENCH_TAG', '')}.parquet"; df.to_parquet(path)
    print(f"saved {path} rows={len(df)} mins={round((time.time() - t0) / 60, 1)}")
