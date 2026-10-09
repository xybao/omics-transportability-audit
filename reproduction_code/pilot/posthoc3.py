"""Scientific model, split or calibration functions used by the evaluation pipeline."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-4                                   # 概率裁剪阈值（公开写明）：NLL 单样本上限 ≈ 9.2
PI_STAR = float(os.environ.get("PI_STAR", "0.5"))
clip = lambda p: np.clip(p, EPS, 1 - EPS)
logit = lambda p: np.log(clip(p) / (1 - clip(p)))
sigm = lambda z: 1 / (1 + np.exp(-z))


def shift_prior(p, pi_s, pi_t):
    a = clip(p) * pi_t / pi_s; b = (1 - clip(p)) * (1 - pi_t) / (1 - pi_s); return a / (a + b)


def fit_logistic(z, y, w=None, slope=True):
    """返回 (intercept, slope)。slope=False 时固定斜率为 1，只估计截距（offset 模型，牛顿法）。"""
    w = np.ones(len(y)) if w is None else w
    if len(np.unique(y)) < 2: return 0.0, 1.0
    if slope:
        m = LogisticRegression(C=1e4, max_iter=2000).fit(z[:, None], y, sample_weight=w)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    a = 0.0
    for _ in range(50):
        q = sigm(z + a); grad = np.sum(w * (y - q)); hess = np.sum(w * q * (1 - q)) + 1e-9
        step = float(np.clip(grad / hess, -1, 1)); a = float(np.clip(a + step, -10, 10))
        if abs(step) < 1e-8: break
    return float(a), 1.0


def wmetrics(y, p, w):
    p = clip(p); w = w / w.sum(); z = logit(p)
    nll = -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))); brier = np.sum(w * (p - y) ** 2)
    citl, _ = fit_logistic(z, y, w, slope=False); _, slope = fit_logistic(z, y, w, slope=True)
    order = np.argsort(p); cw = np.cumsum(w[order]); edges = np.searchsorted(cw, np.linspace(0, 1, 11)[1:-1])
    ece = sum(w[b].sum() * abs(np.average(y[b], weights=w[b]) - np.average(p[b], weights=w[b]))
              for b in np.split(order, edges) if len(b) and w[b].sum() > 0)
    return dict(nll=nll, brier=brier, cal_int=citl, cal_slope=slope, ece=float(ece), auroc=roc_auc_score(y, p),
                n_eff=float(1 / np.sum(w ** 2)))


def em_prior(p, pi_s):
    pi = pi_s
    for _ in range(300):
        new = shift_prior(p, pi_s, pi).mean()
        if abs(new - pi) < 1e-7: break
        pi = new
    return float(np.clip(pi, 0.02, 0.98))


def bbse_prior(p_cal, y_cal, p_t):
    tpr = (p_cal[y_cal == 1] > 0.5).mean(); fpr = (p_cal[y_cal == 0] > 0.5).mean()
    if tpr - fpr < 0.05: return np.nan
    return float(np.clip(((p_t > 0.5).mean() - fpr) / (tpr - fpr), 0.02, 0.98))


def conformal_quantile(scores, alpha=0.1):
    if not 0 < alpha < 1: raise ValueError('alpha must be between zero and one')
    scores = np.asarray(scores)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite calibration scores')
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if rank > len(scores) else np.sort(scores)[rank - 1]


def conformal(p_cal, y_cal, p_t, y_t, alpha=0.1):
    S = np.zeros((len(p_t), 2), bool)
    for k in (0, 1):
        s = 1 - np.where(k == 1, p_cal, 1 - p_cal)[y_cal == k]
        q = conformal_quantile(s, alpha)
        S[:, k] = (1 - np.where(k == 1, p_t, 1 - p_t)) <= q
    size = S.sum(1)
    return dict(cov=S[np.arange(len(y_t)), y_t].mean(), cov0=S[y_t == 0, 0].mean(), cov1=S[y_t == 1, 1].mean(),
                both=(size == 2).mean(), single=(size == 1).mean(), empty=(size == 0).mean())


def evaluate(y, p, pi_from, layer):
    """layer A：原样；layer B：预测搬到 π*，样本按类别加权到 π*。"""
    if layer == "A": return wmetrics(y, p, np.ones(len(y)))
    ps = y.mean(); w = np.where(y == 1, PI_STAR / ps, (1 - PI_STAR) / (1 - ps))
    return wmetrics(y, shift_prior(p, pi_from, PI_STAR), w)


def one_unit(key, d):
    rng = np.random.default_rng(zlib_seed(key)); rows = []
    cp, cc = d[d.split == "CAL_P"], d[d.split == "CAL_C"]; pi_s = float(d.train_prev.iloc[0])
    a, b = fit_logistic(logit(cp.p.values), cp.y.values); platt = lambda p: sigm(a + b * logit(p))
    pi_cal = cp.y.mean()                                       # platt 后的概率以校准集先验为基准
    for split in ("ID", "OOD"):
        t = d[d.split == split]; y = t.y.values; p0 = t.p.values
        if len(np.unique(y)) < 2: continue
        pp = platt(p0); base = dict(zip(("held_out", "seed", "method", "prep"), key), split=split, n=len(y), base_rate=y.mean())
        pi_em = em_prior(pp, pi_cal); pi_bb = bbse_prior(platt(cp.p.values), cp.y.values, pp)
        ladder = {"raw": (p0, pi_s), "platt": (pp, pi_cal),
                  "em": (shift_prior(pp, pi_cal, pi_em), pi_em),
                  "bbse": (shift_prior(pp, pi_cal, pi_bb), pi_bb) if not np.isnan(pi_bb) else (pp, pi_cal),
                  "prior*": (shift_prior(pp, pi_cal, y.mean()), y.mean())}
        for name, (q, pi_q) in ladder.items():
            for layer in ("A", "B"):
                rows.append(dict(base, posthoc=name, layer=layer, pi_hat=pi_q, **evaluate(y, q, pi_q, layer)))
        # oracle：目标标签 2 折交叉拟合的截距+斜率
        idx = rng.permutation(len(y)); h = len(y) // 2; qo = np.empty(len(y))
        for u, v in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
            ao, bo = fit_logistic(logit(pp[u]), y[u]); qo[v] = sigm(ao + bo * logit(pp[v]))
        for layer in ("A", "B"):
            rows.append(dict(base, posthoc="oracle*", layer=layer, pi_hat=y.mean(), **evaluate(y, qo, y.mean(), layer)))
        # 少量目标标签：随机抽 k 个样本标注 → 更新截距（k<20）或截距+斜率，在其余样本上评分
        if split == "OOD":
            for k, use_slope in ((10, False), (20, False), (40, False), (40, True)):
                if len(y) < k + 20: continue
                acc = []
                for r in range(5):
                    lab = rng.choice(len(y), k, replace=False); rest = np.setdiff1d(np.arange(len(y)), lab)
                    if len(np.unique(y[lab])) < 2 or len(np.unique(y[rest])) < 2: continue
                    al, bl = fit_logistic(logit(pp[lab]), y[lab], slope=use_slope)
                    ql = sigm(al + bl * logit(pp[rest]))
                    ref = {nm: evaluate(y[rest], ladder[nm][0][rest], ladder[nm][1], "A")["nll"] for nm in ("platt", "em")}
                    m = evaluate(y[rest], ql, y[lab].mean(), "A"); m.update(nll_platt_same=ref["platt"], nll_em_same=ref["em"]); acc.append(m)
                if acc:
                    rows.append(dict(base, posthoc=f"lab{k}{'s' if use_slope else ''}", layer="A", pi_hat=np.nan, **pd.DataFrame(acc).mean().to_dict()))
        # 决策：阈值在源域 CAL_P 上预设（使源域敏感度≥0.90 的最大阈值），目标标签只用于评分
        pc_ = platt(cp.p.values); pos = np.sort(pc_[cp.y.values == 1])
        tau = pos[int(np.floor(0.10 * len(pos)))] if len(pos) >= 10 else 0.5
        rows.append(dict(base, posthoc="dec_platt", layer="A", tau=tau, sens=(pp[y == 1] >= tau).mean(), spec=(pp[y == 0] < tau).mean()))
        # 共形：分位数来自独立的 CAL_C
        pcc = platt(cc.p.values)
        for name in ("platt", "em"):
            qt = ladder[name][0]; qc = pcc if name == "platt" else shift_prior(pcc, pi_cal, pi_em)
            rows.append(dict(base, posthoc=f"conf_{name}", layer="A", **conformal(qc, cc.y.values, qt, y)))
    return rows


def zlib_seed(key):
    import zlib; return zlib.crc32("|".join(map(str, key)).encode())


def cohort_summary(x, col):
    """先队列内对种子（和方法）取平均，再跨队列汇总。"""
    per = x.groupby("held_out")[col].mean()
    return per


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "crc"
    import glob
    files = [f for f in (glob.glob(f"{ROOT}/runs/preds3_{task}.parquet") + glob.glob(f"{ROOT}/runs/preds3_{task}_*.parquet")) if "fullsrc" not in f and "smoke" not in f and "valid" not in os.path.basename(f)[len(f"preds3_{task}"):]]
    df = pd.concat([pd.read_parquet(f) for f in files])
    groups = list(df.groupby(["held_out", "seed", "method", "prep"]))
    res = Parallel(n_jobs=24)(delayed(one_unit)(k, d) for k, d in groups)
    out = pd.DataFrame([r for rows in res for r in rows]); out["task"] = task
    suffix = "" if PI_STAR == 0.5 else f"_pi{PI_STAR:g}"
    out.to_csv(f"{ROOT}/runs/posthoc3_{task}{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    main = out[~out.posthoc.str.startswith(("conf_", "lab", "dec_"))]

    for layer, title in (("B", f"层 B：标准化先验 π*={PI_STAR}（ID 与 OOD 先验严格对齐）"), ("A", "层 A：原始样本分布")):
        print(f"\n################ {task} | {title} | 跨方法平均；单元=留出队列（先对种子取平均）")
        rows = []
        for (prep, ph), x in main[main.layer == layer].groupby(["prep", "posthoc"]):
            r = dict(prep=prep, posthoc=ph)
            for col in ("auroc", "nll", "brier", "cal_int", "cal_slope"):
                i = x[x.split == "ID"].groupby("held_out")[col].mean(); o = x[x.split == "OOD"].groupby("held_out")[col].mean()
                if col in ("cal_int",): i, o = i.abs(), o.abs()
                if col == "cal_slope": i, o = (i - 1).abs(), (o - 1).abs()
                r[f"{col}_ID"] = i.median(); r[f"{col}_OOD"] = o.median()
                if col == "nll": r["nll_worse"] = f"{int((o > i + 0.02).sum())}/{len(o)}"
            rows.append(r)
        t = pd.DataFrame(rows).set_index(["prep", "posthoc"])
        order = ["raw", "platt", "em", "bbse", "prior*", "oracle*"]
        t = t.reindex([(p, h) for p in main.prep.unique() for h in order])
        t.columns = [c.replace("cal_int", "|截距|").replace("cal_slope", "|斜率-1|") for c in t.columns]
        print(t.round(3).to_string())

    print(f"\n---- 先验估计误差（估计值 − 目标研究样本病例比例；OOD；跨队列）")
    e = main[(main.split == "OOD") & (main.layer == "A") & main.posthoc.isin(["em", "bbse"])].copy()
    e["err"] = e.pi_hat - e.base_rate
    pe = e.groupby(["prep", "posthoc", "held_out"]).err.mean().abs().groupby(["prep", "posthoc"]).agg(["median", "max"])
    print(pe.round(3).to_string())

    print("\n---- 少量目标标签 vs 无标签（OOD；同一批剩余样本上的 NLL；跨队列中位数）")
    lb = out[out.posthoc.str.startswith("lab")]
    if len(lb):
        t = lb.groupby(["prep", "posthoc", "held_out"])[["nll", "nll_platt_same", "nll_em_same"]].mean().groupby(["prep", "posthoc"]).median()
        t.columns = ["用 k 个标签更新", "只做源域校准", "EM 无标签"]; print(t.round(3).to_string())

    print("\n---- 共形预测（目标 0.90；分位数来自独立的 CAL_C；按类别分层）")
    c = out[out.posthoc.str.startswith("conf_")]
    t = c.groupby(["prep", "posthoc", "split", "held_out"])[["cov", "cov0", "cov1", "both", "single", "empty"]].mean()
    s = t.groupby(["prep", "posthoc", "split"]).agg({"cov": ["median", "min"], "cov0": "min", "cov1": "min", "both": "median", "single": "median", "empty": "median"})
    s.columns = ["cov_中位", "cov_最差队列", "cov0_最差", "cov1_最差", "双标签率", "单标签率", "空集率"]
    print(s.round(3).to_string())

    print()
    print("---- 决策：源域预设阈值（源域敏感度≥0.90）下的敏感度 / 特异度；跨队列")
    dd = out[out.posthoc == "dec_platt"].groupby(["prep", "split", "held_out"])[["sens", "spec"]].mean()
    print(dd.groupby(["prep", "split"]).agg({"sens": ["median", "min"], "spec": ["median", "min"]}).round(3).to_string())
