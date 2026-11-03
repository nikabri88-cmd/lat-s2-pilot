"""LAT-S2 pilot measures, ported from LAT-S1b to natural-language norms.

Probabilities are binary: p = sigmoid(contrast / T), contrast = logit(state0) - logit(state1) at the answer position.
  Use_R  slope of the model's Delta p on the ideal Delta p over R-twins (both D)          (LAT: on the Bayes filter)
  Use_D  same over D-twins with R off
  Seen_k slope of the lens's Delta p on the model's Delta p over the twins of k            (LAT-S1 Seen)
  rel_R  Seen_R / Seen_D with a joint bootstrap
Classes follow LAT-S1b: rel_R sighted if the 95% CI lower bound >= 0.8, blind if the upper bound <= 0.5; Seen_R
sighted if lower >= 0.3, blind if upper <= 0.1; rel_R is defined when Use_R, Use_D >= 0.2 and Seen_D >= 0.1; a unit
gets a class at a layer when rel_R (main lens), rel_R (control lens) and Seen_R (main lens) agree.
rho_c: Ostrow et al. Eq. 10 with one source position (the answer position) and lag 0, readout
r = gamma * (U[s0] - U[s1]) without the final-norm Jacobian: g_c = (J_c^T r) . v_R, rho_c = |E g_c| / E |g_c|,
v_R the encoding direction of R from regressing h_l on [D, R, A, B, line] (LAT-S1b, coherence.encoding_directions).
Bootstrap: 1000 resamples of base items.
"""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
N_BOOT = 1000
REL_SIGHTED, REL_BLIND = 0.8, 0.5
SEEN_SIGHTED, SEEN_BLIND = 0.3, 0.1
USE_MIN, SEEN_D_MIN = 0.2, 0.1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def boot_indices(n_items, seed):
    return np.random.default_rng([seed, 99]).integers(0, n_items, (N_BOOT, n_items))


def ratio(num_item, den_item, B):
    """Ratio of sums over items with bootstrap over items."""
    den = den_item.sum()
    point = num_item.sum() / den if den > 0 else np.nan
    dsum = den_item[B].sum(1)
    bs = num_item[B].sum(1) / np.where(dsum > 0, dsum, np.nan)
    return float(point), bs


def summ(point, bs):
    ok = bs[np.isfinite(bs)]
    ci = [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) > 10 else [np.nan, np.nan]
    return dict(point=float(point), ci=ci)


class Twins:
    """Index of the four prompts of each base item: ix[item, D, R]."""

    def __init__(self, meta, n_items):
        self.N = n_items
        self.ix = np.full((n_items, 2, 2), -1)
        self.ix[meta["item"], meta["D"], meta["R"]] = np.arange(len(meta["item"]))
        assert (self.ix >= 0).all()

    def pairs(self, k, fixed=None):
        """(item, idx_hi, idx_lo) for twins of k: k=1 minus k=0. fixed: value of the other factor, or None for both."""
        out = []
        for v in ((0, 1) if fixed is None else (fixed,)):
            for i in range(self.N):
                if k == "R":
                    out.append((i, self.ix[i, v, 1], self.ix[i, v, 0]))
                else:
                    out.append((i, self.ix[i, 1, v], self.ix[i, 0, v]))
        return np.array(out)

    def slope(self, dy, dx, items, B):
        num = np.bincount(items, weights=dy * dx, minlength=self.N)
        den = np.bincount(items, weights=dx * dx, minlength=self.N)
        return ratio(num, den, B)


def ideal_p(meta):
    """Ideal probability of state 0: the correct answer is y = D xor R."""
    return (meta["y"] == 0).astype(float)


def behaviour(meta, c_out, tw, B):
    p = sigmoid(c_out)
    ip = ideal_p(meta)
    res = {}
    for k, fixed in (("R", None), ("D", 0), ("D_R1", 1)):
        kk = "D" if k.startswith("D") else "R"
        P = tw.pairs(kk, fixed)
        pt, bs = tw.slope(p[P[:, 1]] - p[P[:, 2]], ip[P[:, 1]] - ip[P[:, 2]], P[:, 0], B)
        res["Use_" + k] = summ(pt, bs) | {"_bs": bs}
    correct = (np.sign(c_out) == np.where(meta["y"] == 0, 1, -1))
    res["acc_R0"] = float(correct[meta["R"] == 0].mean())
    res["acc_R1"] = float(correct[meta["R"] == 1].mean())
    return res


def seen(c_lens, temp, c_out, tw, B):
    """Seen_R (R-twins, both D) and Seen_D (D-twins, R off) for one lens at one layer."""
    if not np.isfinite(c_lens).all():
        return None
    q = sigmoid(c_lens / temp)
    p = sigmoid(c_out)
    out = {}
    for k, fixed in (("R", None), ("D", 0)):
        P = tw.pairs(k, fixed)
        pt, bs = tw.slope(q[P[:, 1]] - q[P[:, 2]], p[P[:, 1]] - p[P[:, 2]], P[:, 0], B)
        out["Seen_" + k] = summ(pt, bs) | {"_bs": bs}
    rp = out["Seen_R"]["point"] / out["Seen_D"]["point"] if out["Seen_D"]["point"] != 0 else np.nan
    out["rel_R"] = summ(rp, out["Seen_R"]["_bs"] / out["Seen_D"]["_bs"])
    return out


def rel_class(rel, beh, s):
    defined = (beh["Use_R"]["point"] >= USE_MIN and beh["Use_D"]["point"] >= USE_MIN
               and s["Seen_D"]["point"] >= SEEN_D_MIN)
    if not defined:
        return "undefined"
    lo, hi = rel["ci"]
    return "sighted" if lo >= REL_SIGHTED else ("blind" if hi <= REL_BLIND else "none")


def seen_class(sr):
    lo, hi = sr["ci"]
    return "sighted" if lo >= SEEN_SIGHTED else ("blind" if hi <= SEEN_BLIND else "none")


def unit_class(readings):
    return readings[0] if len(set(readings)) == 1 and readings[0] in ("sighted", "blind") else "none"


def features(meta):
    pm = lambda x: 2.0 * x - 1.0
    q = meta["qline"]
    return np.stack([pm(meta["D"]), pm(meta["R"]), pm(meta["A"]), pm(meta["B"]),
                     (q == 1).astype(float), (q == 2).astype(float)], 1)


def encoding_direction(X, F, col=1):
    """Coefficient of feature `col` (R) in the least-squares regression of X on F with intercept, normalised."""
    F1 = np.concatenate([F, np.ones((len(F), 1))], 1)
    coef, *_ = np.linalg.lstsq(F1, np.asarray(X, np.float64), rcond=None)
    w = coef[col]
    return w / np.linalg.norm(w)


def rho_c(g):
    g = np.asarray(g, np.float64)
    a = np.abs(g).mean()
    return dict(rho_c=float(abs(g.mean()) / a) if a > 0 else np.nan, a=float(a), Jv=float(abs(g.mean())))


def coherence_layer(res_l, W_l, meta, F):
    v = encoding_direction(res_l, F)
    g = W_l.astype(np.float64) @ v
    out = rho_c(g)
    out["rho_c_R1"] = rho_c(g[meta["R"] == 1])["rho_c"]
    out["rho_c_R0"] = rho_c(g[meta["R"] == 0])["rho_c"]
    return out


def probe_ba(X, y, groups, seed, n_fold=5):
    """Out-of-fold balanced accuracy of a linear probe (standardise, PCA to at most 128 dimensions fitted on the
    training fold, logistic regression), folds by base item."""
    ug = np.unique(groups)
    rng = np.random.default_rng([seed, 5])
    fold_of_item = dict(zip(ug, rng.permutation(len(ug)) % n_fold))
    folds = np.array([fold_of_item[g] for g in groups])
    pred = np.zeros_like(y)
    for f in range(n_fold):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        k = int(min(128, tr.sum() - 1, X.shape[1]))
        clf = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed),
                            LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def strip(d):
    """Drop bootstrap arrays before saving."""
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, (np.floating,)):
        return float(d)
    return d
