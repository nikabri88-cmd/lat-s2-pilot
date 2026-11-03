"""Control for the 'applied' probe: how well do surface features of the prompt (fact value, queried line, rooms,
distractor devices and states, time) predict whether the 1.5B model applies the norm on an R=1 prompt?
Items are regenerated exactly as in the run (norms.make_items with the run's seed)."""
import sys, glob, os
import numpy as np
sys.path.insert(0, sys.argv[2])
import norms as NM
import measures_s2 as MS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DEV = [d for d, _ in NM.DISTRACTORS]


def surface(item, D):
    f = [D, item["qline"] == 0, item["qline"] == 1, item["qline"] == 2, item["A"], item["B"],
         int(item["time"][:2]), int(item["time"][3:]), item["rooms"][item["qline"]]]
    f += [r for k, r in enumerate(item["rooms"]) if k != item["qline"]]
    for slot in range(2):
        f += [item["dist"][slot][0] == d for d in DEV]
    return np.array(f, dtype=float)


def cv_ba(X, yv, groups, make, seed=0):
    ug = np.unique(groups); fold = dict(zip(ug, np.random.default_rng([seed, 5]).permutation(len(ug)) % 5))
    folds = np.array([fold[g] for g in groups]); pred = np.zeros_like(yv)
    for k in range(5):
        tr, te = folds != k, folds == k
        if len(np.unique(yv[tr])) < 2:
            pred[te] = yv[tr][0]; continue
        pred[te] = make().fit(X[tr], yv[tr]).predict(X[te])
    return float(np.mean([np.mean(pred[yv == c] == c) for c in np.unique(yv)]))


for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    z = np.load(f, allow_pickle=False)
    name = os.path.basename(f)[:-4]
    domain = name.split("__")[0]
    items = NM.make_items(int(z["item"].max()) + 1, domain, 2026)
    a = (z["c_out"] < 0).astype(int); y = z["y"]; r1 = z["R"] == 1
    app = (a == y)[r1].astype(int)
    if not 0.2 <= app.mean() <= 0.8:
        continue
    S = np.stack([surface(items[i], d) for i, d in zip(z["item"][r1], z["D"][r1])])
    g = z["item"][r1]
    lin = cv_ba(S, app, g, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)))
    rf = cv_ba(S, app, g, lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0))
    d_only = cv_ba(S[:, :1], app, g, lambda: LogisticRegression(max_iter=2000))
    X13 = z["res_monitored"][:, 1].astype(np.float32)[r1]
    probe = MS.probe_ba(X13, app, g, 0)
    print(f"{name:26s} applied {app.mean():.2f} | surface: linear {lin:.2f}, forest {rf:.2f}, fact value only {d_only:.2f} | L13 probe {probe:.2f}")
