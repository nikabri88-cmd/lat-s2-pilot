"""Per unit with partial norm-following (1.5B debug): is the decision 'applied' already decodable at L6 (early, inputs only)?
Also: does the regenerated item list match the saved qline/A/B, and which single surface feature best predicts 'applied'."""
import sys, glob, os
import numpy as np
sys.path.insert(0, sys.argv[2])
import norms as NM
import measures_s2 as MS
from sklearn.tree import DecisionTreeClassifier

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
    name = os.path.basename(f)[:-4]; domain = name.split("__")[0]
    items = NM.make_items(int(z["item"].max()) + 1, domain, 2026)
    ok = all(items[i]["qline"] == q and items[i]["A"] == a and items[i]["B"] == b
             for i, q, a, b in zip(z["item"], z["qline"], z["A"], z["B"]))
    a = (z["c_out"] < 0).astype(int); r1 = z["R"] == 1
    app = (a == z["y"])[r1].astype(int)
    acc0 = float((a == z["y"])[~r1].mean())
    if not 0.2 <= app.mean() <= 0.8:
        print(f"{name:26s} items match {ok} | acc R0 {acc0:.2f}, applied on R1 {app.mean():.2f} (skipped)")
        continue
    g = z["item"][r1]
    bas = [MS.probe_ba(z["res_monitored"][:, j].astype(np.float32)[r1], app, g, 0) for j in range(3)]
    feats = {"D": z["D"][r1], "qline": z["qline"][r1], "A": z["A"][r1], "B": z["B"][r1],
             "room_q": np.array([items[i]["rooms"][items[i]["qline"]] for i in g]),
             "hour": np.array([int(items[i]["time"][:2]) for i in g]),
             "dist1": np.array([NM.DISTRACTORS.index(items[i]["dist"][0]) for i in g]),
             "dist2": np.array([NM.DISTRACTORS.index(items[i]["dist"][1]) for i in g])}
    single = {k: cv_ba(v[:, None].astype(float), app, g, lambda: DecisionTreeClassifier(max_depth=2, random_state=0))
              for k, v in feats.items()}
    best = sorted(single.items(), key=lambda kv: -kv[1])[:3]
    print(f"{name:26s} items match {ok} | acc R0 {acc0:.2f}, applied on R1 {app.mean():.2f} | 'applied' probe "
          f"L6 {bas[0]:.2f}, L13 {bas[1]:.2f}, L20 {bas[2]:.2f} | best single features: "
          + ", ".join(f"{k} {v:.2f}" for k, v in best))
