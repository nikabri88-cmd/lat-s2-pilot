"""Audit of probe_y (LAT-S2 pilot, Qwen2.5-1.5B debug run): does the linear probe for the norm-applied answer y = D xor R
track what the model will actually say, or the inputs D and R? Uses the saved residuals at the monitored layers.
  a        = the model's actual answer (sign of its own answer contrast)
  applied  = a == y on prompts with the norm on (R = 1): did the model apply the norm on this prompt?"""
import sys, glob, os
import numpy as np
sys.path.insert(0, sys.argv[2])
import measures_s2 as MS
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    z = np.load(f, allow_pickle=False)
    name = os.path.basename(f)[:-4]
    y, D, R, item = z["y"], z["D"], z["R"], z["item"]
    a = (z["c_out"] < 0).astype(int)
    agree = a == y
    r1 = R == 1
    app = agree[r1].astype(int)
    out = dict(unit=name, applied=float(app.mean()), n_dis=int((~agree).sum()))
    for j, l in enumerate(z["monitored"]):
        X = z["res_monitored"][:, j].astype(np.float32)
        ba_y = MS.probe_ba(X, y, item, 0)
        ba_a = MS.probe_ba(X, a, item, 0)
        rec = dict(ba_y=ba_y, ba_a=ba_a)
        if (~agree).sum() >= 5:
            # out-of-fold predictions of the y-probe on the prompts where the model does NOT give y
            ug = np.unique(item); fold = dict(zip(ug, np.random.default_rng([0, 5]).permutation(len(ug)) % 5))
            folds = np.array([fold[g] for g in item]); pred = np.zeros_like(y)
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            for k in range(5):
                tr, te = folds != k, folds == k
                clf = make_pipeline(StandardScaler(), PCA(n_components=int(min(128, tr.sum() - 1)), random_state=0),
                                    LogisticRegression(C=0.1, max_iter=2000)).fit(X[tr], y[tr])
                pred[te] = clf.predict(X[te])
            dis = ~agree
            rec["yprobe_on_disagreements_says_y"] = float((pred[dis] == y[dis]).mean())
        if 0.2 <= app.mean() <= 0.8:
            ba = MS.probe_ba(X[r1], app, item[r1], 0)
            null = [MS.probe_ba(X[r1], rng.permutation(app), item[r1], s) for s in range(40)]
            rec["applied_ba"] = ba
            rec["applied_p"] = float((1 + sum(n >= ba for n in null)) / (1 + len(null)))
        lens = z["lens_main"][:, int(l)]
        if len(np.unique(a[r1])) == 2:
            rec["lens_auc_answer_R1"] = float(roc_auc_score(a[r1], -lens[r1]))
        out[int(l)] = rec
    rows.append(out)

for o in rows:
    print(f"\n{o['unit']}: norm applied on {o['applied']:.2f} of R=1 prompts, {o['n_dis']} prompts where the model does not give y")
    for l in (6, 13, 20):
        r = o[l]
        s = f"  L{l}: probe y {r['ba_y']:.2f} | probe actual answer {r['ba_a']:.2f}"
        if "yprobe_on_disagreements_says_y" in r:
            s += f" | y-probe on disagreements says y {r['yprobe_on_disagreements_says_y']:.2f}"
        if "applied_ba" in r:
            s += f" | 'applied' probe {r['applied_ba']:.2f} (perm p {r['applied_p']:.2f})"
        if "lens_auc_answer_R1" in r:
            s += f" | lens AUC for answer (R=1) {r['lens_auc_answer_R1']:.2f}"
        print(s)
