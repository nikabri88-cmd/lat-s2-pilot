"""Same probe vs lens comparison for the plain fact D (no norm, R = 0 prompts), 1.5B debug: L13 probe for D vs lens AUC for D."""
import sys, glob, os
import numpy as np
sys.path.insert(0, sys.argv[2])
import measures_s2 as MS
from sklearn.metrics import roc_auc_score
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    z = np.load(f, allow_pickle=False); r0 = z["R"] == 0
    row = []
    for j, l in enumerate(z["monitored"]):
        X = z["res_monitored"][:, j].astype(np.float32)[r0]
        pb = MS.probe_ba(X, z["D"][r0], z["item"][r0], 0)
        auc = roc_auc_score(z["D"][r0], -z["lens_main"][r0, int(l)])
        row.append(f"L{l}: probe {pb:.2f}, lens AUC {auc:.2f}")
    print(f"{os.path.basename(f)[:-4]:26s} " + " | ".join(row))
