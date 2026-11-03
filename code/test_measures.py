"""Synthetic checks of the pilot measures: a lens that shows the applied rule must be sighted, a lens that shows only
the raw fact must be blind, and rho_c must separate sign-consistent from sign-mixed readouts."""
import numpy as np
import measures_s2 as MS

N = 64
rng = np.random.default_rng(0)
items = np.repeat(np.arange(N), 4)
D = np.tile([0, 0, 1, 1], N); R = np.tile([0, 1, 0, 1], N); y = D ^ R
meta = dict(item=items, D=D, R=R, y=y, A=rng.integers(0, 2, 4 * N), B=rng.integers(0, 2, 4 * N),
            qline=rng.integers(0, 3, 4 * N))
tw = MS.Twins(meta, N); B = MS.boot_indices(N, 1)
c_out = np.where(y == 0, 8.0, -8.0) + rng.normal(0, 0.5, 4 * N)
beh = MS.behaviour(meta, c_out, tw, B)
print("Use_R", round(beh["Use_R"]["point"], 3), "Use_D", round(beh["Use_D"]["point"], 3), "acc", beh["acc_R0"], beh["acc_R1"])
sighted = 0.6 * c_out + rng.normal(0, 0.3, 4 * N)                      # lens tracks the applied rule
blind = np.where(D == 0, 5.0, -5.0) + rng.normal(0, 0.3, 4 * N)        # lens shows only the raw fact
for name, cl in (("sighted", sighted), ("blind", blind)):
    s = MS.seen(cl, 1.0, c_out, tw, B)
    rc = MS.rel_class(s["rel_R"], beh, s); sc = MS.seen_class(s["Seen_R"])
    print(name, "Seen_R", round(s["Seen_R"]["point"], 3), s["Seen_R"]["ci"], "Seen_D", round(s["Seen_D"]["point"], 3),
          "rel", round(s["rel_R"]["point"], 3), "->", rc, sc, MS.unit_class([rc, rc, sc]))
g_same = np.abs(rng.normal(1, 0.3, 400)); g_mixed = rng.normal(0, 1, 400)
print("rho_c same-sign", round(MS.rho_c(g_same)["rho_c"], 3), "mixed", round(MS.rho_c(g_mixed)["rho_c"], 3))
X = rng.normal(size=(4 * N, 32)); u = rng.normal(size=32); u /= np.linalg.norm(u)
X += np.outer(2 * R - 1, u) * 3
v = MS.encoding_direction(X, MS.features(meta))
print("encoding direction cos to planted R direction", round(float(abs(v @ u)), 3))
print("probe BA for y on X with planted R only", round(MS.probe_ba(X, y, items, 0), 3), "for R", round(MS.probe_ba(X, R, items, 0), 3))
