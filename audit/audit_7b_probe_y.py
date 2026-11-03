"""7B main run: probe_y (balanced accuracy of the linear probe for y = D xor R) at the monitored layers,
for norm-device pairs the model follows and for those it ignores. Reads results_main/summary.json."""
import json, sys
s = json.load(open(sys.argv[1]))
fol = [r for r in s["rows"] if r["follows"]]
ign = [r for r in s["rows"] if not r["follows"]]
for l in s["monitored"]:
    k = f"L{l}"
    f = [r[k]["probe_y"] for r in fol]
    print(f"layer {l}: followed {len(f)} pairs, mean probe_y {sum(f) / len(f):.3f}, "
          f"{sum(v >= 0.8 for v in f)} with probe_y >= 0.8")
    for r in ign:
        print(f"    ignored {r['domain']}/{r['norm']}: acc with norm on {r['acc_R1']:.2f}, probe_y {r[k]['probe_y']:.3f}")
