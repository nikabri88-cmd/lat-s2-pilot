"""Recheck of the 7B main run from its per-pair JSON files (results_main/units): recompute every reading and class
from the stored bootstrap intervals with the rules in code/measures_s2.py, compare them with the stored ones and
with results_main/summary.json, and count classes at every layer among the pairs the model follows.
Usage: python3 audit/recheck_main_units.py results_main code"""
import glob, json, os, sys
sys.path.insert(0, sys.argv[2])
import measures_s2 as MS

res_dir = sys.argv[1]
units = {}
for f in sorted(glob.glob(os.path.join(res_dir, "units", "*.json"))):
    u = json.load(open(f))
    units[(u["domain"], u["norm"])] = u
summary = json.load(open(os.path.join(res_dir, "summary.json")))
rows = {(r["domain"], r["norm"]): r for r in summary["rows"]}
assert set(units) == set(rows), "units and summary cover different pairs"

mismatch = 0
first = {}
for key, u in units.items():
    beh = u["behaviour"]
    for l, e in u["layers"].items():
        rd = [MS.rel_class(e["main"]["rel_R"], beh, e["main"]), MS.rel_class(e["control"]["rel_R"], beh, e["control"]),
              MS.seen_class(e["main"]["Seen_R"])]
        cls = MS.unit_class(rd)
        if rd != e["readings"] or cls != e["class"]:
            mismatch += 1
            print("MISMATCH", key, l, rd, e["readings"], cls, e["class"])
    fr = next((int(l) for l in sorted(u["layers"], key=int) if u["layers"][l]["main"]["Seen_D"]["point"] >= MS.SEEN_D_MIN), None)
    first[key] = fr
    r = rows[key]
    for l in summary["monitored"]:
        if r[f"L{l}"]["cls"] != u["layers"][str(l)]["class"]:
            mismatch += 1; print("SUMMARY MISMATCH", key, l)
    if fr != r["first_readable"]["layer"] or (fr is not None and r["first_readable"]["cls"] != u["layers"][str(fr)]["class"]):
        mismatch += 1; print("FIRST-READABLE MISMATCH", key, fr, r["first_readable"]["layer"])
print(f"stored readings, classes and summary entries recomputed: {mismatch} mismatches")

fol = [k for k, u in units.items() if u["follows"]]
print(f"\npairs the model follows: {len(fol)} of {len(units)}")
print("layer  reads fact  sighted  blind  no class")
for l in sorted(units[fol[0]]["layers"], key=int):
    reads = [k for k in fol if units[k]["layers"][l]["main"]["Seen_D"]["point"] >= MS.SEEN_D_MIN]
    c = [units[k]["layers"][l]["class"] for k in fol]
    print(f"{int(l):5d}  {len(reads):10d}  {c.count('sighted'):7d}  {c.count('blind'):5d}  {len(fol) - c.count('sighted') - c.count('blind'):8d}")

print("\nfirst readable layer of each followed pair: class, readings, rel_R main (95% CI), Seen_R main (95% CI)")
for k in sorted(fol):
    l = first[k]; e = units[k]["layers"][str(l)]
    rel, sr = e["main"]["rel_R"], e["main"]["Seen_R"]
    print(f"{k[0]:5s} {k[1]:18s} L{l:<3d} {e['class']:8s} {str(e['readings']):38s} "
          f"rel {rel['point']:.2f} [{rel['ci'][0]:.2f}, {rel['ci'][1]:.2f}]  Seen_R {sr['point']:.3f} [{sr['ci'][0]:.3f}, {sr['ci'][1]:.3f}]")
