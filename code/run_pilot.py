"""LAT-S2 pilot runner: lenses, temperatures, per-unit measures, summary and the go / stop rule.

Unit = norm x domain. Monitored layers: quarter, half and three quarters of the depth (layer l = output of block l).
Go rule (pilot plan): at one monitored layer, among units where the model follows the norm (accuracy >= 0.8 with
the norm on and off), at least 3 units are sighted and at least 3 are blind. Everything else is descriptive.
"""
import json
import os
import sys
import time
import numpy as np
import torch
import jlens

import norms as NM
import model_io as MI
import measures_s2 as MS

PRESETS = {
    "debug": dict(model="Qwen/Qwen2.5-1.5B-Instruct", n_items=24, n_lens_prompts=8, n_temp_prompts=4,
                  dim_batch="auto", batch_size="auto", control="half", dtype="auto", min_gpu_gb=0,
                  ignore_preflight=True),   # debug checks the code end to end; preflight is still logged
    "main": dict(model="Qwen/Qwen2.5-7B-Instruct", n_items=96, n_lens_prompts=32, n_temp_prompts=8,
                 dim_batch="auto", batch_size="auto", control="community", dtype="auto", min_gpu_gb=20,
                 community_repo="anicka/jlens-qwen2.5-7b-instruct",
                 community_file="qwen2.5-7b-instruct_jlens.pt"),
}
COMMON = dict(domains=["light", "door"], norms=list(NM.NORM_NAMES), lens_max_len=128, seed=2026,
              follow_min=0.8, n_go=3)


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


CORPUS_SOURCES = [("Salesforce/wikitext", "wikitext-2-raw-v1", "train"),
                  ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation"),
                  ("wikitext", "wikitext-2-raw-v1", "train")]


def lens_corpus(n, seed, allow_fallback, log):
    """Generic text for fitting and calibrating the lens: WikiText paragraphs of at least 600 characters
    (Salesforce/wikitext on the Hugging Face Hub; the older id 'wikitext' as a last resort)."""
    errors = []
    try:
        from datasets import load_dataset
    except Exception as e:
        errors.append(f"datasets: {type(e).__name__}")
        load_dataset = None
    for repo, conf, split in (CORPUS_SOURCES if load_dataset else []):
        try:
            ds = load_dataset(repo, conf, split=split)
        except Exception as e:
            errors.append(f"{repo}/{conf}/{split}: {type(e).__name__}")
            continue
        rng = np.random.default_rng([seed, 11])
        out, tried = [], 0
        for i in rng.permutation(len(ds)):
            t = ds[int(i)]["text"].strip()
            tried += 1
            if len(t) >= 600 and not t.startswith("="):
                out.append(t)
            if len(out) >= n:
                break
        if len(out) >= n:
            name = f"{repo} {conf} {split}"
            log(f"lens corpus: {name}, {len(out)} paragraphs (scanned {tried})")
            return out, name
        errors.append(f"{repo}/{conf}/{split}: only {len(out)} paragraphs")
    if not allow_fallback:
        raise RuntimeError("no lens corpus could be loaded: " + "; ".join(errors))
    log(f"lens corpus: unavailable ({'; '.join(errors)}); fallback text for local tests only")
    rng = np.random.default_rng([seed, 12])
    words = ("the of a in to and is was for on with as by at from that light door room status report state "
             "log fan window heater gate open closed off sensor hour minute test mode").split()
    return [" ".join(rng.choice(words, 160)) for _ in range(n)], "fallback"


def preflight(lm, cfg, log, n_items=8):
    """Behaviour only, before the expensive lens fit: does the model follow each norm (accuracy with R on and off)?
    The go rule needs at least 2 * n_go classified units, so fewer following units stop the pilot here."""
    rows = []
    for domain in cfg["domains"]:
        tok_pair, tok_mode = NM.answer_tokens(lm.tokenizer, domain)
        cap = MI.Capture(lm, tok_pair)
        for norm in cfg["norms"]:
            texts, meta = NM.unit_prompts(lm.tokenizer, norm, domain, n_items, cfg["seed"] + 1)
            c = MI.capture_all(cap, texts, cfg["batch_size"], grads=False)["c_out"]
            ok = np.sign(c) == np.where(meta["y"] == 0, 1, -1)
            a0, a1 = float(ok[meta["R"] == 0].mean()), float(ok[meta["R"] == 1].mean())
            rows.append(dict(domain=domain, norm=norm, acc_R0=a0, acc_R1=a1, tok_mode=tok_mode,
                             follows=bool(a0 >= cfg["follow_min"] and a1 >= cfg["follow_min"])))
            log(f"preflight {domain}/{norm}: accuracy norm off {a0:.2f}, norm on {a1:.2f}"
                f"{'' if rows[-1]['follows'] else '  (not followed)'}")
    return rows


def get_lens(lm, prompts, path, ckpt, cfg, log):
    if os.path.exists(path):
        log(f"lens: loading {os.path.basename(path)}")
        return jlens.JacobianLens.load(path)
    t0 = time.time()
    log(f"lens: fitting on {len(prompts)} prompts, dim_batch {cfg['dim_batch']}, max_len {cfg['lens_max_len']}")
    # the running-sum checkpoint is large (fp32, n_layers * d_model^2); it lives on the runtime's local disk
    lens = jlens.fit(lm, prompts, dim_batch=cfg["dim_batch"], max_seq_len=cfg["lens_max_len"],
                     checkpoint_path=ckpt, checkpoint_every=4)
    lens.save(path)
    log(f"lens: fitted in {(time.time() - t0) / 60:.1f} min -> {os.path.basename(path)}")
    return lens


def run_unit(lm, lenses, temps, norm, domain, cfg, monitored, log):
    tok_pair, tok_mode = NM.answer_tokens(lm.tokenizer, domain)
    texts, meta = NM.unit_prompts(lm.tokenizer, norm, domain, cfg["n_items"], cfg["seed"])
    cap = MI.Capture(lm, tok_pair)
    t0 = time.time()
    data = MI.capture_all(cap, texts, cfg["batch_size"], grads=True)
    src = cap.src_layers
    tw = MS.Twins(meta, cfg["n_items"])
    B = MS.boot_indices(cfg["n_items"], cfg["seed"])
    beh = MS.behaviour(meta, data["c_out"], tw, B)
    F = MS.features(meta)
    per_layer = {}
    contrasts = {name: MI.lens_contrasts(lm, lens, data["res"], tok_pair, src) for name, lens in lenses.items()}
    for j, l in enumerate(src):
        entry = {}
        for name in lenses:
            T = temps[name].get(l)
            s = MS.seen(contrasts[name][:, j], T, data["c_out"], tw, B) if T else None
            entry[name] = s
        readings = []
        if entry.get("main") and entry.get("control"):
            readings = [MS.rel_class(entry["main"]["rel_R"], beh, entry["main"]),
                        MS.rel_class(entry["control"]["rel_R"], beh, entry["control"]),
                        MS.seen_class(entry["main"]["Seen_R"])]
        entry["readings"] = readings
        entry["class"] = MS.unit_class(readings) if readings else "none"
        entry["coherence"] = MS.coherence_layer(data["res"][:, l], data["W"][:, l], meta, F)
        if l in monitored:
            X = data["res"][:, l]
            entry["probes"] = {k: MS.probe_ba(X, meta[k], meta["item"], cfg["seed"]) for k in ("y", "R", "D")}
        per_layer[str(l)] = entry
    follows = beh["acc_R0"] >= cfg["follow_min"] and beh["acc_R1"] >= cfg["follow_min"]
    out = dict(norm=norm, domain=domain, tok_pair=[int(t) for t in tok_pair], tok_mode=tok_mode,
               n_items=cfg["n_items"], n_prompts=len(texts), follows=bool(follows),
               behaviour=MS.strip(beh), layers=MS.strip(per_layer), seconds=round(time.time() - t0, 1),
               example_prompt=texts[1])
    arrays = dict(c_out=data["c_out"], **{f"lens_{k}": v for k, v in contrasts.items()},
                  **{k: v for k, v in meta.items()},
                  res_monitored=data["res"][:, monitored].astype(np.float16),
                  W_monitored=data["W"][:, monitored].astype(np.float16), monitored=np.array(monitored))
    return out, arrays


def first_readable(r):
    """Earliest layer at which the main lens reads the fact (Seen_D >= 0.1): the first place where the monitor can
    see the answer at all, and so where a rule applied later would leave it blind (added after the debug run on
    the 1.5B model, before any 7B data)."""
    for l in sorted(int(k) for k in r["layers"]):
        m = r["layers"][str(l)].get("main")
        if m and m["Seen_D"]["point"] >= MS.SEEN_D_MIN:
            return l
    return None


def count_classes(results, layer_of, cfg):
    cls = {"sighted": [], "blind": []}
    for r in results:
        l = layer_of(r)
        if l is None:
            continue
        e = r["layers"][str(l)]
        if r["follows"] and e["class"] in cls:
            cls[e["class"]].append(e["coherence"]["rho_c"])
    go = len(cls["sighted"]) >= cfg["n_go"] and len(cls["blind"]) >= cfg["n_go"]
    expl = {}
    if cls["sighted"] and cls["blind"]:
        from scipy.stats import mannwhitneyu
        s, b = np.array(cls["sighted"]), np.array(cls["blind"])
        auc = float(np.mean([(x < y) + 0.5 * (x == y) for x in s for y in b]))
        expl = dict(auc_rho_c_sighted_lower=auc,
                    p_one_sided=float(mannwhitneyu(s, b, alternative="less").pvalue))
    return dict(n_sighted=len(cls["sighted"]), n_blind=len(cls["blind"]), go=bool(go), exploratory=expl)


def layer_entry(e):
    m = e.get("main") or {}
    return dict(cls=e["class"], readings=e["readings"], rel_main=m.get("rel_R", {}).get("point"),
                rel_ctrl=(e.get("control") or {}).get("rel_R", {}).get("point"),
                seen_R=m.get("Seen_R", {}).get("point"), seen_D=m.get("Seen_D", {}).get("point"),
                rho_c=e["coherence"]["rho_c"], probe_y=e.get("probes", {}).get("y"))


def summarise(results, monitored, cfg):
    """Go rule: at one of the three fixed layers, or at each unit's first readable layer, at least n_go sighted
    and n_go blind units among the units that follow their norm."""
    verdict = {str(l): count_classes(results, lambda r, l=l: l, cfg) for l in monitored}
    verdict["first_readable"] = count_classes(results, first_readable, cfg)
    rows = []
    for r in results:
        row = dict(norm=r["norm"], domain=r["domain"], follows=r["follows"],
                   acc_R0=round(r["behaviour"]["acc_R0"], 3), acc_R1=round(r["behaviour"]["acc_R1"], 3),
                   Use_R=round(r["behaviour"]["Use_R"]["point"], 3))
        for l in monitored:
            row[f"L{l}"] = layer_entry(r["layers"][str(l)])
        ls = first_readable(r)
        row["first_readable"] = dict(layer=ls, **(layer_entry(r["layers"][str(ls)]) if ls is not None else {}))
        rows.append(row)
    return dict(monitored=monitored, verdict=verdict, go=any(v["go"] for v in verdict.values()), rows=rows)


def main(mode="debug", out_dir="lat_s2_pilot_out", overrides=None, allow_fallback_corpus=False, lm=None):
    cfg = {**COMMON, **PRESETS[mode], **(overrides or {})}
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "units"), exist_ok=True)
    ckpt_dir = "/content/lat_s2_ckpt" if os.path.isdir("/content") else out_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    log = Log(os.path.join(out_dir, "log.txt"))
    jlens.configure_logging()          # per-prompt progress of the lens fit
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    if gpu_gb < cfg["min_gpu_gb"]:
        raise RuntimeError(f"mode {mode} needs a GPU with at least {cfg['min_gpu_gb']} GB (L4, A100 or G4); "
                           f"this runtime has {gpu_gb:.0f} GB")
    if cfg["dim_batch"] == "auto":
        cfg["dim_batch"] = 16 if gpu_gb >= 30 else 8
    if cfg["batch_size"] == "auto":
        cfg["batch_size"] = 8 if gpu_gb >= 30 else 4
    log(f"LAT-S2 pilot, mode {mode}: {json.dumps({k: v for k, v in cfg.items() if k != 'norms'})}")
    if lm is None:
        lm = MI.load_model(cfg["model"], cfg["dtype"])
    L = lm.n_layers
    monitored = [L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1]
    log(f"model {cfg['model']}: {L} layers, d_model {lm.d_model}, dtype {next(lm._lm_head.parameters()).dtype}; "
        f"monitored layers {monitored}")
    pf_path = os.path.join(out_dir, "preflight.json")
    if os.path.exists(pf_path):
        pf = json.load(open(pf_path))
    else:
        pf = preflight(lm, cfg, log)
        json.dump(pf, open(pf_path, "w"), indent=1)
    n_follow = sum(r["follows"] for r in pf)
    log(f"preflight: {n_follow} of {len(pf)} units follow their norm")
    if n_follow < 2 * cfg["n_go"] and not cfg.get("ignore_preflight"):
        log(f"STOP before the lens fit: the go rule needs at least {2 * cfg['n_go']} followed units")
        summary = dict(mode=mode, model=cfg["model"], go=False, stopped="preflight", preflight=pf, config=cfg)
        json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
        return summary
    corpus, corpus_name = lens_corpus(cfg["n_lens_prompts"] + cfg["n_temp_prompts"], cfg["seed"],
                                      allow_fallback_corpus, log)
    fit_prompts, temp_prompts = corpus[:cfg["n_lens_prompts"]], corpus[cfg["n_lens_prompts"]:]
    lenses = {}
    if cfg["control"] == "half":
        h = len(fit_prompts) // 2
        lenses["main"] = get_lens(lm, fit_prompts[:h], os.path.join(out_dir, "lens_main.pt"),
                                  os.path.join(ckpt_dir, "lens_main.ckpt"), cfg, log)
        lenses["control"] = get_lens(lm, fit_prompts[h:], os.path.join(out_dir, "lens_control.pt"),
                                     os.path.join(ckpt_dir, "lens_control.ckpt"), cfg, log)
    else:
        lenses["main"] = get_lens(lm, fit_prompts, os.path.join(out_dir, "lens_main.pt"),
                                  os.path.join(ckpt_dir, "lens_main.ckpt"), cfg, log)
        try:
            lenses["control"] = jlens.JacobianLens.from_pretrained(cfg["community_repo"],
                                                                   filename=cfg["community_file"])
            log(f"control lens: {cfg['community_repo']} ({lenses['control']})")
        except Exception as e:
            log(f"control lens: community lens unavailable ({type(e).__name__}: {e}); "
                f"falling back to a second own lens on half of the prompts")
            h = len(fit_prompts) // 2
            lenses["control"] = get_lens(lm, fit_prompts[h:], os.path.join(out_dir, "lens_control.pt"),
                                         os.path.join(ckpt_dir, "lens_control.ckpt"), cfg, log)
    for name, lens in lenses.items():
        missing = [l for l in monitored if l not in lens.jacobians]
        if missing:
            log(f"WARNING lens {name} has no J for monitored layers {missing}")
    tpath = os.path.join(out_dir, "temperatures.json")
    if os.path.exists(tpath):
        temps = {k: {int(l): t for l, t in v.items()} for k, v in json.load(open(tpath)).items()}
    else:
        temps = {}
        for name, lens in lenses.items():
            temps[name], n_tok = MI.calibrate_temperatures(lm, lens, temp_prompts, sorted(lens.jacobians),
                                                           max_len=cfg["lens_max_len"])
            log(f"temperatures {name}: monitored {[temps[name].get(l) for l in monitored]} ({n_tok} tokens)")
        json.dump(temps, open(tpath, "w"), indent=1)
    results = []
    units = [(d, n) for d in cfg["domains"] for n in cfg["norms"]]
    for k, (domain, norm) in enumerate(units):
        path = os.path.join(out_dir, "units", f"{domain}__{norm}.json")
        if os.path.exists(path):
            results.append(json.load(open(path)))
            continue
        out, arrays = run_unit(lm, lenses, temps, norm, domain, cfg, monitored, log)
        json.dump(out, open(path, "w"), indent=1)
        np.savez_compressed(path.replace(".json", ".npz"), **arrays)
        results.append(out)
        cl = [out["layers"][str(l)]["class"] for l in monitored]
        log(f"unit {k + 1}/{len(units)} {domain}/{norm}: follows {out['follows']} "
            f"(acc {out['behaviour']['acc_R0']:.2f}/{out['behaviour']['acc_R1']:.2f}), classes {cl}, "
            f"rho_c {[round(out['layers'][str(l)]['coherence']['rho_c'], 2) for l in monitored]}, "
            f"{out['seconds']:.0f} s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarise(results, monitored, cfg)
    summary.update(mode=mode, model=cfg["model"], lens_corpus=corpus_name, config=cfg, preflight=pf,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    log(f"VERDICT: {'GO' if summary['go'] else 'STOP'} | " +
        "; ".join(f"{'L' + l if l.isdigit() else l}: {v['n_sighted']} sighted, {v['n_blind']} blind"
                  for l, v in summary["verdict"].items()))
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "debug", sys.argv[2] if len(sys.argv) > 2 else "lat_s2_pilot_out")
