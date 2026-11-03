"""LAT-S2 pilot: model loading, batched forward with residual capture, readout gradients for rho_c.

Layer convention (jlens): layer l is the output of block l; the final residual is the output of the last block,
before the final norm. The monitor at layer l reads h_l at the answer position (the last prompt token).
"""
import math
import numpy as np
import torch
import jlens


def pick_dtype(name="auto"):
    if name != "auto":
        return getattr(torch, name)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32


def load_model(model_name, dtype="auto", device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=pick_dtype(dtype), device_map=device)
    except (ImportError, ValueError, TypeError):
        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=pick_dtype(dtype)).to(device)
    return wrap(hf, tok)


def wrap(hf, tok):
    """HF model -> jlens LensModel (freezes parameters, finds blocks / final norm / LM head)."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return jlens.from_hf(hf, tok)


class Capture:
    """One batched forward pass that records, at the last position, the residual after every block, the model's
    answer contrast, and the gradient of the readout r . h_final[last] with respect to each layer's residual at
    the last position (w_c(l) = J_c(l)^T r; rho_c needs g_c = w_c . v)."""

    def __init__(self, lm, tok_pair):
        self.lm = lm
        self.L = lm.n_layers
        self.src_layers = list(range(self.L - 1))
        U = lm._lm_head.weight
        self.U2 = U[list(tok_pair)].detach().float()                       # (2, d)
        gamma = getattr(lm._final_norm, "weight", None)
        g = gamma.detach().float() if gamma is not None else torch.ones(U.shape[1], device=U.device)
        # readout without the Jacobian of the final RMSNorm: r = gamma * (U[t0] - U[t1]); Qwen2 RMSNorm scales by
        # weight (no 1 + w offset); for other families check the norm's formula before use.
        self.r = (g * (self.U2[0] - self.U2[1])).to(U.device)

    def encode(self, texts):
        enc = self.lm.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc.input_ids.to(self.lm.input_device)
        mask = enc.attention_mask.to(self.lm.input_device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids, mask, pos

    def run(self, texts, grads=True):
        ids, mask, pos = self.encode(texts)
        acts = {}

        def make_hook(i):
            def hook(module, inputs, output):
                t = output if torch.is_tensor(output) else output[0]
                if i == 0 and grads:
                    t.requires_grad_(True)
                acts[i] = t
            return hook

        handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(self.lm.layers)]
        try:
            with torch.set_grad_enabled(grads):
                self.lm._text_module(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
                h_final = acts[self.L - 1][:, -1, :]
                logits2 = torch.nn.functional.linear(self.lm._final_norm(h_final).float(), self.U2)
                c_out = (logits2[:, 0] - logits2[:, 1]).detach().cpu().numpy()
                res = np.stack([acts[l][:, -1, :].detach().float().cpu().numpy() for l in range(self.L)], 1)
                W = None
                if grads:
                    s = (h_final.float() @ self.r).sum()
                    gs = torch.autograd.grad(s, [acts[l] for l in self.src_layers])
                    W = np.stack([g[:, -1, :].float().cpu().numpy() for g in gs], 1)
        finally:
            for h in handles:
                h.remove()
        return dict(res=res, c_out=c_out, W=W)


def capture_all(cap, texts, batch_size, grads=True, log=None):
    out = dict(res=[], c_out=[], W=[])
    for b0 in range(0, len(texts), batch_size):
        r = cap.run(texts[b0:b0 + batch_size], grads=grads)
        for k in out:
            if r[k] is not None:
                out[k].append(r[k])
        if log and (b0 // batch_size) % 10 == 0:
            log(f"    batch {b0 // batch_size + 1}/{math.ceil(len(texts) / batch_size)}")
    return {k: (np.concatenate(v) if v else None) for k, v in out.items()}


@torch.no_grad()
def lens_contrasts(lm, lens, res, tok_pair, layers, chunk=256):
    """Lens answer contrast at each layer: unembed(J_l h_l) restricted to the two answer tokens.
    res: (n, L, d) residuals at the answer position. Returns (n, len(layers)) with NaN where the lens has no J_l."""
    U2 = lm._lm_head.weight[list(tok_pair)].float()
    dev = U2.device
    out = np.full((res.shape[0], len(layers)), np.nan)
    for j, l in enumerate(layers):
        if l not in lens.jacobians:
            continue
        J = lens.jacobians[l].to(dev)
        for c0 in range(0, res.shape[0], chunk):
            h = torch.from_numpy(res[c0:c0 + chunk, l]).to(dev)
            t = h @ J.T
            z = torch.nn.functional.linear(lm._final_norm(t.to(lm._lm_head.weight.dtype)).float(), U2)
            out[c0:c0 + chunk, j] = (z[:, 0] - z[:, 1]).cpu().numpy()
    return out


@torch.no_grad()
def calibrate_temperatures(lm, lens, texts, layers, max_len=128, skip_first=16,
                           grid=tuple(np.exp(np.linspace(np.log(0.25), np.log(16), 33)))):
    """One temperature per lens layer: minimises next-token cross-entropy of softmax(lens logits / T) on generic
    text (positions >= skip_first), as for the LAT-S1 lenses."""
    layers = [l for l in layers if l in lens.jacobians]
    acts = {l: [] for l in layers}
    targets = []
    for text in texts:
        ids = lm.encode(text, max_length=max_len)
        if ids.shape[1] <= skip_first + 1:
            continue
        with jlens.ActivationRecorder(lm.layers, at=layers) as rec:
            lm.forward(ids)
            for l in layers:
                acts[l].append(rec.activations[l][0, skip_first:-1].float().cpu())
        targets.append(ids[0, skip_first + 1:])
    n = int(sum(t.numel() for t in targets))
    temps = {}
    if not n:
        return temps, n
    dev = lm._lm_head.weight.device
    for l in layers:
        J = lens.jacobians[l].to(dev)
        tot = torch.zeros(len(grid))
        for a, target in zip(acts[l], targets):
            logits = lm.unembed(a.to(dev) @ J.T).float()                       # (P, V)
            for k, T in enumerate(grid):
                tot[k] += torch.nn.functional.cross_entropy(logits / T, target.to(dev), reduction="sum").cpu()
        temps[l] = float(grid[int(torch.argmin(tot))])
        del J
    return temps, n
