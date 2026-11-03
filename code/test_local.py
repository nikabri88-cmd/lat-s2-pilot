"""Local end-to-end smoke test of the LAT-S2 pilot on a tiny random Qwen2 with a locally trained tokenizer.
Numbers are meaningless; the test checks shapes, pairing, measures, rho_c, lenses, saving and the summary.
Also checks that batched left-padded capture equals one-by-one capture."""
import os
import sys
import shutil
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ext", "jacobian-lens"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import norms as NM
import model_io as MI
import run_pilot as RP

CHAT = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def build_tokenizer():
    class Plain:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
            return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")
    plain = Plain()
    corpus = []
    for domain in NM.DOMAINS:
        for norm in NM.NORM_NAMES:
            texts, _ = NM.unit_prompts(plain, norm, domain, 6, 1)
            corpus += texts
    corpus += [" on off open closed light door Reported state:"] * 200
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=SPECIAL,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = CHAT
    return tok


def main():
    torch.manual_seed(0)
    tok = build_tokenizer()
    for d in NM.DOMAINS:
        print("answer tokens", d, NM.answer_tokens(tok, d))
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=False)
    hf = Qwen2ForCausalLM(cfg).eval()
    lm = MI.wrap(hf, tok)
    print(lm)

    # batched (left padding) vs one by one
    texts, meta = NM.unit_prompts(tok, "negate", "light", 3, 5)
    texts = texts[:6]
    tp, _ = NM.answer_tokens(tok, "light")
    cap = MI.Capture(lm, tp)
    batched = cap.run(texts)
    single = [cap.run([t]) for t in texts]
    for k in ("res", "c_out", "W"):
        b = batched[k]
        s = np.concatenate([x[k] for x in single])
        err = np.abs(b - s).max() / (np.abs(s).max() + 1e-12)
        print(f"batched vs single {k}: max rel err {err:.2e}")
        assert err < 1e-4, k
    # gradient check: g = W . v equals the finite-difference change of r . h_final when h_l moves along v
    l = 3
    v = np.random.default_rng(0).normal(size=lm.d_model)
    v /= np.linalg.norm(v)
    eps = 1e-3

    def readout_with_shift(text, shift):
        ids, mask, pos = cap.encode([text])

        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            t2 = t.clone()
            t2[:, -1, :] += torch.tensor(shift, dtype=t.dtype)
            return t2 if torch.is_tensor(output) else (t2,) + tuple(output[1:])
        h = lm.layers[l].register_forward_hook(hook)
        acts = {}
        h2 = lm.layers[lm.n_layers - 1].register_forward_hook(
            lambda m, i, o: acts.__setitem__("f", o if torch.is_tensor(o) else o[0]))
        with torch.no_grad():
            lm._text_module(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        h.remove(); h2.remove()
        return float(acts["f"][0, -1].float() @ cap.r)
    fd = (readout_with_shift(texts[0], eps * v) - readout_with_shift(texts[0], -eps * v)) / (2 * eps)
    an = float(single[0]["W"][0, l] @ v)
    print(f"JVP check at layer {l}: analytic {an:.6f} finite-diff {fd:.6f}")
    assert abs(an - fd) < 1e-3 * max(1.0, abs(fd)), "gradient mismatch"

    out = os.path.join(HERE, "..", "lat_s2_pilot_localtest")
    shutil.rmtree(out, ignore_errors=True)
    summary = RP.main("debug", out, overrides=dict(n_items=12, n_lens_prompts=4, n_temp_prompts=2, dim_batch=8,
                                                   batch_size=5, domains=["light", "door"],
                                                   norms=["negate", "code_words", "after_toggle"], ignore_preflight=True),
                      allow_fallback_corpus=True, lm=lm)
    print("verdict", summary["verdict"])
    print("row example", summary["rows"][0])
    # resume: second call must load lenses, temperatures and units without recomputing
    summary2 = RP.main("debug", out, overrides=dict(n_items=12, n_lens_prompts=4, n_temp_prompts=2, dim_batch=8,
                                                    batch_size=5, domains=["light", "door"],
                                                    norms=["negate", "code_words", "after_toggle"], ignore_preflight=True),
                       allow_fallback_corpus=True, lm=lm)
    assert summary2["rows"] == summary["rows"]
    print("resume OK; files:", sorted(os.listdir(out)), len(os.listdir(os.path.join(out, "units"))))


if __name__ == "__main__":
    main()
