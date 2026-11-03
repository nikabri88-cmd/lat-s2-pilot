# LAT-S2 pilot: Jacobian-lens monitor and text-stated norms in Qwen2.5-7B-Instruct

Julia Ryzhkova, Siriona · ORCID 0009-0005-4194-6059 · 2 November 2026

In Qwen2.5-7B-Instruct, a Jacobian-lens (J-lens) monitor decodes the answer with a text-stated norm applied as soon as it decodes the fact the norm acts on. Among the 21 of 24 norm–device pairs the model follows, the lens shows 8–9 sighted norms and no blind norm at any monitored layer, so LAT-S2 stops in its current form. Linear probes decode the norm-applied answer at layer 13 where the lens is silent; an audit traces this gap to input information present before the answer forms.

The pilot carries over the measures of LAT-S1 and LAT-S1b ([doi:10.5281/zenodo.23842442](https://doi.org/10.5281/zenodo.23842442)). The full write-up, with setup, results, the audit, limitations and references, is in [NOTE.md](NOTE.md).

## Contents

- `NOTE.md` – the pilot note.
- `SHA256SUMS.txt` – SHA-256 of every file in the repository.
- `code/` – the four pipeline modules (`norms.py`, `model_io.py`, `measures_s2.py`, `run_pilot.py`), the notebook builder, the Colab notebook of the main run and two local tests. Code hash `7188c03ec41d3740`: the first 16 hex digits of SHA-256 over the sorted names and bytes of the four modules (rule in `make_colab_s2_pilot.py`).
- `results_main/` – `summary.json` and `log.txt` of the main run: Qwen2.5-7B-Instruct, one NVIDIA A100-SXM4-40GB in Google Colab, 2 November 2026, 07:07–08:05 UTC.
- `audit/` – scripts and outputs of the audit. `arrays_qwen2.5-1.5b_debug/` holds the saved arrays of 9 norm–device pairs from the Qwen2.5-1.5B-Instruct debug run: answer contrasts, lens readings at every layer, and residuals and answer gradients at layers 6, 13 and 20.

## Reproduce the audit

CPU only, a few minutes (Python 3.13.18, NumPy 2.5.3, scikit-learn 1.9.1):

```
python3 audit/audit_7b_probe_y.py results_main/summary.json
python3 audit/audit_probe_y.py audit/arrays_qwen2.5-1.5b_debug code
python3 audit/audit_layers.py audit/arrays_qwen2.5-1.5b_debug code
python3 audit/audit_surface.py audit/arrays_qwen2.5-1.5b_debug code
python3 audit/audit_fact.py audit/arrays_qwen2.5-1.5b_debug code
```

The outputs are in `audit/outputs/`.

## Held by the author

The fitted main lens (`lens_main.pt`, 694 MB), the per-pair JSON files and arrays of the main run, and the full debug-run outputs. The debug run used an earlier build of `run_pilot.py`; `norms.py`, `model_io.py` and `measures_s2.py` are byte-identical to the main build.

## Licence

CC BY-NC 4.0, see [LICENSE](LICENSE).

## Cite

See [CITATION.cff](CITATION.cff).
