# LAT-S2 pilot: Jacobian-lens monitor and text-stated norms in Qwen2.5-7B-Instruct

Julia Ryzhkova, Siriona · ORCID 0009-0005-4194-6059 · 3 November 2026 (v1.0.1)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23912045.svg)](https://doi.org/10.5281/zenodo.23912045)

In Qwen2.5-7B-Instruct, a Jacobian-lens (J-lens) monitor showed no blind norm. Among the 21 of 24 norm–device pairs the model follows, none was classified blind at any of the 27 lens layers, so LAT-S2 stops in its current form. Before the last lens layer, sighted classes appeared only for the door pairs (8 of 10 at layer 19, 7 at layer 21, all 10 from layer 22); the light pairs stayed unclassified until the last layer. Linear probes decode the norm-applied answer at layer 13 where the lens is silent; a correlational audit finds this gap consistent with input information.

This is version 1.0.1. It corrects the main statement of version 1.0.0 and adds the per-pair result files of the main run; see "Changes in v1.0.1" in [NOTE.md](NOTE.md).

The pilot carries over the measures of LAT-S1 and LAT-S1b ([doi:10.5281/zenodo.23842442](https://doi.org/10.5281/zenodo.23842442)). The full write-up, with setup, results, the audit, limitations and references, is in [NOTE.md](NOTE.md).

## Contents

- `NOTE.md` – the pilot note.
- `SHA256SUMS.txt` – SHA-256 of every file in the repository.
- `code/` – the four pipeline modules (`norms.py`, `model_io.py`, `measures_s2.py`, `run_pilot.py`), the notebook builder, the Colab notebook of the main run and two local tests. Code hash `7188c03ec41d3740`: the first 16 hex digits of SHA-256 over the sorted names and bytes of the four modules (rule in `make_colab_s2_pilot.py`).
- `results_main/` – `summary.json`, `log.txt` and the 24 per-pair result files (`units/`) of the main run: Qwen2.5-7B-Instruct, one NVIDIA A100-SXM4-40GB in Google Colab, 2 November 2026, 07:07–08:05 UTC.
- `audit/` – scripts and outputs of the audit. `arrays_qwen2.5-1.5b_debug/` holds the saved arrays of 9 norm–device pairs from the Qwen2.5-1.5B-Instruct debug run: answer contrasts, lens readings at every layer, and residuals and answer gradients at layers 6, 13 and 20.

## Reproduce the audit

CPU only, a few minutes (Python 3.13.18, NumPy 2.5.3, scikit-learn 1.9.1):

```
python3 -I -B audit/recheck_main_units.py results_main code
python3 -I -B audit/audit_7b_probe_y.py results_main/summary.json
python3 -I -B audit/audit_probe_y.py audit/arrays_qwen2.5-1.5b_debug code
python3 -I -B audit/audit_layers.py audit/arrays_qwen2.5-1.5b_debug code
python3 -I -B audit/audit_surface.py audit/arrays_qwen2.5-1.5b_debug code
python3 -I -B audit/audit_fact.py audit/arrays_qwen2.5-1.5b_debug code
```

The outputs are in `audit/outputs/`.

## Held by the author

The fitted main lens (`lens_main.pt`, 694 MB), the per-pair arrays of the main run (about 16 MB per pair), and the full debug-run outputs. The debug run used an earlier build of `run_pilot.py`; `norms.py`, `model_io.py` and `measures_s2.py` are byte-identical to the main build.

## Licence

CC BY-NC 4.0, see [LICENSE](LICENSE).

## Cite

Ryzhkova, J. (2026). LAT-S2 pilot: Jacobian-lens monitor and text-stated norms in Qwen2.5-7B-Instruct (v1.0.1). Zenodo.

Version 1.0.0: https://doi.org/10.5281/zenodo.23912045. The DOI of version 1.0.1 is added after its release.

See also [CITATION.cff](CITATION.cff).
