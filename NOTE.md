# LAT-S2 pilot: Jacobian-lens monitor and text-stated norms in Qwen2.5-7B-Instruct

3 November 2026 (v1.0.1) · Julia Ryzhkova, Siriona · ORCID 0009-0005-4194-6059

## Summary

In Qwen2.5-7B-Instruct, a Jacobian-lens (J-lens) monitor showed no blind norm. Among the 21 norm–device pairs the model follows, none was classified blind at any of the 27 layers the lens reads, so LAT-S2 stops in its current form. Before the last lens layer, sighted classes appeared only for the door pairs: 8 of 10 at their first readable layer (19), 7 at layer 21 and all 10 from layer 22. The 11 light pairs stayed unclassified until the last layer (26), where the lens reading nearly coincides with the model's output.

The pilot carried the LAT-S1b measures over to a real model: 12 norms that flip a reported device state, two devices (a light and a door), 96 items per norm and device, a J-lens fitted on the model and an independently fitted community lens as control. The lens reads neither the fact nor the answer before layer 19. For the light it first reads the fact at layers 21–23. In five light pairs rel\_R is 0.50–0.55 there, which means that with the norm on the lens reading barely separates the two logged states; in the other six, rel\_R swings from 0.74–2.42 at layer 22, on a weak fact signal, to about 0.5 one layer later.

Linear probes decode the norm-applied answer at layer 13 (mean balanced accuracy 0.83), where the lens is silent. An audit found this gap consistent with input information. The probe is as accurate for the pairs the model ignores, in a smaller model one input feature predicts the decision as well as the probe in five of six pairs, and the same gap appears for the plain fact with the norm off. The audit is correlational, and part of it ran on the 1.5B debug model.

## Background

LAT-S1 and LAT-S1b found that a J-lens monitor on a small trained receiver, fed a translated KV cache, sees a norm in some receivers and is blind to it in others: 9 sighted and 4 blind among the 13 classified receivers of LAT-S1b (release DOI [10.5281/zenodo.23842442](https://doi.org/10.5281/zenodo.23842442)). A receiver counts as blind when it applies the norm in its behaviour while the lens reading tracks the raw fact.

The pilot asked the simplest real-model version of that question. When an instruction-tuned model applies a norm given in the prompt, does the lens, at the first layer where it reads the fact, read the fact with the norm applied? A set of blind norms would have opened LAT-S2, a confirmatory study of what predicts them.

## Setup

**Model and lenses.** Qwen2.5-7B-Instruct (28 layers, bfloat16). The main J-lens comes from Anthropic's [jacobian-lens](https://github.com/anthropics/jacobian-lens) package, fitted on 32 WikiText-2 prompts of up to 128 tokens in 21.9 min. The control is a community lens fitted independently on 50 prompts ([anicka/jlens-qwen2.5-7b-instruct](https://huggingface.co/anicka/jlens-qwen2.5-7b-instruct)); per-layer temperatures for both come from 8 further WikiText prompts.

**Prompts.** The system message gives the assistant role and one norm sentence. The user message holds a three-line status log (one queried device, two distractor devices) and asks which state to report for the queried room; the assistant turn is prefilled with "Reported state:". The answer is the logit contrast of the two state words at the next token: "on" / "off" for the light, "open" / "closed" for the door.

**Norms and items.** Each of the 12 norms flips the report when on and comes with a matched neutral sentence when off. They motivate the flip in different ways: explicit negation, a miswired sensor, Opposite Day, code words, time since a toggle, a neighbouring room, security, inverted notation, a contrarian persona, logical NOT, a toggle press and test mode. Each of the 96 items per norm and device gives four prompts: the fact D (two states) crossed with the norm R (on or off).

**Measures.** They follow LAT-S1b. Use\_R and Use\_D are slopes of the model's answer on the ideal answer across norm twins and fact twins; Seen\_R and Seen\_D are slopes of the lens reading on the model's answer across the same twins, and rel\_R = Seen\_R / Seen\_D. A norm is sighted when the 95% bootstrap intervals (1000 resamples of items) put rel\_R at or above 0.8 and Seen\_R at or above 0.3, and blind when they put rel\_R at or below 0.5 and Seen\_R at or below 0.1.

**Classes and layers.** A class needs Use\_R and Use\_D ≥ 0.2, Seen\_D ≥ 0.1, and agreement of three readings: main-lens rel\_R, control-lens rel\_R and main-lens Seen\_R. Classes are taken at layers 6, 13 and 20 and, per norm, at the first readable layer: the earliest layer where the main lens has Seen\_D ≥ 0.1. Context coherence ρc (stored in the per-pair files) and linear probes (standardise, PCA to at most 128 dimensions, logistic regression with C = 0.1, five folds grouped by item) are exploratory.

**Decision rule.** Fixed in the code before the main run: go to LAT-S2 if, at one monitored layer or at the first readable layer, the norms the model follows include at least 3 sighted and at least 3 blind. A norm counts as followed when the answer accuracy is at least 0.8 with the norm off and with it on.

**Versions.** The notebook installs jacobian-lens without a pinned commit and loads the models without a revision, so the versions below are inferred after the run. jacobian-lens commit 581d398613e5602a5af361e1c34d3a92ea82ba8e (2 July 2026, still the latest commit on 3 November 2026). Hugging Face main revisions of Qwen2.5-7B-Instruct (last updated 12 January 2025), Qwen2.5-1.5B-Instruct (25 September 2024) and the control lens (7 July 2026; file qwen2.5-7b-instruct\_jlens.pt, 693,641,956 bytes). The library versions of the Colab runtime were not recorded.

**Runs.** The main run took place on 2 November 2026, 07:07–08:05 UTC, on one NVIDIA A100-SXM4-40GB in Google Colab (code hash 7188c03ec41d3740). A debug run on Qwen2.5-1.5B-Instruct (24 items per norm and device) checked the pipeline earlier that day; its two lenses were fitted on 4 prompts each under the debug preset of the current build, and its saved arrays feed part of the audit.

## Results

The model follows 21 of the 24 norm–device pairs, and the lens finds no blind norm among them at any layer. With the norm off, the model reports the logged state in every prompt of all 24 pairs. It ignores toggled\_since for both devices (accuracy with the norm on 0.00 and 0.01) and code\_words for the door (0.00). A preflight on 8 items per pair, logged before the lens fit, counted 20 followed pairs (light/code\_words at 0.75); the go rule uses the 96-item values, where light/code\_words reaches 0.92.

| Layer | Followed pairs where the lens reads the fact | Sighted | Blind |
| --- | --- | --- | --- |
| 0–18 (including 6 and 13) | 0 of 21 | 0 | 0 |
| 19 | 10 of 21 (all door) | 8 | 0 |
| 20 | 10 of 21 (all door) | 9 | 0 |
| 21 | 11 of 21 | 7 (all door) | 0 |
| 22–25 | 14–21 of 21 | 10 (all door) | 0 |
| 26 (last lens layer) | 21 of 21 | 21 | 0 |
| First readable (door 19, light 21–23) | 21 of 21 | 8 | 0 |

A recheck from the per-pair files covers all 27 lens layers and finds no blind pair at any of them (audit/outputs/recheck\_main\_units.txt).

**Door.** The lens starts reading the fact at layer 19 and reads the norm there as well: 8 of 10 door norms are sighted (rel\_R 0.97–1.31), 9 at layer 20, 7 at layer 21 and all 10 from layer 22. At layer 19 the other two show rel\_R of about 0.6 (inverted\_notation 0.60, after\_toggle 0.57) and get no class; at layer 21 inverted\_notation, logical\_not and test\_mode drop out of the class.

**Light.** The lens starts reading the fact only at layers 21–23. In four pairs at layer 23 and code\_words at layer 21, rel\_R is 0.50–0.55. rel\_R is a ratio of two effects, not a detection rate: near 0.5 it means that with the norm on, the lens reading barely separates the two logged states, so the lens shows neither the raw fact nor the norm-applied answer. Six pairs at layer 22 have a weak fact signal (Seen\_D 0.11–0.17); their rel\_R of 0.74–2.42 falls to about 0.5 one layer later, the control lens reaches the fact only at layer 23, and Seen\_R stays between the cut-offs, so they get no class. All 11 light pairs become sighted only at layer 26, where the lens reading nearly coincides with the output.

**Closest cases and lens agreement.** At the first readable layer, the nearest thing to a blind norm is code\_words for the light (layer 21): its Seen\_R reading is blind (0.06), while its rel\_R is 0.55 and the control lens does not read the fact there yet. Across all layers, inverted\_notation for the light at layer 24 comes closest: both lenses read the fact, and the upper bounds of the three readings (0.503, 0.504 and 0.12) stop just short of the blind cut-offs (0.5, 0.5 and 0.1). The two lenses agree on rel\_R within 0.03 at the first readable layer (14 pairs where both read the fact) and give the same rel\_R reading in 115 of 116 pair–layer cells where both read the fact; at layers 20–21 their rel\_R values differ by up to 0.13.

## Audit of the probe–lens gap at middle layers

The probe–lens gap at layer 13 is consistent with input information that has not yet taken the form of an answer. Four checks on saved arrays, with no new GPU time, support this reading.

1. **Probe validity.** probe\_y predicts y = D xor R, the norm-applied answer, from the residual at the answer position. Folds are grouped by item, and scaling and PCA are fitted inside each training fold, so no prompt of a test item enters training.
2. **Ignored pairs (7B).** For the three pairs the model ignores (two norms; accuracy with the norm on 0.00–0.01), probe\_y at layer 13 is 0.92, 0.91 and 0.85, above the mean of 0.83 for followed pairs. At layer 20 it is 1.00 for all three, while with the norm on the model answers the opposite of y in 574 of 576 prompts. The probe reads the combination of fact and norm switch whether or not the model acts on it, in line with the linear XORs of feature pairs reported in language-model residual streams ([Marks, 2024](https://www.lesswrong.com/posts/hjJXCn9GsskysDceS/what-s-up-with-llms-representing-xors-of-arbitrary-features)).
3. **Decision probe against input features (1.5B).** In the debug run, six pairs apply the norm on part of the prompts only (25–77%), which allows a direct probe for "the model applied the norm on this prompt". It reaches balanced accuracy 0.69–0.96 at layer 13 and 0.58–0.76 at layer 6. One input feature, the log line holding the queried device, predicts the same decision at 0.73–0.94 in five of six pairs (0.50 in the sixth); in two pairs the fact value does so at 0.77–0.86. In five of six pairs a single input feature does as well as the probe (difference at most 0.04), although the best of eight features is picked on the same folds, which slightly favours the features. The sixth keeps an excess of 0.11 (0.69 against 0.58 for all input features together), which 48 prompts cannot resolve.
4. **Plain-fact control (1.5B).** With the norm off (the neutral sentence in place), a probe reads the fact at layer 13 with balanced accuracy 0.75–0.92, while the lens separates the two facts at AUC 0.53–0.64. At layer 20 both reach 1.00. The same gap thus appears for a fact the model reports without error.

These checks indicate that probe accuracy at layer 13 is weak evidence for a decision hidden from the lens. They are correlational: no causal intervention was run, and checks 3 and 4 used the 1.5B debug model, so they do not establish where the answer forms.

## Discussion

The parallels below are analogies that this pilot does not test. Brain decoding shows a similar pattern: a reader near the output reads the output, and earlier readers find the ingredients. In a speech neuroprosthesis, arrays in ventral premotor area 6v classified words with 94% accuracy, while arrays in area 44 of Broca's area held little to no information about phonemes or words ([Willett et al., 2023](https://doi.org/10.1038/s41586-023-06377-x)). In the model, the J-lens likewise reads late layers and stays silent where the inputs are still being combined.

Decoders also read more than is produced. A motor-cortex decoder picked up inner speech and silently counted numbers that participants did not intend to say, and its authors added a gate and an unlock keyword ([Kunz et al., 2025](https://doi.org/10.1016/j.cell.2025.06.015)). In the model, probes read the norm-applied answer for norms the model ignores; in both cases the work is to separate what is present from what goes to the output.

The LAT question has a counterpart in consciousness research, where report-based measures can mix the activity behind an experience with the activity behind reporting it ([Tsuchiya et al., 2015](https://doi.org/10.1016/j.tics.2015.10.002)). By analogy, the J-lens resembles a report-based measure and probes resemble no-report decoders. A model gives full access and free intervention, so it can serve as a test bench for such reading methods.

## Limitations

- One model family: Qwen2.5-7B-Instruct for the main run, Qwen2.5-1.5B-Instruct for the debug run.
- Simple material: two devices with binary states, three-line logs, and 12 norms that each flip one answer and are stated in plain text in the system message.
- Layers 6 and 13 carry no class because the lens reading of the fact stays at or below 0.03 there, below the 0.1 a class needs; the same holds for every layer before 19. The informative readings are layers 19–26.
- 96 items per pair; class intervals come from a bootstrap over items.
- The answer is a two-token logit contrast at one position; free generation was not studied.
- The lens measures the effect of the norm on that contrast. It does not decode the content of the rule or establish why the model answers as it does.
- The main lens was fitted on 32 prompts. The community lens gives the same rel\_R reading in 115 of 116 pair–layer cells where both read the fact, with rel\_R differences up to 0.13 at layers 20–21.
- Descriptive pilot: the decision rule was fixed in code before the main run, and no confirmatory test was planned.
- The decision probe of the audit rests on the 1.5B debug run (48 prompts with the norm on per pair), made with an earlier build of the driver script; the three measurement modules are byte-identical to the main build.

## Decision and open questions

LAT-S2 stops in its current form: the go rule needed at least 3 blind norms, and the pilot found none. For norms stated in text, the pilot found no blind norm in Qwen2.5-7B-Instruct. Most door pairs were sighted from layer 19 and all of them from layer 22, while the light pairs stayed unclassified until the last layer. A clear side finding is the gap between decoding the norm-applied answer and carrying out the instruction, shown by the audit on three ignored pairs (two norms).

Questions left to separate protocols:

- A latent channel between real models: a norm that reaches the receiver through a transferred KV cache, the setting where LAT-S1 and LAT-S1b found blind receivers.
- Norms learned in the weights, with no norm text in the prompt.
- Choices of an action or a tool call, where the open question is whether the monitor sees the reason for the choice.
- Other model families and larger models.

## Materials

This repository holds the code, the Colab notebook of the main run, the main-run summary and log, the per-pair result files of the main run (results\_main/units), and the audit scripts, outputs and arrays. SHA256SUMS.txt lists every file with its SHA-256; the anchors are below.

| Item | Hash |
| --- | --- |
| Code hash of the four modules | `7188c03ec41d3740` |
| results\_main/summary.json | `9bb80b8d1c28ae551f411cc3b3388d7c7eb1ece1c1f097af3d84615e9217aa78` |
| results\_main/log.txt | `4a51da84d1fb4b62019211e8338b68bfb5968990752a88d3ddc3d6fda30874e8` |

The audit reruns on a CPU in a few minutes with the commands in README.md. The fitted main lens (694 MB), the per-pair arrays of the main run (residuals and gradients, about 16 MB per pair) and the full debug outputs are held by the author.

## Changes in v1.0.1

- Corrected the main statement. Version 1.0.0 said that the monitor sees the norm wherever it sees the fact, which is stronger than the classification supports: clear sighted readings appeared only for the door pairs, and the light pairs stayed unclassified until the last layer. No pair was blind in either version.
- Stated the audit conclusion as what correlational checks support, and marked the neuroscience parallels as untested analogies.
- Added the per-pair result files of the main run (results\_main/units) and audit/recheck\_main\_units.py, which recomputes every reading and class from them and counts classes at all 27 lens layers.
- Recorded the versions of jacobian-lens, the models and the control lens.
- Added the door's dip at layer 21, the closest cases across all layers, the meaning of rel\_R near 0.5 and the exact count of ignored prompts, and gave the lens agreement for all layers.
- Removed the compiled Python caches (code/\_\_pycache\_\_) included in v1.0.0: running the audit rewrote them, so `sha256sum -c` failed on two lines afterwards. The README commands now run Python with `-B`.
- These changes follow two independent checks: one of v1.0.0, which reproduced the file hashes, the code hash and all audit outputs, and one of the draft of v1.0.1 against the per-pair files.

## References

- Anthropic. jacobian-lens (software). [github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
- anicka. jlens-qwen2.5-7b-instruct (community J-lens). [huggingface.co/anicka/jlens-qwen2.5-7b-instruct](https://huggingface.co/anicka/jlens-qwen2.5-7b-instruct)
- Kunz, E. M., Abramovich Krasa, B., Kamdar, F., et al., Willett, F. R. (2025). Inner speech in motor cortex and implications for speech neuroprostheses. Cell. [doi:10.1016/j.cell.2025.06.015](https://doi.org/10.1016/j.cell.2025.06.015)
- Marks, S. (2024). What's up with LLMs representing XORs of arbitrary features? LessWrong. [lesswrong.com](https://www.lesswrong.com/posts/hjJXCn9GsskysDceS/what-s-up-with-llms-representing-xors-of-arbitrary-features)
- Qwen Team. Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct (model weights). [huggingface.co/Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Ryzhkova, J. (2026). LAT-S1 and LAT-S1b: what a Jacobian-lens monitor sees in a latent KV-cache channel (v1.0.0). Zenodo. [doi:10.5281/zenodo.23842442](https://doi.org/10.5281/zenodo.23842442)
- Tsuchiya, N., Wilke, M., Frässle, S., Lamme, V. A. F. (2015). No-report paradigms: extracting the true neural correlates of consciousness. Trends in Cognitive Sciences, 19(12), 757–770. [doi:10.1016/j.tics.2015.10.002](https://doi.org/10.1016/j.tics.2015.10.002)
- Willett, F. R., Kunz, E. M., Fan, C., et al. (2023). A high-performance speech neuroprosthesis. Nature, 620, 1031–1036. [doi:10.1038/s41586-023-06377-x](https://doi.org/10.1038/s41586-023-06377-x)
- WikiText-2 (lens fitting corpus). [huggingface.co/datasets/Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
