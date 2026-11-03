# LAT-S2 pilot: Jacobian-lens monitor and text-stated norms in Qwen2.5-7B-Instruct

2 November 2026 · Julia Ryzhkova, Siriona · ORCID 0009-0005-4194-6059

## Summary

In Qwen2.5-7B-Instruct, a Jacobian-lens (J-lens) monitor sees a norm stated in the prompt wherever it already sees the fact the norm acts on. Among the 21 norms the model follows, the pilot found 8–9 sighted norms and no blind norm at any monitored layer, so LAT-S2 stops in its current form.

The pilot carried the LAT-S1b measures over to a real model: 12 norms that flip a reported device state, two devices (a light and a door), 96 items per norm and device, a J-lens fitted on the model and an independently fitted community lens as control. The lens reads nothing at layers 6 and 13. It reads the door norms from layer 19 (8–9 sighted); for the light it starts reading the fact only at layers 21–23, where five norms read at about half strength (rel 0.50–0.55).

Linear probes decode the norm-applied answer at layer 13 (mean balanced accuracy 0.83), where the lens is silent. An audit traced this gap to input information. The probe is as accurate for norms the model ignores, one input feature predicts the decision as well as the probe in a smaller model, and the same gap appears for the plain fact without any norm.

## Background

LAT-S1 and LAT-S1b found that a J-lens monitor on a small trained receiver, fed a translated KV cache, sees a norm in some receivers and is blind to it in others: 9 sighted and 4 blind among the 13 classified receivers of LAT-S1b (release DOI [10.5281/zenodo.23842442](https://doi.org/10.5281/zenodo.23842442)). A receiver counts as blind when it applies the norm in its behaviour while the lens reading tracks the raw fact.

The pilot asked the simplest real-model version of that question. When an instruction-tuned model applies a norm given in the prompt, does the lens, at the first layer where it reads the fact, read the fact with the norm applied? A set of blind norms would have opened LAT-S2, a confirmatory study of what predicts them.

## Setup

**Model and lenses.** Qwen2.5-7B-Instruct (28 layers, bfloat16). The main J-lens comes from Anthropic's [jacobian-lens](https://github.com/anthropics/jacobian-lens) package, fitted on 32 WikiText-2 prompts of up to 128 tokens in 21.9 min. The control is a community lens fitted independently on 50 prompts ([anicka/jlens-qwen2.5-7b-instruct](https://huggingface.co/anicka/jlens-qwen2.5-7b-instruct)); per-layer temperatures for both come from 8 further WikiText prompts.

**Prompts.** The system message gives the assistant role and one norm sentence. The user message holds a three-line status log (one queried device, two distractor devices) and asks which state to report for the queried room; the assistant turn is prefilled with "Reported state:". The answer is the logit contrast of the two state words at the next token: "on" / "off" for the light, "open" / "closed" for the door.

**Norms and items.** Each of the 12 norms flips the report when on and comes with a matched neutral sentence when off. They motivate the flip in different ways: explicit negation, a miswired sensor, Opposite Day, code words, time since a toggle, a neighbouring room, security, inverted notation, a contrarian persona, logical NOT, a toggle press and test mode. Each of the 96 items per norm and device gives four prompts: the fact D (two states) crossed with the norm R (on or off).

**Measures.** They follow LAT-S1b. Use\_R and Use\_D are slopes of the model's answer on the ideal answer across norm twins and fact twins; Seen\_R and Seen\_D are slopes of the lens reading on the model's answer across the same twins, and rel\_R = Seen\_R / Seen\_D. A norm is sighted when the 95% bootstrap intervals (1000 resamples of items) put rel\_R at or above 0.8 and Seen\_R at or above 0.3, and blind when they put rel\_R at or below 0.5 and Seen\_R at or below 0.1.

**Classes and layers.** A class needs Use\_R and Use\_D ≥ 0.2, Seen\_D ≥ 0.1, and agreement of three readings: main-lens rel\_R, control-lens rel\_R and main-lens Seen\_R. Classes are taken at layers 6, 13 and 20 and, per norm, at the first readable layer: the earliest layer where the main lens has Seen\_D ≥ 0.1. Context coherence ρc and linear probes (standardise, PCA to at most 128 dimensions, logistic regression with C = 0.1, five folds grouped by item) are reported as exploratory.

**Decision rule.** Fixed in the code before the main run: go to LAT-S2 if, at one monitored layer or at the first readable layer, the norms the model follows include at least 3 sighted and at least 3 blind. A norm counts as followed when the answer accuracy is at least 0.8 with the norm off and with it on.

**Runs.** The main run took place on 2 November 2026, 07:07–08:05 UTC, on one NVIDIA A100-SXM4-40GB in Google Colab (code hash 7188c03ec41d3740). A debug run on Qwen2.5-1.5B-Instruct (24 items per norm and device) checked the pipeline earlier that day; its saved arrays feed part of the audit.

## Results

The model follows 21 of the 24 norm–device pairs, and the lens finds no blind norm among them at any layer. With the norm off, the model reports the logged state in every prompt of all 24 pairs. It ignores toggled\_since for both devices (accuracy with the norm on 0.00 and 0.01) and code\_words for the door (0.00).

| Layer | Followed pairs where the lens reads the fact | Sighted | Blind |
| --- | --- | --- | --- |
| 6 | 0 of 21 | 0 | 0 |
| 13 | 0 of 21 | 0 | 0 |
| 20 | 10 of 21 (all door) | 9 | 0 |
| First readable (door 19, light 21–23) | 21 of 21 | 8 | 0 |

**Door.** The lens starts reading the fact at layer 19 and reads the norm there as well: 8 of 10 door norms are sighted (rel\_R 0.97–1.31), and 9 of 10 at layer 20. The other two read at about 0.6 (inverted\_notation 0.60, after\_toggle 0.57) and get no class.

**Light.** The lens starts reading the fact only at layers 21–23. Four norms at layer 23 and code\_words at layer 21 read at about half strength (rel\_R 0.50–0.55). Six norms at layer 22 have a weak fact signal (Seen\_D 0.11–0.17) and unstable ratios, so their three readings disagree.

**Closest case and lens agreement.** The nearest thing to a blind norm is code\_words for the light at layer 21: its Seen\_R reading is blind (0.06), while its rel\_R is 0.55 and the control lens does not read the fact there yet. At the first readable layer, wherever the control lens also reads the fact (14 norms), the two lenses agree on rel\_R within 0.03.

## Audit of the probe–lens gap at middle layers

The probe–lens gap at layer 13 reflects input information at a stage before the answer forms. Four checks on saved arrays, with no new GPU time, support this.

1. **Probe validity.** probe\_y predicts y = D xor R, the norm-applied answer, from the residual at the answer position. Folds are grouped by item, and scaling and PCA are fitted inside each training fold, so no prompt of a test item enters training.
2. **Ignored norms (7B).** For the three norms the model never applies, probe\_y at layer 13 is 0.92, 0.91 and 0.85, above the mean of 0.83 for followed norms. At layer 20 it is 1.00 for all three, while with the norm on the model answers the opposite of y in every prompt. The probe reads the combination of fact and norm switch whether or not the model acts on it, in line with the linear XORs of feature pairs known in language-model residual streams ([Marks, 2024](https://www.lesswrong.com/posts/hjJXCn9GsskysDceS/what-s-up-with-llms-representing-xors-of-arbitrary-features)).
3. **Decision probe against input features (1.5B).** In the debug run, six pairs apply the norm on part of the prompts only (25–77%), which allows a direct probe for "the model applied the norm on this prompt". It reaches balanced accuracy 0.69–0.96 at layer 13 and 0.58–0.76 at layer 6. One input feature, the log line holding the queried device, predicts the same decision at 0.73–0.94; in two pairs the fact value does so at 0.77–0.86. In five of six pairs a single input feature does as well as the probe (difference at most 0.04). The sixth leaves a small excess: 0.69 against 0.58 for all input features together, on 48 prompts.
4. **Plain-fact control (1.5B).** Without any norm, a probe reads the fact at layer 13 with balanced accuracy 0.75–0.92, while the lens separates the two facts at AUC 0.53–0.64. At layer 20 both reach 1.00. The same gap thus appears for a fact the model reports without error.

At middle layers the residual carries the inputs and their combinations; the lens reads what has already taken the form of an answer. A causal follow-up of the layer-13 gap is therefore not warranted by these data.

## Discussion

The pilot shows inside a language model a pattern known from brain decoding: a reader near the output reads the output, and earlier readers find the ingredients. In a speech neuroprosthesis, arrays in ventral premotor area 6v classified words with 94% accuracy, while arrays in area 44 of Broca's area held little to no information about phonemes or words ([Willett et al., 2023](https://doi.org/10.1038/s41586-023-06377-x)). The J-lens sits at the same place relative to the answer: it reads late layers and stays silent where the model is still combining inputs.

Decoders also read more than is produced. A motor-cortex decoder picked up inner speech and silently counted numbers that participants did not intend to say, and its authors added a gate and an unlock keyword ([Kunz et al., 2025](https://doi.org/10.1016/j.cell.2025.06.015)). In the model, probes read the norm-applied answer for norms the model ignores; in both cases the work is to separate what is present from what goes to the output.

The LAT question has a counterpart in consciousness research, where report-based measures can mix the activity behind an experience with the activity behind reporting it ([Tsuchiya et al., 2015](https://doi.org/10.1016/j.tics.2015.10.002)). The J-lens resembles a report-based measure, and probes resemble no-report decoders. A model gives full access and free intervention, so it can serve as a test bench for such reading methods.

## Limitations

- One model family: Qwen2.5-7B-Instruct for the main run, Qwen2.5-1.5B-Instruct for the debug run.
- Simple material: two devices with binary states, three-line logs, and 12 norms that each flip one answer and are stated in plain text in the system message.
- Layers 6 and 13 carry no class by construction, since a class needs the lens to read the fact (Seen\_D ≥ 0.1). The informative readings are the first readable layer and layer 20 for the door.
- 96 items per pair; class intervals come from a bootstrap over items.
- The answer is a two-token logit contrast at one position; free generation was not studied.
- The main lens was fitted on 32 prompts; the community lens agrees with it wherever both read the fact.
- Descriptive pilot: the decision rule was fixed in code before the main run, and no confirmatory test was planned.
- The decision probe of the audit rests on the 1.5B debug run (48 prompts with the norm on per pair), made with an earlier build of the driver script; the three measurement modules are byte-identical to the main build.

## Decision and open questions

LAT-S2 stops in its current form: the go rule needed at least 3 blind norms, and the pilot found none. For norms stated in text, the J-lens on Qwen2.5-7B-Instruct reads the applied norm as soon as it reads the fact, in full for the door and at about half strength for part of the light norms.

Questions left to separate protocols:

- A latent channel between real models: a norm that reaches the receiver through a transferred KV cache, the setting where LAT-S1 and LAT-S1b found blind receivers.
- Norms learned in the weights, with no norm text in the prompt.
- Choices of an action or a tool call, where the open question is whether the monitor sees the reason for the choice.
- Other model families and larger models.

## Materials

This repository holds the code, the Colab notebook of the main run, the main-run summary and log, and the audit scripts, outputs and arrays. SHA256SUMS.txt lists every file with its SHA-256; the anchors are below.

| Item | Hash |
| --- | --- |
| Code hash of the four modules | `7188c03ec41d3740` |
| results\_main/summary.json | `9bb80b8d1c28ae551f411cc3b3388d7c7eb1ece1c1f097af3d84615e9217aa78` |
| results\_main/log.txt | `4a51da84d1fb4b62019211e8338b68bfb5968990752a88d3ddc3d6fda30874e8` |

The audit reruns on a CPU in a few minutes with the commands in README.md. The fitted main lens (694 MB), the per-pair files of the main run and the full debug outputs are held by the author.

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
