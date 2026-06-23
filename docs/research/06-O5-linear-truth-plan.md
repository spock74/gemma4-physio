# O5 — Linear Representation of Truth, all layers, on Gemma 3 (Design)

This document defines the research plan to reproduce and evaluate the core findings of *"Linear Representation of Truth in LLMs"* (Marks & Tegmark 2023, arXiv:2310.18168) on a fully-instrumented `google/gemma-3-270m-it` and `google/gemma-3-4b-it` model on your Apple Silicon hardware.

## Scientific Objective

Does Gemma 3 represent the "truthfulness" of factual statements along a single, generalization-capable linear direction in its activation space? And is this direction causally load-bearing?

### Hypotheses to Test
1. **H1 (Linear Separability):** There exists a linear direction in residual stream activation space ($d_{\text{truth}}$) that separates true factual statements from false ones, generalizing to held-out statement categories.
2. **H2 (Causal Sufficiency/Steering):** Adding $+c \cdot d_{\text{truth}}$ to residual activations on an ambiguous or false statement steers the model's output distribution toward predicting the true completion.
3. **H3 (Causal Necessity/Ablation):** Projecting out $d_{\text{truth}}$ (ablation) decreases the model's factual accuracy on true statements.

---

## Stimuli & Dataset Design

We will construct a programmatic dataset of true/false statements across multiple categories (geography, basic arithmetic, and simple biology) to avoid loading external online datasets.

### Corpus Structure (`data/eval/truth_contrast.json`)
We will create 80-100 statements structured as follows:
* **Geography:**
  * *"Paris is in France."* (True)
  * *"Paris is in Japan."* (False)
* **Arithmetic:**
  * *"Two plus two is four."* (True)
  * *"Two plus two is five."* (False)
* **Simple Facts:**
  * *"Dogs are mammals."* (True)
  * *"Dogs are reptiles."* (False)

For each statement, we will isolate the activation at the final token (period) before the model makes any subsequent generation.

---

## Rigor Bar (Project Standards)

We will apply the repository's strict methodological controls:
1. **Disjoint splits:** Train/fit the truth direction $d_{\text{truth}}$ on 80% of the categories (e.g., Geography + Arithmetic) and evaluate separation and steering on the remaining 20% (e.g., Simple Facts) to test true generalization.
2. **Specificity Controls:** Every steering or ablation intervention will be compared against $N=20$ random unit directions and at least 1 orthogonal direction.
3. **Paired readouts:** When steering/ablating, we measure the change in the probability of the factual target token, the rank change, and the Kullback-Leibler (KL) divergence of the next-token distribution.
4. **Position Localization:** Interventions will act at specific layers and token positions (last position) to avoid global activation destruction.

---

## Experimental Phases

### Phase O5.1 — Probing & Separability (AUC Profile)
* **Action:** Capture residuals at all layers. Fit $d_{\text{truth}}$ via difference-of-means (and a simple linear probe in PyTorch) on the training split.
* **Metric:** Compute ROC AUC on the held-out validation split.
* **Output:** Plot/table the AUC across all layers (expected peak in middle-to-late layers).

### Phase O5.2 — Causal Steering (Sufficiency)
* **Action:** Steer the model during a forward pass on ambiguous prompts (e.g., *"Is Paris in Japan? Answer Yes or No: "*) by adding $+c \cdot d_{\text{truth}}$.
* **Metric:** Change in log-probability of target answers ("Yes"/"No"). Verify if $d_{\text{truth}}$ outperforms random/orthogonal controls by a factor of $>2$.

### Phase O5.3 — Causal Ablation (Necessity)
* **Action:** Ablate $d_{\text{truth}}$ on known true statements.
* **Metric:** Measure the drop in correct-token logits, rank demotion, and KL divergence compared to random controls.

---

## Verification Plan

### Success Criteria
* **Linear Probing:** Held-out validation AUC $\ge 0.85$ at one or more layers.
* **Steering specificity:** $d_{\text{truth}}$ steering log-prob change is $\ge 2\times$ larger than the maximum change induced by random/orthogonal controls.
