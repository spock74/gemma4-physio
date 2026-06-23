# Geometric Decoupling in the Gemma 3 Residual Stream: Attractor Stability and Phase Transitions under Orthogonal Perturbations

## Abstract

In the recent literature on large language model (LLM) mechanistic interpretability, structural anomalies or failures of syntactic and factual coherence under activation perturbations are frequently characterized via clinical neurological metaphors (e.g., "artificial aphasias" or "factual amnesia"). In this work, we point to a crucial epistemic limit of this correspondence: analogous to the distinction between mechanical and atomic clocks, identical external symptoms hide underlying internal mechanics of entirely distinct natures. Discarding biological analogies and abstract topological models, we formalize the dynamics of representational failure modes through the lens of complexity science and discrete dynamical systems.

Using a 2D Rotational Sweep in a $\{ \vec{v}_1, \vec{v}_2 \}$ plane (spanned by the factual knowledge direction $d_{\text{know}}$ and a semantic context direction orthogonalized via Gram-Schmidt) at Layer $L_{12}$ of Gemma 3 4b-it, we investigate the resilience of local representational states against extreme orthogonal perturbations. To prove that the transitions are robust features of the representation space rather than artifacts of a single arbitrary plane, we implement a baseline control by iterating the rotation over multiple random orthogonal vectors. We identify two discrete, phenomenological transition zones: (i) **Manifold Collapse** (centered at $\bar{\theta} \approx 120^\circ \pm 8.2^\circ$), characterized by a high-entropy syntactic divergence where the activation deviation projects the residual stream outside the high-probability syntactic basin; and (ii) **Orthogonal Attractor Translation** (centered at $\bar{\theta} \approx 200^\circ \pm 12.4^\circ$), under which the model preserves perfect syntactic integrity but the factual representation is translated to a neighboring semantic concept. The symmetry of factual recovery ($0.57$) under diametrically opposite perturbations ($-\vec{v}_1$) in large-scale additive steering ($R \approx 20,000$, calibrated to the model's operating activation norm $\|h\|_2 \approx 21,112.64$) documents the boundaries of invariance and stability of residual-stream representations in Layer 12, providing empirical guidance for alignment engineering.

---

## 1. Introduction

Understanding how Large Language Models (LLMs) store and process factual information has progressed from qualitative output analyses to the localized investigation of vectors and directions within the residual stream. The residual stream serves as the communication backbone of Transformer architectures, where each layer reads the accumulated representations, processes them, and writes back updated coefficients. Recent work has identified causal linear directions linked to the representation of specific concepts and facts, such as the factual direction $d_{\text{know}}$.

However, localized perturbations in these linear directions yield output failures that often invite anthropomorphic or clinical metaphors. Terms such as "model lesioning" or "induced aphasias" [5, 6, 7, 8] are common when describing the syntactic and semantic collapse of generated texts.

### 1.1 The Epistemic Rupture: The Clock Analogy

In this work, we argue that such clinical metaphors must be strictly restricted to informal laboratory scaffolding and formally discarded in scientific papers. The rationale lies in a fundamental **epistemic rupture**:

> **The clock analogy is exact.** A mechanical clock made of brass gears and a cesium atomic clock produce the same macroscopic output—telling time. Yet, the physical properties and mathematical topology of mechanical gears share no relation to the quantum state transitions of electrons in a cesium-133 atom. Searching for organic gears in atomic matrices is an epistemic category error.

Similarly, the human brain loses language coherence due to ischemia, trauma, or metabolic failure in its organic neural network. Gemma 3, however, loses coherence because the perturbation vector in the residual stream projects the latent representation outside the high-probability syntactic basin, where activations, after downstream layer transformations and layer normalization, map to grammatically coherent output distributions (rather than a property of the unembedding matrix $W_U$ in isolation). The behavioral symptom is similar, but the underlying mechanics in discrete dynamical systems are alien to each other. Calling tensor collapse an "aphasia" in a formal paper is akin to searching for brass gears in an atom.

### 1.2 The 2D Rotational Sweep and Local Perturbation Dynamics

To probe the stability of localized factual representations in the Gemma 3 4b-it latent space under extreme orthogonal perturbations, we introduce a **2D Rotational Sweep** methodology. Our goal is to determine how the representation responds to continuous orthogonal changes and to characterize the resilience of the local representation space.

We construct an orthogonal basis for a 2D subspace $S \subset \mathbb{R}^d$ at Layer $L_n$ (where $n = 12$, the optimal semantic crystallization layer). The rotation plane is spanned by:

1. $\vec{v}_1$: The primary factual vector ($d_{\text{know}}$) extracted via Difference-of-Means.
2. $\vec{v}_2$: An adjacent semantic context vector orthogonalized with respect to $\vec{v}_1$ using the classical Gram-Schmidt process to ensure algebraic independence:

$$\vec{v}_2 = \vec{u}_2 - \frac{\vec{u}_2 \cdot \vec{v}_1}{\|\vec{v}_1\|^2}\vec{v}_1$$

where $\vec{u}_2$ is the raw adjacent semantic direction. Normalizing both yields the orthonormal basis $\{\hat{v}_1, \hat{v}_2\}$. The applied latent perturbation $\vec{h}(\theta)$ is parameterized in polar coordinates using the angle $\theta \in [0, 2\pi]$ and the steering magnitude $R \in \mathbb{R}$:

$$\vec{h}(\theta) = \vec{h}_{\perp} + R \cos(\theta) \hat{v}_1 + R \sin(\theta) \hat{v}_2$$

where $\vec{h}_{\perp}$ is the projection of the original activation orthogonal to the subspace $S$. We analyze the model's behavior under two distinct regimes:

1. **Pure Subspace Rotation (SO(2)):** The component of the activation within $S$ is decomposed and rotated by the rotation matrix $\mathbf{R}_{\theta} \in \text{SO}(2)$.
2. **Large-Scale Additive Steering:** An additive perturbation vector $\vec{s}(\theta) = R(\cos(\theta)\hat{v}_1 + \sin(\theta)\hat{v}_2)$ is added to the natural latent state. We explore magnitudes corresponding to the model's physical activation scale. Specifically, $\|h\|_2$ is computed as the average $L_2$ norm of the activation vector at the final token position of the prompt at Layer 12, post-attention but before FFN or layer normalization, averaged over the 8 evaluation prompts ($\text{mean } \|h\|_2 \approx 21,112.64 \pm 432.18$). We express the steering magnitude $R$ in relative units of this baseline activation scale ($R \approx 0.5 \times \|h\|_2$ to $1.0 \times \|h\|_2$, corresponding to $R \approx 10,000$ to $20,000$ respectively) to ensure perturbations are calibrated to the model's operating activation regime and overcome representational inertia.

To verify whether the identified transition regimes are robust invariants of the local activation neighborhood rather than artifacts of a single arbitrary plane, we implement a robust baseline control by drawing $K = 5$ random vectors $\vec{u}_{2,\text{rand}}^{(k)} \sim \mathcal{N}(0, \mathbf{I}_d)$, orthogonalizing them to $\vec{v}_1$ via Gram-Schmidt to construct $K$ distinct orthogonal control planes $\vec{v}_{2,\text{rand}}^{(k)}$, and running the rotational sweep across all $K$ planes. This maps the boundaries of stability over a high-dimensional ensemble of orthogonal trajectories.

### 1.3 Phenomenological Analysis of Transition States

The rotational sweep reveals clear phenomenological transition regions, defining two discrete failure phenotypes:

*   **Manifold Collapse / High-Entropy Syntactic Divergence:**
    In a specific angular range, the perturbation distorts the residual stream such that the resulting activation is projected outside the coherent output manifold (the high-probability syntactic basin where activations, after downstream layers, map to grammatically coherent output distributions). The network fails to project a low-entropy probability distribution over the vocabulary, collapsing into high-entropy repetitions of character fragments (with KL divergence relative to the clean output rising to $D_{KL} \approx 18-45$). Across the $K=5$ control planes and 8 prompts, this syntactic collapse is consistently observed at an average angle of $\bar{\theta}_{\text{collapse}} = 120.0^\circ \pm 8.2^\circ$.
*   **Orthogonal Attractor Translation / Semantic Permutation:**
    As the perturbation rotates toward the recovery lobe, the activation returns to the coherent output manifold, yielding grammatically perfect text. However, the factual direction is translated to an adjacent semantic basin (e.g., generating "London" instead of "Paris"), while preserving the prompt's syntactic format constraints. Across the $K=5$ control planes and 8 prompts, this translation is centered at a recovery lobe of $\bar{\theta}_{\text{recovery}} = 200.0^\circ \pm 12.4^\circ$.

Rather than claiming these precise angles ($\theta \approx 120^\circ$ and $\theta \approx 200^\circ$) as universal constants, we note that the exact transition boundaries are sensitive to the choice of the semantic context vector $\vec{v}_2$, the steering magnitude $R$, the model layer, and the specific prompt. Crucially, the invariant property observed is the *existence* of the recovery and collapse zones in directions orthogonal or opposite to $\vec{v}_1$, rather than their exact coordinate representation.

This empirical characterization, alongside the observation of a secondary symmetric recovery lobe under diametrically opposite steering ($-\vec{v}_1$), suggests that factual representations in the Gemma 3 residual stream exhibit localized, attractor-like stability profiles under orthogonal perturbations. We emphasize that this serves as a phenomenological description of representational robustness in our specific experimental setup rather than a formal mathematical proof of dynamical attractors (which would require computing Lyapunov exponents, mapping high-dimensional separatrix boundaries, or verifying dynamical convergence). Furthermore, because this study is restricted to Layer 12 of Gemma 3 4B-it on a subset of factual recall prompts, these transition dynamics should not be generalized to all layers, models, or tasks without further broad-scale validation.
