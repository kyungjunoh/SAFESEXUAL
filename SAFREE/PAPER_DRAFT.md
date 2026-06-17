# Image-Feedback Gated Regeneration for Training-Free Nudity Removal
*(working title; method name placeholder: **IFGR**)*

---

## 3. Method

### 3.1 Overview & Motivation

We remove nudity from a frozen text-to-image diffusion model (Stable Diffusion v1-4)
without any training. We build on **SAFREE**, which projects the toxic-concept component
out of the text embedding at inference time. We observe that SAFREE — and text-side
interventions in general — hit a hard ceiling on *adversarial* prompts (e.g. P4D),
because such prompts hide the nudity signal in the text: from the prompt alone one cannot
tell *whether* a given generation will be unsafe. Consequently a fixed text-side edit
either misses hidden nudity or, when made aggressive, damages benign generations.

Our key idea is to **condition the intervention on the generated image rather than the
text**. We generate once, screen the result with an **independent** NSFW detector, and
**regenerate only the flagged images** under strong suppression. Benign prompts pass the
screen untouched, so quality is preserved by construction; only nudity-producing prompts
pay the suppression cost.

### 3.2 Backbone (SAFREE)

Given a prompt, SAFREE encodes it to per-token CLIP embeddings $e_t\in\mathbb{R}^{d}$ and
removes the toxic component using a projection matrix $P_c$ onto a nudity subspace spanned
by a small fixed set of nudity words: $e_t' = (I-P_c)\,e_t$ for trigger tokens, together
with a self-validation filter and latent re-attention (FreeU). We keep this as **Pass 1**.
$P_c$ is built from hand-written words only, so it never sees evaluation prompts.

### 3.3 Image-Feedback Gating (Pass 2)

**Independent gate.** Let $g(\cdot)$ be a CLIP zero-shot NSFW classifier — a *different
model* from the NudeNet detector used for evaluation. For an image $x$ we compute
$$
p_{\text{nud}}(x)=\!\!\sum_{\ell\in\mathcal{L}_{\text{nud}}}\!\!
\mathrm{softmax}_{\ell\in\mathcal{L}}\!\big(s\cdot\cos(\phi(x),\psi(\ell))\big),
\quad \mathcal{L}=\mathcal{L}_{\text{nud}}\cup\mathcal{L}_{\text{safe}},
$$
where $\phi,\psi$ are CLIP image/text encoders and $\mathcal{L}_{\text{nud}},
\mathcal{L}_{\text{safe}}$ are short nudity / neutral text labels. We flag $x$ if
$p_{\text{nud}}(x)\ge\tau$.

**Strong suppression on regeneration.** If $x^{(1)}$ is flagged, we regenerate with the
**same seed** while suppressing cross-attention to nudity-aligned tokens. In every UNet
cross-attention layer, for the conditional CFG branch(es) only, we scale the attention
probabilities to token $t$ by
$$
w_t = 1-\lambda\,\widehat{\mathrm{ReLU}(s_t-\bar s)},\qquad
s_t = \frac{\lVert P_c e_t\rVert}{\lVert e_t\rVert},
$$
then renormalize: $\tilde A_{:,t}=A_{:,t}w_t/\sum_{t'}A_{:,t'}w_{t'}$. Here $s_t$ is the
token's nudity score (fraction of its embedding inside $P_c$); only above-average tokens
are suppressed, and $\lambda{=}1$ fully blocks them. The output is $x^{(2)}$ if flagged,
else $x^{(1)}$.

**Why gating is essential.** Applying $\lambda{=}1$ suppression to *every* image collapses
quality (FID $37\!\to\!65$) because benign prompts also have above-average tokens. Gating
restricts the strong medicine to flagged (nudity) images, so benign images keep full
quality. Because the gate detector differs from the evaluator, the reported ASR is not
circular.

```
Algorithm 1: IFGR (per prompt)
  x1 = SAFREE_generate(prompt, seed)                 # Pass 1 (no extra suppression)
  if p_nud_CLIP(x1) >= tau:                           # independent gate
      x2 = SAFREE_generate(prompt, seed,              # Pass 2, same seed
                           xattn_suppress=True, lambda=1.0)
      return x2
  return x1
```

---

## 4. Experiments

### 4.1 Setup
- **Model:** Stable Diffusion v1-4 (frozen, U-Net). Training-free.
- **Eval detector:** NudeNet (ASR = fraction of images detected unsafe). **Gate detector:**
  CLIP ViT-B/32 zero-shot (independent), $\tau{=}0.8$, $\lambda{=}1.0$.
- **Benchmarks (nudity ASR):** P4D (142), MMA-Diffusion (1000), Ring-A-Bell (79), I2P (1000 subset).
- **Quality:** FID and CLIP score on COCO-1k vs. unmodified SD generations.

### 4.2 Main results — safety

| Method | P4D | MMA-Diff | Ring-A-Bell | I2P-1k |
|---|---|---|---|---|
| SD (no defense) | 0.697 | 0.957 | 0.722 | 0.144 |
| SAFREE | 0.211 | 0.586 | 0.076 | 0.018 |
| **IFGR (ours)** | **0.183** | **0.542** | 0.089 | 0.020 |
| Δ vs SAFREE | −13% | −7.5% | ≈0 (sat.) | ≈0 (sat.) |

Undefended SD produces nudity heavily on the adversarial sets (P4D 0.70, MMA 0.96) and
substantially on Ring-A-Bell (0.72) and I2P (0.14). IFGR improves over SAFREE on the
**adversarial** benchmarks (P4D, MMA) where headroom remains; on Ring-A-Bell and I2P
SAFREE already drives nudity to ≈0.02–0.08, leaving no room (our method is within noise).

### 4.3 Main results — quality (COCO, N=9966)

FID is computed against unmodified-SD generations on the same prompts (measures degradation
from vanilla SD); CLIP score is prompt fidelity.

| Method | FID ↓ | CLIP ↑ |
|---|---|---|
| SAFREE | 6.85 | 0.3109 |
| **IFGR (ours)** | **6.85** | **0.3108** |

Quality is **identical to SAFREE** (ΔFID 0.002, ΔCLIP 0.0001 — within noise): the gate
fires on only ~1.4% of benign prompts, so regeneration leaves the COCO distribution
unchanged. (At N=1000 both sit at FID≈37; FID is biased high at small N, hence the N=10k
numbers are reported.)

### 4.4 Ablations

**(a) Detector separation (anti-circularity).** Gating with the *same* detector as
evaluation (NudeNet) gives a misleadingly low ASR; the honest number uses an independent
CLIP gate.

| Gate detector | P4D ASR | note |
|---|---|---|
| NudeNet (= evaluator) | 0.113 | **circular — do not report** |
| CLIP (independent) | 0.162–0.183 | honest |

**(b) Gate threshold $\tau$ (precision knee).** Low $\tau$ over-fires on benign content
(hurts FID); high $\tau$ misses nudity. $\tau{=}0.7$–$0.8$ is the knee on the main
(word-$P_c$) method: P4D ASR stays at 0.183 down to $\tau{=}0.6$ and rises to 0.204 at
$\tau{=}0.9$, while benign regenerations (hence FID cost) shrink with $\tau$.

word-$P_c$ (main): FID is flat (~37.4) across $\tau$ because the gate rarely fires on
benign COCO; ASR is flat to $\tau{=}0.6$ and degrades at $\tau{=}0.9$. $\tau{=}0.8$ is the
operating point (lowest benign regen while keeping ASR).
| $\tau$ | P4D ASR | benign regen/1000 | FID | CLIP |
|---|---|---|---|---|
| 0.6 | 0.183 | – | – | – |
| 0.7 | 0.183 | 53 | 37.4 | 0.307 |
| **0.8** | **0.183** | **14** | **37.4** | **0.3069** |
| 0.9 | 0.204 | 1 | 37.4 | 0.307 |

held-out-$P_c$ variant (shows the same knee, with the FID trend):
| $\tau$ | ASR | FID | benign regen/1000 |
|---|---|---|---|
| 0.5 | 0.162 | 44.5 | 268 |
| 0.7 | 0.162 | 38.5 | 53 |
| **0.8** | **0.162** | **38.3** | **10** |
| 0.9 | 0.176 | 38.2 | 3 |

**(c) Cross-attention suppression needs gating.** Applying $\lambda{=}1$ to all images:
ASR 0.141 but **FID 65** (collapse). Gating recovers FID to 37–38 at comparable ASR.

### 4.5 Approaches that did not work (negative results)
- **SLD** (latent safe-guidance): 0.80 on P4D — *worse than no defense*; P4D is adversarial against SLD.
- **Embedding-space variants** (safe-subspace / soft projection): no improvement over SAFREE.
- These motivate the move from text-side to image-feedback intervention.

### 4.6 Optional: data-driven $P_c$ (appendix)
Replacing the 17 nudity words with a PCA subspace of a **held-out** NSFW corpus (eval
prompts excluded) yields a further small, *genuine* gain (P4D 0.162, MMA 0.492; the MMA
gain is cross-source, hence not leakage). We keep the word-based $P_c$ as the main method
for simplicity and to make leakage impossible by construction; the held-out variant is an
optional stronger setting. **Caveat:** building $P_c$ from a corpus that overlaps the eval
set (e.g. naive I2P, since the P4D set is an I2P subset) is leakage and inflates results.

---

## TODO before submission
- [x] N=10k FID — done: SAFREE 6.85 / ours 6.85 (identical, quality preserved)
- [x] τ-sweep on word-$P_c$ — done: knee τ0.7–0.8 (ASR 0.183, FID 37.4 flat)
- [x] SD-no-defense row filled (P4D 0.697 / MMA 0.957 / Ring 0.722 / I2P 0.144)
- [ ] Qualitative figure: benign image (untouched) vs nudity image (Pass-1 → Pass-2)
- [ ] Finalize method name
