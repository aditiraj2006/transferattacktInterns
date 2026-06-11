# Technical Report: VMI-FGSM Transfer Attack
## Track A — Adversarial Transfer Attacks on Face Recognition
**Author:** Intern  
**Date:** June 2026  
**Repo branch:** `trackA_vmi_fgsm`

---

## 1. Overview

This report documents the implementation and evaluation of **VMI-FGSM** (Variance-tuned Momentum Iterative FGSM) as a new transfer attack on face recognition, compared against the five vanilla baselines already in the repository.

---

## 2. Why VMI-FGSM?

The five existing baselines are:

| Attack | Core idea |
|---|---|
| PGD | Basic iterative projected gradient |
| MI-FGSM | Adds momentum to stabilise gradient direction |
| TI-FGSM | Uses translated copies of the image for gradient |
| SI-NI-FGSM | Scale-invariance + Nesterov lookahead |
| MI-ADMIX-DI-TI | Combines momentum + admix + DI + TI |

VMI-FGSM (CVPR 2021, Wang & He) is **not present** in the repo.  It extends MI-FGSM by adding a **variance-tuning step**: at each iteration it samples N neighbouring points in an L-inf ball around the current adversarial example, estimates the gradient variance across those samples, and uses the variance as a correction term before the momentum update.

**Why this helps transfer:** Models that over-fit to the surrogate's loss landscape tend to live in high-variance gradient regions.  By estimating and correcting for that variance, VMI-FGSM produces perturbations that lie in flatter, more model-agnostic regions of the loss surface — which is exactly what makes adversarial examples transfer.

**Why this is a good choice for this assignment:**
- Clean, well-understood paper with reproducible pseudocode
- Directly improves on the strongest existing baseline (MI-FGSM) — so comparison is meaningful
- Works with face-verification cosine-distance loss without modification
- No extra model or dataset required
- CVPR 2021 (recent, peer-reviewed, highly cited)

---

## 3. Algorithm

### 3.1 MI-FGSM recap (baseline)

```
g_0 = 0
For t = 0 ... T-1:
    g_{t+1} = mu * g_t  +  grad_x L(x_t) / ||grad_x L(x_t)||_1
    x_{t+1} = Clip_{x,eps}( x_t + alpha * sign(g_{t+1}) )
```

### 3.2 VMI-FGSM (new)

```
g_0 = 0,  v_0 = 0
For t = 0 ... T-1:

    # Variance estimation step (NEW)
    For k = 1 ... N:
        x_k = x_t + Uniform(-beta*eps, beta*eps)   # neighbour sample
        v_k = grad_x L(x_k)
    v_t = Var({v_1, ..., v_N})                     # element-wise variance

    # Corrected gradient
    g_corrected = grad_x L(x_t) + v_t              # Eq. 6 in paper

    # Momentum update (same as MI-FGSM from here)
    g_{t+1} = mu * g_t  +  g_corrected / ||g_corrected||_1
    x_{t+1} = Clip_{x,eps}( x_t + alpha * sign(g_{t+1}) )
```

The only addition over MI-FGSM is the `N` extra gradient calls per step to compute `v_t`.

### 3.3 Loss function for face verification

For **impersonation** (make the adversarial face look like identity B):
```
L = 1 - cosine_sim( embed(x_adv), embed(x_target) )
```

For **dodging** (make the system fail to recognise identity A):
```
L = cosine_sim( embed(x_adv), embed(x_source) )
```
and the gradient sign is negated (we maximise distance).

---

## 4. Implementation Details

### 4.1 Files added

```
core/vmi_fgsm.py          — Core VMI-FGSM algorithm (pure numpy)
core/gradient_utils.py    — TF GradientTape wrappers for DeepFace models
experiments/run_new_attack.py  — End-to-end runner on the subset
scripts/compare_results.py     — Comparison tables and charts
scripts/test_vmi_fgsm_sanity.py — Unit tests (no models required)
docs/technical_report.md       — This file
```

### 4.2 Integration with existing code

`core/vmi_fgsm.py` is a **standalone module** — it does not modify `core/transfer_attack_core.py`.  The `gradient_fn` callback pattern means any DeepFace model can be plugged in with one line change.

### 4.3 Hyperparameters

| Parameter | Default | Paper value | Notes |
|---|---|---|---|
| epsilon | 0.05 | 16/255 ≈ 0.063 | Slightly conservative for face images |
| steps (T) | 10 | 10 | Standard |
| mu | 1.0 | 1.0 | Full momentum |
| N | 5 | 20 | Reduced for speed; paper uses 20 |
| beta | 1.5 | 1.5 | Neighbourhood radius as fraction of eps |
| alpha | eps/T | eps/T | Auto-computed |

**Note:** Using N=5 (vs paper's N=20) trades some accuracy for 4× speed.  For a final comparison with more compute, N=20 is recommended.

---

## 5. Evaluation Protocol

Identical to the baseline evaluation:

1. Use `docs/subset_input_pairs.csv` (same subset as baselines).
2. Attacker models: FaceNet512, ArcFace, GhostFaceNet, VGG-Face.
3. Victim models: FaceNet, FaceNet512, ArcFace, GhostFaceNet, VGG-Face, IR152.
4. Skip rules: FaceNet512→FaceNet512, FaceNet512→FaceNet; all others skip self.
5. Thresholds from `core/verification_thresholds.json`.
6. Metric: **breach rate** = fraction of pairs where the attack succeeded.

---

## 6. Expected Results

Based on the CVPR 2021 paper results and the nature of variance tuning, VMI-FGSM is expected to:

- Match or exceed MI-FGSM on **impersonation** (MI-FGSM is already the strongest vanilla baseline for impersonation)
- Show larger gains on **dodging** where the gradient landscape is typically noisier
- Transfer better to **IR-152** which is architecturally different from the attacker models
- Show most improvement when the attacker is **ArcFace** (margin-based loss leads to sharper gradient regions where variance correction helps most)

### 6.1 Where VMI-FGSM may not improve

- When N is very small (N=1 degrades to MI-FGSM)
- Against **VGG-Face** as victim (shallower network, less gradient noise)
- Very small epsilon budgets where the neighbourhood collapses

---

## 7. Comparison Summary Template

After running `experiments/run_new_attack.py` and `scripts/compare_results.py`, the output `results_new/comparison_overall.csv` will look like:

| Attack | Breach Rate (%) | Delta vs VMI-FGSM |
|---|---|---|
| VMI-FGSM | **TBD** | 0.00 |
| SI-NI-FGSM | baseline | TBD |
| MI-FGSM | baseline | TBD |
| MI-ADMIX-DI-TI | baseline | TBD |
| TI-FGSM | baseline | TBD |
| PGD | baseline | TBD |

*(Fill in from actual run output.)*

---

## 8. Reproducibility

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Sanity test (no models needed, ~2 seconds)
python scripts/test_vmi_fgsm_sanity.py

# 3. Run VMI-FGSM on subset
python experiments/run_new_attack.py \
    --pairs      docs/subset_input_pairs.csv \
    --faces_dir  /path/to/dataset_extractedfaces \
    --thresholds core/verification_thresholds.json \
    --output_dir results_new/

# 4. Compare with baseline
python scripts/compare_results.py \
    --baseline_eval results_baseline/subset_attack_eval_long.csv \
    --new_eval      results_new/new_attack_eval_long.csv \
    --output_dir    results_new/
```

Expected runtime: ~30–60 min on CPU (subset size × 4 attackers × N=5 samples × 10 steps).  
Use GPU or reduce N for faster iteration.

---

## 9. Limitations and Failure Modes

1. **Numerical gradient fallback is too slow** for large images — always use the TF backprop path (`gradient_utils.py`).
2. **N=5 is a compromise** — the paper uses N=20 for best results.
3. **IR-152 requires manual setup** (raw model files from Google Drive) — if unavailable, victim evaluation will skip it.
4. **DeepFace version sensitivity** — `gradient_utils.py` accesses internal DeepFace APIs that may change between versions.  Tested with deepface==0.0.93.
5. **Variance correction can amplify noise** at very small epsilon — prefer epsilon ≥ 0.03.

---

## 10. References

- Wang X, He K. *Enhancing the Transferability of Adversarial Attacks through Variance Tuning.* CVPR 2021. https://arxiv.org/abs/2103.15571
- Dong Y et al. *Boosting Adversarial Attacks with Momentum.* CVPR 2018. (MI-FGSM)
- Lin J et al. *Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks.* ICLR 2020. (SI-NI-FGSM)
- Trustworthy-AI-Group/TransferAttack framework: https://github.com/Trustworthy-AI-Group/TransferAttack
