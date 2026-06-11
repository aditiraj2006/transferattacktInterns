"""
scripts/test_vmi_fgsm_sanity.py
================================
Sanity-check VMI-FGSM without any real face models.
Uses a tiny synthetic "model" so this can be run on any machine immediately.

Run:  python scripts/test_vmi_fgsm_sanity.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.vmi_fgsm import vmi_fgsm_attack, cosine_similarity


def fake_gradient_fn(image: np.ndarray, target_emb: np.ndarray) -> np.ndarray:
    """
    Toy gradient: pretend the 'model' embeds by averaging pixel values.
    Loss = 1 - cosine_sim(mean(image), target_emb)
    Gradient = -target_emb (broadcast) / ||target_emb||
    This is just for structural testing, not for realism.
    """
    emb = image.mean(axis=(0, 1))                   # shape (3,)
    target = target_emb.flatten() / (np.linalg.norm(target_emb) + 1e-8)
    # d(1 - cosine_sim) / d(image) ≈ constant broadcast
    grad = np.ones_like(image) * (-target.mean())
    return grad.astype(np.float32)


def test_output_shape():
    img = np.random.rand(112, 112, 3).astype(np.float32)
    tgt = np.random.rand(512).astype(np.float32)
    adv = vmi_fgsm_attack(img, tgt, fake_gradient_fn, steps=3, N=2)
    assert adv.shape == img.shape, f"Shape mismatch: {adv.shape} vs {img.shape}"
    print("PASS  output_shape")


def test_epsilon_bound():
    img = np.random.rand(64, 64, 3).astype(np.float32)
    tgt = np.random.rand(512).astype(np.float32)
    eps = 0.03
    adv = vmi_fgsm_attack(img, tgt, fake_gradient_fn, epsilon=eps, steps=5, N=3)
    diff = np.abs(adv - img)
    assert diff.max() <= eps + 1e-5, f"Epsilon violated: max diff = {diff.max():.6f} > {eps}"
    print(f"PASS  epsilon_bound  (max_diff={diff.max():.5f})")


def test_pixel_range():
    img = np.random.rand(64, 64, 3).astype(np.float32)
    tgt = np.random.rand(512).astype(np.float32)
    adv = vmi_fgsm_attack(img, tgt, fake_gradient_fn, steps=5, N=3,
                           clip_min=0.0, clip_max=1.0)
    assert adv.min() >= -1e-6, f"Pixel below 0: {adv.min()}"
    assert adv.max() <= 1.0 + 1e-6, f"Pixel above 1: {adv.max()}"
    print("PASS  pixel_range")


def test_image_is_changed():
    """Attack should actually modify the image."""
    np.random.seed(42)
    img = np.random.rand(64, 64, 3).astype(np.float32)
    tgt = np.random.rand(512).astype(np.float32)
    adv = vmi_fgsm_attack(img, tgt, fake_gradient_fn, steps=5, N=2)
    assert not np.allclose(adv, img), "Adversarial image identical to original — attack did nothing"
    print(f"PASS  image_is_changed  (mean_delta={np.abs(adv - img).mean():.5f})")


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    c = np.array([-1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, c) + 1.0) < 1e-6
    print("PASS  cosine_similarity")


def test_n_samples_effect():
    """N=1 and N=10 should both run and produce valid output."""
    img = np.random.rand(32, 32, 3).astype(np.float32)
    tgt = np.random.rand(512).astype(np.float32)
    for N in [1, 10]:
        adv = vmi_fgsm_attack(img, tgt, fake_gradient_fn, steps=3, N=N)
        assert adv.shape == img.shape
    print("PASS  n_samples_effect")


if __name__ == "__main__":
    print("Running VMI-FGSM sanity tests (no real models required)\n")
    test_cosine_similarity()
    test_output_shape()
    test_epsilon_bound()
    test_pixel_range()
    test_image_is_changed()
    test_n_samples_effect()
    print("\nAll tests passed.")
