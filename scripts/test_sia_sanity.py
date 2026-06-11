"""
scripts/test_sia_sanity.py
Run: python scripts/test_sia_sanity.py
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sia_attack import cosine_sim_np, structure_invariant_transform


def test_cosine():
    a = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_sim_np(a, a) - 1.0) < 1e-6
    assert abs(cosine_sim_np(a, -a) + 1.0) < 1e-6
    print("PASS cosine_sim_np")


def test_sit_shape():
    img = np.random.rand(112, 112, 3).astype(np.float32)
    out = structure_invariant_transform(tf.constant(img[np.newaxis], dtype=tf.float32))
    assert out.shape == (1, 112, 112, 3)
    print("PASS SIT shape preserved")


def test_sit_changes_image():
    img = np.random.rand(112, 112, 3).astype(np.float32)
    out = structure_invariant_transform(tf.constant(img[np.newaxis], dtype=tf.float32))
    assert not np.allclose(out.numpy()[0], img)
    print("PASS image changed")


if __name__ == "__main__":
    print("Running SIA sanity tests\n")
    test_cosine()
    test_sit_shape()
    test_sit_changes_image()
    print("\nAll tests passed.")