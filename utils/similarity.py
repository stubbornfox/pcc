from __future__ import annotations

import numpy as np


def euclidean_distance(a, b):
  dist = np.linalg.norm(a - b)
  return dist


def cosine_similarity(a, b):
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)

  if a_norm * b_norm == 0:
    return 0.

  return np.sum(a * b) / (a_norm * b_norm)
