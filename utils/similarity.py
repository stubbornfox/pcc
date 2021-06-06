from __future__ import annotations

import numpy as np


def euclidean_distance(a, b):
  dist = np.linalg.norm(a - b)
  return dist
