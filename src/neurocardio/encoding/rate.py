import numpy as np


def rate_encode(signal, num_steps: int, seed: int = 0) -> np.ndarray:
    """Poisson-style rate encoder. Min-max scales the signal to [0, 1] firing
    probabilities and samples spikes over `num_steps`. Returns [num_steps, L]."""
    x = np.asarray(signal, dtype=np.float64)
    lo, hi = x.min(), x.max()
    p = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    rng = np.random.default_rng(seed)
    draws = rng.random((num_steps, len(x)))
    return (draws < p).astype(np.float32)
