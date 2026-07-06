import numpy as np


def delta_encode(signal, threshold: float) -> np.ndarray:
    """Level-crossing (delta) encoder. Emits at most one up or down spike per
    timestep when the signal has moved >= threshold from the running reference.
    Returns float32 array [L, 2]: column 0 = up spikes, column 1 = down spikes.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    up = np.zeros(n, dtype=np.float32)
    down = np.zeros(n, dtype=np.float32)
    ref = x[0]
    for t in range(1, n):
        if x[t] - ref >= threshold:
            up[t] = 1.0
            ref += threshold
        elif ref - x[t] >= threshold:
            down[t] = 1.0
            ref -= threshold
    return np.stack([up, down], axis=1)


def delta_decode(spikes, threshold: float, initial: float = 0.0) -> np.ndarray:
    """Inverse of delta_encode: cumulative reconstruction from up/down spikes."""
    steps = (spikes[:, 0] - spikes[:, 1]) * threshold
    return initial + np.cumsum(steps)
