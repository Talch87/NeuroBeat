import numpy as np

from neurocardio.encoding.rate import rate_encode


def test_rate_encode_shape_and_range():
    sig = np.linspace(0.0, 1.0, 8)
    spikes = rate_encode(sig, num_steps=50, seed=0)
    assert spikes.shape == (50, 8)
    assert set(np.unique(spikes)).issubset({0.0, 1.0})


def test_higher_value_fires_more_often():
    sig = np.array([0.05, 0.95])
    spikes = rate_encode(sig, num_steps=2000, seed=1)
    rate_low = spikes[:, 0].mean()
    rate_high = spikes[:, 1].mean()
    assert rate_high > rate_low
