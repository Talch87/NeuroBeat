import numpy as np

from neurocardio.encoding.delta import delta_decode, delta_encode


def test_flat_signal_produces_no_spikes():
    sig = np.full(10, 0.3)
    spikes = delta_encode(sig, threshold=0.1)
    assert spikes.shape == (10, 2)
    assert spikes.sum() == 0.0


def test_step_up_produces_up_spikes():
    sig = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    spikes = delta_encode(sig, threshold=0.5)
    assert list(spikes[:, 0]) == [0.0, 0.0, 1.0, 1.0, 0.0]  # up channel
    assert spikes[:, 1].sum() == 0.0  # no down spikes


def test_step_down_produces_down_spikes():
    sig = np.array([1.0, 1.0, 0.0, 0.0])
    spikes = delta_encode(sig, threshold=0.5)
    assert spikes[:, 1].sum() == 2.0  # two down crossings
    assert spikes[:, 0].sum() == 0.0


def test_reconstruction_error_bounded_by_threshold():
    rng = np.random.default_rng(0)
    # scale kept well below threshold so per-step changes stay < threshold
    # (avoids delta-modulator slope-overload, where a single-step move of
    # >= 2*threshold cannot be closed by the one-spike-per-step comparator).
    sig = np.cumsum(rng.standard_normal(500)) * 0.02
    thr = 0.1
    spikes = delta_encode(sig, threshold=thr)
    recon = delta_decode(spikes, threshold=thr, initial=sig[0])
    assert np.max(np.abs(recon - sig)) <= thr + 1e-9
