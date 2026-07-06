import numpy as np
import torch

from neurocardio.config import Config
from neurocardio.stream.detector import Detection, StreamDetector


class _Const2Model(torch.nn.Module):
    def forward(self, x):
        b = x.shape[0]
        out = torch.zeros(b, 5)
        out[:, 2] = 1.0  # always "VEB"
        return out


def test_stream_detector_emits_one_detection_per_beat():
    cfg = Config()
    fs = cfg.data.fs
    n = fs * 5
    sig = np.zeros(n)
    true_peaks = [400, 800, 1200, 1600]
    for p in true_peaks:
        sig[p - 2 : p + 3] += np.array([0.2, 0.6, 1.0, 0.6, 0.2])
    det = StreamDetector(_Const2Model(), cfg)
    detections = det.process(sig)
    assert len(detections) == len(true_peaks)
    assert all(isinstance(d, Detection) for d in detections)
    assert all(d.label == "VEB" for d in detections)
    idxs = [d.sample_index for d in detections]
    assert idxs == sorted(idxs)
