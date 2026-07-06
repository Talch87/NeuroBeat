from dataclasses import dataclass

import numpy as np
import torch

from neurocardio.config import Config
from neurocardio.data.preprocess import bandpass_filter, normalize
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.encoding.delta import delta_encode
from neurocardio.stream.qrs import find_r_peaks


@dataclass
class Detection:
    sample_index: int
    label: str
    score: float


class StreamDetector:
    """Runs a trained model over a continuous single-lead signal, emitting one
    Detection per detected beat. Mirrors the on-patch anomaly-logging path."""

    def __init__(self, model, config: Config):
        self.model = model
        self.cfg = config

    def _window(self, signal, peak) -> "np.ndarray | None":
        before = self.cfg.data.window_before
        after = self.cfg.data.window_after
        start, end = peak - before, peak + after
        if start < 0 or end > len(signal):
            return None
        return signal[start:end]

    def process(self, signal) -> list:
        sig = normalize(
            bandpass_filter(
                signal,
                fs=self.cfg.data.fs,
                low=self.cfg.data.bandpass_low,
                high=self.cfg.data.bandpass_high,
                order=self.cfg.data.filter_order,
            )
        )
        peaks = find_r_peaks(sig, fs=self.cfg.data.fs)
        self.model.eval()
        detections = []
        with torch.no_grad():
            for p in sorted(peaks):
                beat = self._window(sig, int(p))
                if beat is None:
                    continue
                spikes = delta_encode(beat, self.cfg.encoder.delta_threshold)
                x = torch.from_numpy(spikes).unsqueeze(0)  # [1, T, 2]
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                idx = int(probs.argmax(dim=1))
                detections.append(
                    Detection(
                        sample_index=int(p),
                        label=AAMI_CLASSES[idx],
                        score=float(probs[0, idx]),
                    )
                )
        return detections
