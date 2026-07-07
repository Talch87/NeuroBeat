import numpy as np
import torch
from torch.utils.data import DataLoader

from neurocardio.data.dataset import ECGBeatDataset
from neurocardio.models.baselines import CNN1D
from neurocardio.train.loop import (
    class_weights_from_labels,
    resolve_device,
    set_seed,
    train,
)


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_resolve_device_degrades_to_cpu_when_no_cuda():
    # "cpu" is always literal; "auto"/"cuda" resolve to cuda only if available.
    assert resolve_device("cpu") == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected
    assert resolve_device("cuda") == expected


def test_class_weights_inverse_frequency():
    # 90 of class 0, 10 of class 1, 0 of class 2 -> rarer class gets more weight,
    # absent class gets 0.
    labels = np.array([0] * 90 + [1] * 10)
    w = class_weights_from_labels(labels, n_classes=3)
    assert len(w) == 3
    assert w[1] > w[0] > 0.0
    assert w[2] == 0.0
    # inverse-frequency: weight ratio equals inverse count ratio (90/10 = 9)
    assert abs(w[1] / w[0] - 9.0) < 1e-9


def test_overfits_tiny_batch():
    # Task 12 tests the TRAINING LOOP (forward -> loss -> backward -> step). We
    # use the CNN1D baseline on raw, linearly separable beats: a conventional
    # model that overfits 8 samples monotonically and deterministically, so the
    # loop's mechanics are what is under test, not SNN spike-learning dynamics.
    # The SNN + delta + train integration (and that it actually learns) is
    # exercised separately in the end-to-end smoke test.
    set_seed(0)
    beats = np.zeros((8, 64), dtype=np.float32)
    labels = np.zeros(8, dtype=np.int64)
    for i in range(8):
        cls = i % 2
        start = 10 + cls * 30
        beats[i, start : start + 10] = 1.0  # early bump vs late bump
        labels[i] = cls

    ds = ECGBeatDataset(beats, labels)  # raw beats, no spike encoding
    loader = DataLoader(ds, batch_size=8)
    model = CNN1D(n_classes=2)
    history = train(model, loader, loader, epochs=100, lr=0.01, device="cpu")
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_acc"][-1] >= 0.99
