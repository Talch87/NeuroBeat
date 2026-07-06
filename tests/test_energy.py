import torch

from neurocardio.deploy.energy import spike_stats, synaptic_operations
from neurocardio.models.snn import SNNClassifier


def test_synaptic_operations_pure_function():
    spike_counts = {"fc1": 5, "fc2": 3}
    fan_out = {"fc1": 10, "fc2": 4}
    assert synaptic_operations(spike_counts, fan_out) == 5 * 10 + 3 * 4


def test_spike_stats_runs_model_and_counts():
    torch.manual_seed(0)
    model = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.ones(1, 64, 2)
    stats = spike_stats(model, x)
    assert stats["total_spikes"] >= 0
    assert "synops" in stats
    assert stats["synops"] >= 0
