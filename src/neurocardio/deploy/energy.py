import torch

from neurocardio.models.snn import SNNClassifier


def synaptic_operations(spike_counts: dict, fan_out: dict) -> int:
    """Pure SynOps proxy: sum over layers of presynaptic_spikes * fan_out."""
    return int(sum(spike_counts[k] * fan_out[k] for k in spike_counts))


def spike_stats(model: SNNClassifier, x: torch.Tensor) -> dict:
    """Run one forward pass, counting input spikes to each Linear layer and the
    resulting SynOps. Assumes the SNNClassifier fc1/fc2/lif structure."""
    model.eval()
    b, t, _ = x.shape
    mem1 = model.lif1.reset_mem()
    mem2 = model.lif2.reset_mem()
    fc1_in_spikes = 0
    fc2_in_spikes = 0
    total_spikes = 0
    with torch.no_grad():
        for step in range(t):
            inp = x[:, step, :]
            fc1_in_spikes += int(inp.sum())
            spk1, mem1 = model.lif1(model.fc1(inp), mem1)
            fc2_in_spikes += int(spk1.sum())
            spk2, mem2 = model.lif2(model.fc2(spk1), mem2)
            total_spikes += int(spk1.sum()) + int(spk2.sum())
    counts = {"fc1": fc1_in_spikes, "fc2": fc2_in_spikes}
    fan_out = {"fc1": model.fc1.out_features, "fc2": model.fc2.out_features}
    return {
        "total_spikes": total_spikes,
        "synops": synaptic_operations(counts, fan_out),
        "spike_counts": counts,
    }
