import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate


class SNNClassifier(nn.Module):
    """Two-layer feedforward LIF network trained with surrogate gradients.
    Input x: [B, T, C] spike trains. Output: spike-count logits [B, n_classes]."""

    def __init__(
        self,
        in_features: int = 2,
        hidden: int = 128,
        n_classes: int = 5,
        beta: float = 0.9,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(in_features, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        out_sum = torch.zeros(b, self.fc2.out_features, device=x.device)
        for step in range(t):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            # Readout accumulates the output-layer MEMBRANE POTENTIAL, not its
            # spikes. A spike-count readout has a dead-neuron trap: the loss is
            # minimised by the network going silent (all-zero logits -> uniform
            # softmax -> ln(n_classes)), and the surrogate gradient is too weak
            # to escape it. Integrating the continuous membrane gives an
            # always-nonzero gradient so the classifier actually trains. The
            # hidden layer still spikes (that is where the sparse-compute /
            # energy benefit lives; see deploy.energy.spike_stats).
            out_sum = out_sum + mem2
        return out_sum
