import torch

from neurocardio.models.snn import SNNClassifier


def test_forward_output_shape():
    model = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.randint(0, 2, (4, 256, 2)).float()  # [B, T, C]
    out = model(x)
    assert out.shape == (4, 5)


def test_forward_is_deterministic_with_fixed_seed():
    torch.manual_seed(0)
    m1 = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    torch.manual_seed(0)
    m2 = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.ones(2, 32, 2)
    assert torch.allclose(m1(x), m2(x))


def test_gradients_flow():
    torch.manual_seed(0)
    model = SNNClassifier(in_features=2, hidden=16, n_classes=5)
    x = torch.ones(3, 64, 2)
    out = model(x)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)
