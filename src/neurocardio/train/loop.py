import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> str:
    """Resolve 'auto'/'cuda'/'cpu' to a concrete device string, degrading to
    'cpu' when CUDA is unavailable so configs stay portable across machines."""
    if device in ("auto", "cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def class_weights_from_labels(labels, n_classes: int) -> list[float]:
    """Inverse-frequency class weights for CrossEntropyLoss. Classes absent from
    `labels` get weight 0. Normalized so weights average ~1 over present classes."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    total = counts.sum()
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = total / (n_classes * counts[present])
    return weights.tolist()


def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(total, 1)


def train(
    model,
    train_loader,
    val_loader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    class_weights=None,
) -> dict:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight = (
        None
        if class_weights is None
        else torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    history = {"train_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(y)
        history["train_loss"].append(epoch_loss / len(train_loader.dataset))
        history["val_acc"].append(_accuracy(model, val_loader, device))
    return history
