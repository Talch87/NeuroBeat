import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1-D CNN baseline on raw beats [B, L], optionally with n_rr RR features
    concatenated before the classifier head (for a fair match to the SNN inputs)."""

    def __init__(self, n_classes: int = 5, n_rr: int = 0):
        super().__init__()
        self.n_rr = n_rr
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32 + n_rr, n_classes)

    def forward(self, x: torch.Tensor, rr: torch.Tensor = None) -> torch.Tensor:
        h = self.net(x.unsqueeze(1)).squeeze(-1)  # [B, 32]
        if self.n_rr and rr is not None:
            h = torch.cat([h, rr], dim=1)
        return self.head(h)


class LSTMClassifier(nn.Module):
    """LSTM baseline on raw beats [B, L] (length-L, 1-feature seq), optionally with
    n_rr RR features concatenated before the head (fair match to the SNN inputs)."""

    def __init__(self, n_classes: int = 5, hidden: int = 64, n_rr: int = 0):
        super().__init__()
        self.n_rr = n_rr
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden + n_rr, n_classes)

    def forward(self, x: torch.Tensor, rr: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))  # [B, L, 1]
        h = out[:, -1, :]
        if self.n_rr and rr is not None:
            h = torch.cat([h, rr], dim=1)
        return self.head(h)


class _TCNBlock(nn.Module):
    """Two causal dilated convolutions with a residual connection."""

    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        L = x.size(-1)
        h = torch.relu(self.c1(x)[..., :L])  # causal trim
        h = torch.relu(self.c2(h)[..., :L])
        res = x if self.down is None else self.down(x)
        return h + res


class TCN1D(nn.Module):
    """Compact temporal convolutional network on raw beats [B, L], optionally with
    n_rr RR features concatenated before the head. Small and embedded-realistic:
    three dilated residual blocks (dilations 1, 2, 4) at a narrow channel width."""

    def __init__(self, n_classes: int = 5, n_rr: int = 0, ch: int = 16, k: int = 3,
                 dilations=(1, 2, 4)):
        super().__init__()
        self.n_rr = n_rr
        blocks, cin = [], 1
        for d in dilations:
            blocks.append(_TCNBlock(cin, ch, k, d))
            cin = ch
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ch + n_rr, n_classes)

    def forward(self, x: torch.Tensor, rr: torch.Tensor = None) -> torch.Tensor:
        h = x.unsqueeze(1)
        for b in self.blocks:
            h = b(h)
        h = self.pool(h).squeeze(-1)
        if self.n_rr and rr is not None:
            h = torch.cat([h, rr], dim=1)
        return self.head(h)


class _ResBlock1D(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride),
                                      nn.BatchNorm1d(cout))

    def forward(self, x):
        r = x if self.down is None else self.down(x)
        h = torch.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return torch.relu(h + r)


class ResNetLite1D(nn.Module):
    """Compact 1-D residual CNN on raw beats [B, L], optionally with n_rr RR features
    concatenated before the head. A strided stem plus two residual blocks."""

    def __init__(self, n_classes: int = 5, n_rr: int = 0, ch=(16, 32)):
        super().__init__()
        self.n_rr = n_rr
        self.stem = nn.Sequential(nn.Conv1d(1, ch[0], 7, stride=2, padding=3),
                                  nn.BatchNorm1d(ch[0]), nn.ReLU())
        self.b1 = _ResBlock1D(ch[0], ch[0])
        self.b2 = _ResBlock1D(ch[0], ch[1], stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ch[1] + n_rr, n_classes)

    def forward(self, x: torch.Tensor, rr: torch.Tensor = None) -> torch.Tensor:
        h = self.stem(x.unsqueeze(1))
        h = self.b1(h)
        h = self.b2(h)
        h = self.pool(h).squeeze(-1)
        if self.n_rr and rr is not None:
            h = torch.cat([h, rr], dim=1)
        return self.head(h)
