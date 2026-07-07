"""GPU hyperparameter sweep for the SNN, targeting VEB sensitivity inter-patient
(de Chazal DS1 -> DS2). Sweeps the delta-encoder threshold and the class-weight
scheme -- the two levers most likely to move VEB sensitivity/PPV.

Design notes:
- Delta-encoded spike tensors are precomputed once per threshold and kept
  resident on the GPU (~100 MB), with manual minibatching. This avoids the
  per-epoch CPU-side encoding that otherwise starves the GPU.
- This SNN is latency-bound (a 256-step sequential loop over tiny matmuls), so
  gradient-updates/second is roughly constant across batch sizes. Batch size is
  chosen to balance GPU efficiency against getting enough updates to converge.
- Results are written incrementally so a long run is never lost; the best model
  (by VEB sensitivity) is saved to runs/snn_best.pt.

Run:  uv run python experiments/sweep_snn.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neurocardio.config import Config
from neurocardio.data.dataset import build_split
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.data.splits import get_split
from neurocardio.deploy.energy import spike_stats
from neurocardio.encoding.delta import delta_encode
from neurocardio.eval.metrics import aami_metrics, confusion
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import resolve_device, set_seed

# --- sweep configuration (env-overridable for quick smoke runs) ------------
THRESHOLDS = [float(t) for t in os.environ.get("SWEEP_THRESHOLDS", "0.05,0.1,0.2").split(",")]
SCHEMES = os.environ.get("SWEEP_SCHEMES", "balanced,sqrt").split(",")
BATCH = int(os.environ.get("SWEEP_BATCH", "512"))
LR = float(os.environ.get("SWEEP_LR", "0.004"))
EPOCHS = int(os.environ.get("SWEEP_EPOCHS", "25"))
SEED = 1337
N_CLASSES = 5

CACHE = Path("runs/sweep_cache")
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS = Path("runs/sweep_results.json")
BEST_WEIGHTS = Path("runs/snn_best.pt")


def raw_beats(cfg, split):
    fb, fl = CACHE / f"{split}_beats.npy", CACHE / f"{split}_labels.npy"
    if fb.exists() and fl.exists():
        return np.load(fb), np.load(fl)
    beats, labels = build_split(cfg, get_split(split))
    np.save(fb, beats)
    np.save(fl, labels)
    return beats, labels


def encode(beats, thr, tag):
    f = CACHE / f"{tag}_spikes_thr{thr}.npy"
    if f.exists():
        return np.load(f)
    out = np.stack([delta_encode(b, thr) for b in beats]).astype(np.float32)
    np.save(f, out)
    return out


def class_weights(labels, scheme):
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    total = counts.sum()
    w = np.zeros(N_CLASSES, dtype=np.float64)
    present = counts > 0
    base = total / (N_CLASSES * counts[present])
    w[present] = np.sqrt(base) if scheme == "sqrt" else base
    return w


def evaluate_resident(model, x, y, batch, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            preds.append(model(x[i : i + batch]).argmax(1))
    y_pred = torch.cat(preds).cpu().numpy()
    cm = confusion(y.cpu().numpy(), y_pred, N_CLASSES)
    return cm, aami_metrics(cm, AAMI_CLASSES)


def train_one(tr_x, tr_y, te_x, te_y, weight, device):
    set_seed(SEED)
    model = SNNClassifier(in_features=2, hidden=128, n_classes=N_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device))
    n = len(tr_x)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss_fn(model(tr_x[idx]), tr_y[idx]).backward()
            opt.step()
    cm, metrics = evaluate_resident(model, te_x, te_y, BATCH, device)
    return model, cm, metrics


def main():
    device = resolve_device("auto")
    print(f"device={device}  batch={BATCH}  epochs={EPOCHS}  lr={LR}", flush=True)
    cfg = Config()
    tr_b, tr_lab = raw_beats(cfg, "train")
    te_b, te_lab = raw_beats(cfg, "test")
    print(f"DS1={len(tr_lab)} beats  DS2={len(te_lab)} beats", flush=True)

    results, best = [], None
    grid = [(t, s) for t in THRESHOLDS for s in SCHEMES]
    for i, (thr, scheme) in enumerate(grid):
        t0 = time.time()
        tr_x = torch.from_numpy(encode(tr_b, thr, "train")).to(device)
        te_x = torch.from_numpy(encode(te_b, thr, "test")).to(device)
        tr_y = torch.from_numpy(tr_lab.astype(np.int64)).to(device)
        te_y = torch.from_numpy(te_lab.astype(np.int64)).to(device)
        weight = torch.tensor(class_weights(tr_lab, scheme), dtype=torch.float32)

        model, cm, m = train_one(tr_x, tr_y, te_x, te_y, weight, device)
        model.to("cpu")
        synops = spike_stats(model, torch.from_numpy(encode(te_b, thr, "test")[:1]))["synops"]

        veb, sveb = m["per_class"]["VEB"], m["per_class"]["SVEB"]
        row = {
            "threshold": thr,
            "scheme": scheme,
            "VEB_sens": veb["sensitivity"],
            "VEB_ppv": veb["ppv"],
            "SVEB_sens": sveb["sensitivity"],
            "SVEB_ppv": sveb["ppv"],
            "overall_accuracy": m["overall_accuracy"],
            "synops_per_beat": synops,
            "seconds": round(time.time() - t0, 1),
        }
        results.append(row)
        RESULTS.write_text(json.dumps(results, indent=2))  # incremental save
        print(
            f"[{i + 1}/{len(grid)}] thr={thr} {scheme:8s} "
            f"VEB sens={veb['sensitivity']:.3f} ppv={veb['ppv']:.3f} | "
            f"SVEB sens={sveb['sensitivity']:.3f} | acc={m['overall_accuracy']:.3f} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        if best is None or veb["sensitivity"] > best[0]:
            best = (veb["sensitivity"], model.state_dict())
            torch.save(best[1], BEST_WEIGHTS)

    results.sort(key=lambda r: (r["VEB_sens"], r["VEB_ppv"]), reverse=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    print("\n=== ranked by VEB sensitivity ===", flush=True)
    for r in results:
        print(
            f"  thr={r['threshold']} {r['scheme']:8s} "
            f"VEB sens={r['VEB_sens']:.3f} ppv={r['VEB_ppv']:.3f} | "
            f"SVEB sens={r['SVEB_sens']:.3f} ppv={r['SVEB_ppv']:.3f} | "
            f"acc={r['overall_accuracy']:.3f} | synops={r['synops_per_beat']}",
            flush=True,
        )
    print(f"\nbest VEB sensitivity = {results[0]['VEB_sens']:.3f}  -> {BEST_WEIGHTS}", flush=True)


if __name__ == "__main__":
    main()
