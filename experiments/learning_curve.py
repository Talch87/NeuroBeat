"""Learning curve: does the SNN VEB detector improve with more training data?

Retrains the exact NeuroBeat-VEB config (T64, H128, threshold 0.12, 100 epochs) on
stratified fractions of DS1-train, fits the sens-first operating point on DS1-val
(frozen), and reports DS2 VEB sensitivity / PPV / F1 at each training size. The data
subset for a given fraction is fixed (seed 42) so only the training-set SIZE varies;
we average over 3 model seeds to separate data effect from optimization noise.

If the curve has flattened by 100% of DS1, more same-distribution data will not help
the VEB result and we can say so with evidence; if it is still climbing, it will.

Usage:
  python experiments/learning_curve.py --model-seeds 3
"""
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from freeze_veb_v1 import V1_CONFIG, op_sens_first, metr  # noqa: E402
from lock_snn_rr import class_weights, load_data, logits_of, train_model  # noqa: E402
from neurocardio.config import Config  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

N_CLASSES = 5
VEB = 2
FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def stratified_subset(labels_np, frac, seed=42):
    """Keep `frac` of each class's indices (class balance preserved), deterministic."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(N_CLASSES):
        idx = np.where(labels_np == c)[0]
        if len(idx) == 0:
            continue
        k = max(1, int(round(frac * len(idx)))) if frac < 1.0 else len(idx)
        keep.append(rng.choice(idx, k, replace=False) if k < len(idx) else idx)
    return np.sort(np.concatenate(keep))


def f1(s, p):
    return 2 * s * p / max(s + p, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/learning_curve.json")
    args = ap.parse_args()
    device = resolve_device("auto")
    cfg = Config()

    data = load_data(cfg, V1_CONFIG, device, external_specs=None, augment_specs=None)
    trx, trr, trl, vax, var, val_l, tex, ter, tel, _ = data
    trl_np = trl.cpu().numpy()
    in_features = 2 * len(V1_CONFIG["orders"])
    batch = V1_CONFIG["batch"]
    print(f"device={device} full_train={len(trl_np)} "
          f"(VEB {int((trl_np==VEB).sum())})", flush=True)

    rows = []
    for frac in FRACTIONS:
        idx = stratified_subset(trl_np, frac)
        idx_t = torch.from_numpy(idx).to(device)
        sx, sr, sl = trx[idx_t], trr[idx_t], trl[idx_t]
        sl_np = sl.cpu().numpy()
        weight = class_weights(sl_np, V1_CONFIG["scheme"])
        n_veb = int((sl_np == VEB).sum())
        seed_metrics = []
        for ms in range(args.model_seeds):
            t0 = time.time()
            model = train_model(sx, sr, sl, V1_CONFIG["hidden"], in_features, weight,
                                V1_CONFIG["lr"], V1_CONFIG["epochs"], batch, device, ms)
            (bias, _vs, _vp), feas = op_sens_first(logits_of(model, vax, var, batch), val_l)
            d = metr(logits_of(model, tex, ter, batch), tel, bias)
            seed_metrics.append((d["VEB_sens"], d["VEB_ppv"]))
            print(f"  frac {frac:.2f} (n={len(idx)}, VEB {n_veb}) seed {ms}: "
                  f"DS2 {d['VEB_sens']}/{d['VEB_ppv']} feas={feas} ({time.time()-t0:.0f}s)",
                  flush=True)
        sens = np.array([m[0] for m in seed_metrics]); ppv = np.array([m[1] for m in seed_metrics])
        f1s = np.array([f1(s, p) for s, p in seed_metrics])
        rows.append({"frac": frac, "n_train": len(idx), "n_veb_train": n_veb,
                     "DS2_VEB_sens_mean": round(float(sens.mean()), 4),
                     "DS2_VEB_ppv_mean": round(float(ppv.mean()), 4),
                     "DS2_VEB_f1_mean": round(float(f1s.mean()), 4),
                     "DS2_VEB_f1_min": round(float(f1s.min()), 4),
                     "DS2_VEB_f1_max": round(float(f1s.max()), 4)})
        print(f"frac {frac:.2f}: DS2 VEB F1 {f1s.mean():.4f} "
              f"[{f1s.min():.4f}, {f1s.max():.4f}] "
              f"sens {sens.mean():.4f} ppv {ppv.mean():.4f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2))

    # ---- figure ----
    nt = [r["n_train"] for r in rows]
    f1m = [r["DS2_VEB_f1_mean"] for r in rows]
    lo = [r["DS2_VEB_f1_mean"] - r["DS2_VEB_f1_min"] for r in rows]
    hi = [r["DS2_VEB_f1_max"] - r["DS2_VEB_f1_mean"] for r in rows]
    sn = [r["DS2_VEB_sens_mean"] for r in rows]
    pv = [r["DS2_VEB_ppv_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(nt, f1m, yerr=[lo, hi], marker="o", color="#1f77b4", capsize=4,
                label="DS2 VEB F1 (mean, min-max)", zorder=3)
    ax.plot(nt, sn, marker="^", ls="--", color="#2ca02c", alpha=0.7, label="DS2 VEB sensitivity")
    ax.plot(nt, pv, marker="s", ls="--", color="#ff7f0e", alpha=0.7, label="DS2 VEB PPV")
    ax.set_xlabel("DS1 training beats (single-stage SNN, T=64)")
    ax.set_ylabel("DS2 value")
    ax.set_title("Learning curve: DS2 VEB detection vs training-set size")
    ax.grid(ls=":", alpha=0.4); ax.legend(loc="lower right"); ax.set_ylim(0, 1.02)
    fig.savefig("paper/figures/fig7_learningcurve.png", dpi=200, bbox_inches="tight")
    fig.savefig("paper/figures/fig7_learningcurve.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {args.out} and paper/figures/fig7_learningcurve.png", flush=True)


if __name__ == "__main__":
    main()
