"""Confidence intervals for the gated-cascade DS2 VEB result.

Reports both a beat-level bootstrap (independence assumption) and a patient/record-
level bootstrap (resamples whole DS2 records, respecting within-record correlation).
The record-level interval is the honest one to quote; the beat-level one is shown for
contrast and is expected to be narrower.

Reconstructs the frozen cascade predictions from cached logits + the frozen biases,
so it does not retrain anything.

Usage:
  python experiments/stats_ci.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import freeze_veb_v1 as f  # noqa: E402
import gated_ensemble_veb as g  # noqa: E402
from lock_snn_rr import load_data, logits_of  # noqa: E402
from neurocardio.config import Config  # noqa: E402
from neurocardio.data.dataset import build_split_rr  # noqa: E402
from neurocardio.data.splits import DS2_RECORDS  # noqa: E402
from neurocardio.models.snn import SNNClassifier  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

VEB = 2


def sens_ppv(pred, true):
    tp = int((pred & true).sum()); fn = int((~pred & true).sum()); fp = int((pred & ~true).sum())
    return tp / max(tp + fn, 1), tp / max(tp + fp, 1)


def ci(samples, lo=2.5, hi=97.5):
    s = np.sort(samples)
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


def main():
    device = resolve_device("auto")
    cfg = Config()

    # ---- reconstruct frozen cascade predictions on DS2 ----
    gj = json.loads(Path("models/neurobeat-veb-v1/gated_ensemble.json").read_text())
    b1 = np.array(gj["screener"]["bias"]); b2 = np.array(gj["confirmer"]["bias"])
    d = load_data(cfg, g.SCREENER, device, external_specs=None, augment_specs=None)
    tex, ter = d[6], d[7]
    m1 = SNNClassifier(in_features=2 * len(g.SCREENER["orders"]), hidden=g.SCREENER["hidden"],
                       n_classes=5, n_rr=3).to(device)
    m1.load_state_dict(torch.load("models/neurobeat-veb-v1/screener_weights.pt", map_location=device))
    scr = logits_of(m1, tex, ter, 512)
    conf = np.mean([np.load(f.CACHE / f"seed{s}_ds2.npy") for s in g.ENSEMBLE_SEEDS], axis=0)
    lab = np.load(f.CACHE / "labels_ds2.npy")
    true = lab == VEB
    flag = (scr + b1).argmax(1) == VEB
    casc = flag & ((conf + b2).argmax(1) == VEB)

    # ---- per-beat DS2 record ids (build order matches the cached split) ----
    rid = []
    total = 0
    for rec in DS2_RECORDS:
        beats, _, _ = build_split_rr(cfg, [rec])
        rid.extend([rec] * len(beats)); total += len(beats)
    rid = np.array(rid)
    assert total == len(lab), f"record beats {total} != cached DS2 {len(lab)}"
    records = list(DS2_RECORDS)
    by_rec = {r: np.where(rid == r)[0] for r in records}

    s0, p0 = sens_ppv(casc, true)
    print(f"gated cascade DS2: VEB sens {s0:.4f}  PPV {p0:.4f}  "
          f"(VEB n={int(true.sum())}, beats={len(lab)}, records={len(records)})", flush=True)

    B = 2000
    # beat-level bootstrap
    rng = np.random.default_rng(0); N = len(lab); bs, bp = [], []
    for _ in range(B):
        idx = rng.integers(0, N, N); s, p = sens_ppv(casc[idx], true[idx]); bs.append(s); bp.append(p)
    # record-level bootstrap: resample records with replacement, pool their beats
    rng = np.random.default_rng(1); rs, rp = [], []
    R = len(records)
    for _ in range(B):
        chosen = rng.integers(0, R, R)
        idx = np.concatenate([by_rec[records[c]] for c in chosen])
        s, p = sens_ppv(casc[idx], true[idx]); rs.append(s); rp.append(p)

    bl_s, bl_p = ci(bs), ci(bp)
    rl_s, rl_p = ci(rs), ci(rp)
    print(f"\nbeat-level    95% CI  sens [{bl_s[0]:.3f}, {bl_s[1]:.3f}]  ppv [{bl_p[0]:.3f}, {bl_p[1]:.3f}]",
          flush=True)
    print(f"record-level  95% CI  sens [{rl_s[0]:.3f}, {rl_s[1]:.3f}]  ppv [{rl_p[0]:.3f}, {rl_p[1]:.3f}]",
          flush=True)
    out = {"sens": round(s0, 4), "ppv": round(p0, 4), "veb_n": int(true.sum()),
           "beat_ci": {"sens": [round(x, 4) for x in bl_s], "ppv": [round(x, 4) for x in bl_p]},
           "record_ci": {"sens": [round(x, 4) for x in rl_s], "ppv": [round(x, 4) for x in rl_p]},
           "n_records": len(records)}
    Path("runs/cascade_ci.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> runs/cascade_ci.json", flush=True)


if __name__ == "__main__":
    main()
