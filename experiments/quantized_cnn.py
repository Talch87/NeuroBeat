"""Quantized (int8) CNN baseline under the identical val-locked protocol.

Adds the embedded-realistic comparison point: does 8-bit post-training weight
quantization preserve VEB detection, and what is the memory footprint? For each
seed we train the float CNN, evaluate it val-locked, then quantize its weights to
int8 (per-tensor symmetric, weights only; activations left in float) and evaluate
the quantized model val-locked on its own validation logits (no test tuning).

This is weight-only PTQ; full int8 (weights + activations) would need activation
calibration and is noted as future work. MAC count is unchanged by quantization,
but an int8 MAC is materially cheaper than an fp32 MAC on typical MCUs, and the
weight memory is 4x smaller.

Usage:
  python experiments/quantized_cnn.py --seeds 3
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lock_snn_rr import TRAIN_RECORDS, VAL_RECORDS, class_weights, raw_group  # noqa: E402
from freeze_veb_v1 import op_sens_first, metr  # noqa: E402
from baselines_veb import BATCH, logits_of, macs_cnn, train  # noqa: E402
from neurocardio.config import Config  # noqa: E402
from neurocardio.data.splits import DS2_RECORDS  # noqa: E402
from neurocardio.models.baselines import CNN1D  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

N_CLASSES, N_RR = 5, 3


def quant_int8(model):
    """Per-tensor symmetric int8 weight quantization (weights only), in place."""
    for n, p in model.named_parameters():
        if n.endswith("weight"):
            w = p.data
            scale = w.abs().max() / 127.0
            if scale > 0:
                p.data = torch.clamp(torch.round(w / scale), -127, 127) * scale
    return model


def weight_bytes(model, bits):
    nw = sum(p.numel() for n, p in model.named_parameters() if n.endswith("weight"))
    nb = sum(p.numel() for n, p in model.named_parameters() if n.endswith("bias"))
    return nw * bits / 8 + nb * 4  # biases kept fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/quantized_cnn.json")
    args = ap.parse_args()
    device = resolve_device("auto")
    cfg = Config()

    trb, trl, trr = raw_group(cfg, TRAIN_RECORDS, "trainsub")
    vab, val_l, var = raw_group(cfg, VAL_RECORDS, "val")
    teb, tel, ter = raw_group(cfg, DS2_RECORDS, "test")
    L = trb.shape[1]
    mu, sd = trr.mean(0), trr.std(0) + 1e-6
    t = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)  # noqa: E731
    trx, vax, tex = t(trb), t(vab), t(teb)
    trrt, vart, tert = t((trr - mu) / sd), t((var - mu) / sd), t((ter - mu) / sd)
    trlt = torch.from_numpy(trl.astype(np.int64)).to(device)
    weight = class_weights(trl, "sqrt")
    print(f"device={device} beat_len={L} macs={macs_cnn(L):,}", flush=True)

    def evaluate(model):
        (bias, _s, _p), _f = op_sens_first(logits_of(model, vax, vart, BATCH), val_l)
        return metr(logits_of(model, tex, tert, BATCH), tel, bias)

    frows, qrows = [], []
    for s in range(args.seeds):
        model = train(CNN1D(n_classes=N_CLASSES, n_rr=N_RR), trx, trrt, trlt, weight, device, s)
        f = evaluate(model)
        q = evaluate(quant_int8(copy.deepcopy(model)))
        frows.append(f); qrows.append(q)
        print(f"seed {s}: fp32 {f['VEB_sens']}/{f['VEB_ppv']}  ->  int8 {q['VEB_sens']}/{q['VEB_ppv']}",
              flush=True)

    def agg(rows, k):
        a = np.array([r[k] for r in rows]); return round(float(a.mean()), 4), round(float(a.min()), 4)

    ref = CNN1D(n_classes=N_CLASSES, n_rr=N_RR)
    summary = {
        "macs_per_beat": int(macs_cnn(L)),
        "weight_bytes_fp32": int(weight_bytes(ref, 32)),
        "weight_bytes_int8": int(weight_bytes(ref, 8)),
        "fp32": {"VEB_sens": agg(frows, "VEB_sens"), "VEB_ppv": agg(frows, "VEB_ppv")},
        "int8": {"VEB_sens": agg(qrows, "VEB_sens"), "VEB_ppv": agg(qrows, "VEB_ppv")},
    }
    print(f"\nfp32 DS2 VEB sens {summary['fp32']['VEB_sens']} ppv {summary['fp32']['VEB_ppv']}", flush=True)
    print(f"int8 DS2 VEB sens {summary['int8']['VEB_sens']} ppv {summary['int8']['VEB_ppv']}", flush=True)
    print(f"weight memory: fp32 {summary['weight_bytes_fp32']} B -> int8 {summary['weight_bytes_int8']} B", flush=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
