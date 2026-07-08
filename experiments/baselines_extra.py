"""Stronger embedded baselines + paired comparison against the gated cascade.

Adds the baselines the paper previously listed only as future work, under the
identical validation-locked protocol (same DS1 train, same DS1-val sens-first
operating point, same frozen DS2 test):
  - TCN (compact temporal convolutional network)
  - ResNet-lite (compact 1-D residual CNN)
  - gradient-boosted trees (HistGradientBoosting) on morphology + RR, VEB-vs-rest
  - linear SVM on morphology + RR, VEB-vs-rest
  - CNN (fp32 and weight-only int8), retrained here to obtain per-beat predictions

For every model we freeze a single artifact (NN: the seed with the best validation
VEB F1 among sens-first-feasible seeds; classical: a fixed random_state), save its
DS2 per-beat VEB predictions, and run a paired record-level bootstrap of the gated
cascade against it (2,000 record resamples). Nothing is tuned on DS2.

Usage:
  python experiments/baselines_extra.py --seeds 3
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import freeze_veb_v1 as fv  # noqa: E402
import gated_ensemble_veb as g  # noqa: E402
import lock_snn_rr as L  # noqa: E402
from baselines_veb import train as nn_train, logits_of as nn_logits, macs_cnn, BATCH  # noqa: E402
from freeze_veb_v1 import op_sens_first, metr  # noqa: E402
from quantized_cnn import quant_int8  # noqa: E402
from lock_snn_rr import TRAIN_RECORDS, VAL_RECORDS, class_weights, raw_group  # noqa: E402
from neurocardio.config import Config  # noqa: E402
from neurocardio.data.dataset import build_split_rr  # noqa: E402
from neurocardio.data.splits import DS2_RECORDS  # noqa: E402
from neurocardio.models.baselines import CNN1D, TCN1D, ResNetLite1D  # noqa: E402
from neurocardio.models.snn import SNNClassifier  # noqa: E402
from neurocardio.train.loop import resolve_device  # noqa: E402

VEB, N_CLASSES, N_RR = 2, 5, 3
PREDS = Path("runs/preds")


def count_macs(model, Lbeat, device):
    """Exact conv + linear MACs per beat via forward hooks (BN/activations excluded,
    matching the SynOps convention of counting only the dominant operations)."""
    tot = {"v": 0}
    hooks = []

    def conv_hook(m, i, o):
        tot["v"] += o.shape[1] * o.shape[2] * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin_hook(m, i, o):
        tot["v"] += m.out_features * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, Lbeat, device=device), torch.zeros(1, N_RR, device=device))
    for h in hooks:
        h.remove()
    return int(tot["v"])


def params(model):
    return int(sum(p.numel() for p in model.parameters()))


def freeze_nn(build, name, data, device, seeds):
    trx, trrt, trlt, weight, vax, vart, val_l, tex, tert, tel = data
    rows, best = [], None
    for s in range(seeds):
        m = nn_train(build(), trx, trrt, trlt, weight, device, s)
        (bias, vs, vp), feas = op_sens_first(nn_logits(m, vax, vart, BATCH), val_l)
        ds2log = nn_logits(m, tex, tert, BATCH)
        d = metr(ds2log, tel, bias)
        rows.append(d)
        vf1 = 2 * vs * vp / max(vs + vp, 1e-9)
        key = (1 if feas else 0, vf1)
        if best is None or key > best["key"]:
            best = {"key": key, "bias": bias, "model": m, "ds2log": ds2log, "feas": feas}
        print(f"  {name} seed {s}: DS2 {d['VEB_sens']}/{d['VEB_ppv']} feas={feas}", flush=True)
    preds = (best["ds2log"] + best["bias"]).argmax(1) == VEB
    vs = np.array([r["VEB_sens"] for r in rows]); vp = np.array([r["VEB_ppv"] for r in rows])
    agg = {"DS2_VEB_sens": [round(float(vs.mean()), 4), round(float(vs.min()), 4)],
           "DS2_VEB_ppv": [round(float(vp.mean()), 4), round(float(vp.min()), 4)]}
    return best, preds, agg


def threshold_sens_first(val_score, val_veb, target=0.90):
    ths = np.unique(val_score)
    P = int(val_veb.sum()); best_t, best_ppv = None, -1.0
    for t in ths:
        pred = val_score >= t
        tp = int((pred & val_veb).sum()); fp = int((pred & ~val_veb).sum())
        sens = tp / max(P, 1); ppv = tp / max(tp + fp, 1)
        if sens >= target and ppv > best_ppv:
            best_ppv, best_t = ppv, t
    return float(best_t if best_t is not None else ths.min())


def binary_eval(ds2_score, ds2_veb, thr):
    pred = ds2_score >= thr
    tp = int((pred & ds2_veb).sum()); fp = int((pred & ~ds2_veb).sum()); fn = int((~pred & ds2_veb).sum())
    return pred, round(tp / max(tp + fn, 1), 4), round(tp / max(tp + fp, 1), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/baselines_extra.json")
    args = ap.parse_args()
    device = resolve_device("auto"); cfg = Config()
    PREDS.mkdir(parents=True, exist_ok=True)

    trb, trl, trr = raw_group(cfg, TRAIN_RECORDS, "trainsub")
    vab, val_l, var = raw_group(cfg, VAL_RECORDS, "val")
    teb, tel, ter = raw_group(cfg, DS2_RECORDS, "test")
    Lbeat = trb.shape[1]
    mu, sd = trr.mean(0), trr.std(0) + 1e-6
    tt = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)  # noqa: E731
    trx, vax, tex = tt(trb), tt(vab), tt(teb)
    trrt, vart, tert = tt((trr - mu) / sd), tt((var - mu) / sd), tt((ter - mu) / sd)
    trlt = torch.from_numpy(trl.astype(np.int64)).to(device)
    weight = class_weights(trl, "sqrt")
    data = (trx, trrt, trlt, weight, vax, vart, val_l, tex, tert, tel)
    print(f"device={device} beat_len={Lbeat} train={len(trb)} val={len(vab)} DS2={len(teb)}", flush=True)

    summary, preds = {}, {}

    # ---- NN baselines (TCN, ResNet-lite, CNN) ----
    nn_specs = {
        "TCN": lambda: TCN1D(n_classes=N_CLASSES, n_rr=N_RR),
        "ResNet-lite": lambda: ResNetLite1D(n_classes=N_CLASSES, n_rr=N_RR),
        "CNN": lambda: CNN1D(n_classes=N_CLASSES, n_rr=N_RR),
    }
    for name, build in nn_specs.items():
        t0 = time.time()
        best, p, agg = freeze_nn(build, name, data, device, args.seeds)
        preds[name] = p
        summary[name] = {**agg, "params": params(build()),
                         "macs_per_beat": count_macs(build().to(device), Lbeat, device)}
        print(f"{name}: {agg}  {summary[name]['macs_per_beat']:,} MACs  ({time.time()-t0:.0f}s)", flush=True)

        if name == "CNN":  # weight-only int8 variant, val-relocked
            qm = quant_int8(copy.deepcopy(best["model"]))
            (qbias, _, _), _ = op_sens_first(nn_logits(qm, vax, vart, BATCH), val_l)
            q_ds2 = nn_logits(qm, tex, tert, BATCH)
            preds["CNN-int8"] = (q_ds2 + qbias).argmax(1) == VEB
            qm_metr = metr(q_ds2, tel, qbias)
            summary["CNN-int8"] = {"DS2_VEB_sens": [qm_metr["VEB_sens"], qm_metr["VEB_sens"]],
                                   "DS2_VEB_ppv": [qm_metr["VEB_ppv"], qm_metr["VEB_ppv"]],
                                   "macs_per_beat": summary["CNN"]["macs_per_beat"],
                                   "note": "weight-only int8, 4x smaller weights"}
            print(f"CNN-int8: {qm_metr['VEB_sens']}/{qm_metr['VEB_ppv']}", flush=True)

    # ---- classical baselines on morphology + RR (VEB-vs-rest) ----
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.svm import LinearSVC
        bmu, bsd = trb.mean(0), trb.std(0) + 1e-6
        Xtr = np.concatenate([(trb - bmu) / bsd, (trr - mu) / sd], axis=1)
        Xval = np.concatenate([(vab - bmu) / bsd, (var - mu) / sd], axis=1)
        Xds2 = np.concatenate([(teb - bmu) / bsd, (ter - mu) / sd], axis=1)
        ytr = (trl == VEB).astype(int)
        vveb, dveb = val_l == VEB, tel == VEB
        pos = int(ytr.sum()); n = len(ytr)
        sw = np.where(ytr == 1, n / (2 * pos), n / (2 * (n - pos)))

        t0 = time.time()
        gbt = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.1,
                                             random_state=0)
        gbt.fit(Xtr, ytr, sample_weight=sw)
        thr = threshold_sens_first(gbt.predict_proba(Xval)[:, 1], vveb)
        p, s, pv = binary_eval(gbt.predict_proba(Xds2)[:, 1], dveb, thr)
        preds["GBT"] = p
        summary["GBT"] = {"DS2_VEB_sens": [s, s], "DS2_VEB_ppv": [pv, pv],
                          "compute_per_beat": "~200 trees x depth 4",
                          "note": "gradient-boosted trees, morphology+RR, VEB-vs-rest"}
        print(f"GBT: DS2 {s}/{pv}  ({time.time()-t0:.0f}s)", flush=True)

        t0 = time.time()
        svm = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
        svm.fit(Xtr, ytr)
        thr = threshold_sens_first(svm.decision_function(Xval), vveb)
        p, s, pv = binary_eval(svm.decision_function(Xds2), dveb, thr)
        preds["SVM"] = p
        summary["SVM"] = {"DS2_VEB_sens": [s, s], "DS2_VEB_ppv": [pv, pv],
                          "compute_per_beat": f"{Xtr.shape[1]} MACs (linear)",
                          "note": "linear SVM, morphology+RR, VEB-vs-rest"}
        print(f"SVM: DS2 {s}/{pv}  ({time.time()-t0:.0f}s)", flush=True)
    except ImportError as e:
        print(f"[skip classical baselines: {e}]", flush=True)

    # ---- reconstruct the frozen cascade on the same DS2 ordering ----
    gj = json.loads(Path("models/neurobeat-veb-v1/gated_ensemble.json").read_text())
    b1 = np.array(gj["screener"]["bias"]); b2 = np.array(gj["confirmer"]["bias"])
    d = L.load_data(cfg, g.SCREENER, device, external_specs=None, augment_specs=None)
    tex_s, ter_s = d[6], d[7]
    m1 = SNNClassifier(in_features=2 * len(g.SCREENER["orders"]), hidden=g.SCREENER["hidden"],
                       n_classes=5, n_rr=3).to(device)
    m1.load_state_dict(torch.load("models/neurobeat-veb-v1/screener_weights.pt", map_location=device))
    scr = L.logits_of(m1, tex_s, ter_s, 512)
    conf = np.mean([np.load(fv.CACHE / f"seed{s}_ds2.npy") for s in g.ENSEMBLE_SEEDS], axis=0)
    lab = np.load(fv.CACHE / "labels_ds2.npy")
    assert np.array_equal(lab, tel), "DS2 label order mismatch between SNN cache and raw_group"
    flag = (scr + b1).argmax(1) == VEB
    preds["cascade"] = flag & ((conf + b2).argmax(1) == VEB)

    # ---- per-beat DS2 record ids ----
    rid = []
    for rec in DS2_RECORDS:
        beats, _, _ = build_split_rr(cfg, [rec]); rid.extend([rec] * len(beats))
    rid = np.array(rid)
    assert len(rid) == len(lab)
    true = lab == VEB
    by_rec = {r: np.where(rid == r)[0] for r in DS2_RECORDS}
    R = len(DS2_RECORDS)

    for name, p in preds.items():
        np.save(PREDS / f"{name}_ds2.npy", p)

    def f1_of(pred, tru):
        tp = (pred & tru).sum(); fn = (~pred & tru).sum(); fp = (pred & ~tru).sum()
        s = tp / max(tp + fn, 1); pv = tp / max(tp + fp, 1)
        return 2 * s * pv / max(s + pv, 1e-9)

    def paired(a, b):
        rng = np.random.default_rng(0); diffs = []
        for _ in range(2000):
            ch = rng.integers(0, R, R)
            idx = np.concatenate([by_rec[DS2_RECORDS[c]] for c in ch])
            diffs.append(f1_of(a[idx], true[idx]) - f1_of(b[idx], true[idx]))
        diffs = np.sort(np.array(diffs))
        return {"median": round(float(np.median(diffs)), 4),
                "ci": [round(float(diffs[50]), 4), round(float(diffs[1949]), 4)],
                "frac_cascade_better": round(float((diffs > 0).mean()), 3)}

    casc = preds["cascade"]
    paired_res = {name: paired(casc, preds[name]) for name in preds if name != "cascade"}
    summary["paired_vs_cascade_F1"] = paired_res

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== DS2 VEB (mean[min]) ===", flush=True)
    for name in ["TCN", "ResNet-lite", "CNN", "CNN-int8", "GBT", "SVM"]:
        if name in summary:
            s = summary[name]
            print(f"{name:12} sens {s['DS2_VEB_sens']}  ppv {s['DS2_VEB_ppv']}", flush=True)
    print("\n=== paired record-level bootstrap: cascade F1 minus baseline F1 ===", flush=True)
    for name, r in paired_res.items():
        print(f"cascade vs {name:12} {r['median']:+.3f}  CI {r['ci']}  "
              f"(cascade better {r['frac_cascade_better']*100:.0f}%)", flush=True)
    print(f"\nsaved -> {args.out} and {PREDS}/*.npy", flush=True)


if __name__ == "__main__":
    main()
