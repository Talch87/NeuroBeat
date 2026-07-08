"""Render publication figures for the NeuroBeat paper.

Figure 1: protocol / no-leakage diagram (DS1 train + val holdout, frozen operating
          point, DS2 + external tests, cascade routing).
Figure 3: accuracy-energy Pareto (F1 vs operations/beat), SNN SynOps and dense MACs
          drawn as distinct operation units.
Figure 4: cross-database transfer (VEB sensitivity and PPV on DS2 / SVDB / INCART).

Outputs go to paper/figures/*.png (+ .pdf). Numbers are the locked results reported
in the paper; the cascade energy is read from the frozen artifact.

Usage:
  python experiments/make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import freeze_veb_v1 as fv  # noqa: E402

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def f1(s, p):
    return 2 * s * p / (s + p)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/name}.png/.pdf", flush=True)


# ---------------------------------------------------------------- Figure 1
def box(ax, x, y, w, h, text, fc, ec="#333", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y)}


def arrow(ax, p0, p1, style="-|>", color="#444", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 lw=1.2, color=color, linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


def figure1():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    train = "#dbe9f6"; val = "#fde9d0"; frozen = "#e6dcf2"; test = "#dcefdc"; casc = "#f6dede"

    ds1 = box(ax, 0.2, 3.3, 1.5, 0.9, "MIT-BIH\nDS1", "#eee")
    tr = box(ax, 2.2, 4.2, 2.2, 0.85, "DS1 train\n44,573 beats", train)
    vl = box(ax, 2.2, 2.5, 2.2, 0.85, "DS1 val holdout\n201 / 207 / 223", val)
    arrow(ax, ds1["r"], tr["l"]); arrow(ax, ds1["r"], vl["l"])

    net = box(ax, 5.0, 4.2, 2.3, 0.85, "Train LIF SNN\n(5 seeds)", train)
    op = box(ax, 5.0, 2.5, 2.3, 0.85, "Fit + FREEZE\noperating point", val)
    arrow(ax, tr["r"], net["l"]); arrow(ax, vl["r"], op["l"])

    fr = box(ax, 7.9, 3.35, 1.9, 0.85, "Frozen model\n+ op. point", frozen)
    arrow(ax, net["r"], fr["l"]); arrow(ax, op["r"], fr["l"])

    ds2 = box(ax, 10.3, 4.5, 1.55, 0.7, "DS2 test\n49,693", test, fs=8)
    svdb = box(ax, 10.3, 3.45, 1.55, 0.7, "SVDB\n(test-only)", test, fs=8)
    inc = box(ax, 10.3, 2.4, 1.55, 0.7, "INCART\n(test-only)", test, fs=8)
    for b in (ds2, svdb, inc):
        arrow(ax, fr["r"], b["l"])

    ax.text(6, 5.75, "Selection uses only DS1 (train + val); DS2 and external databases are evaluated once, frozen.",
            ha="center", fontsize=9.5, style="italic", color="#333")

    # cascade routing band
    ax.add_patch(FancyBboxPatch((0.2, 0.25), 11.65, 1.5, boxstyle="round,pad=0.01,rounding_size=0.03",
                                lw=1, edgecolor="#c88", facecolor="#fbf3f3", zorder=0))
    ax.text(0.45, 1.55, "Cascade routing (inference)", fontsize=9, color="#a33", weight="bold")
    scr = box(ax, 1.0, 0.55, 2.4, 0.8, "Sparse screener\n(every beat, ~6k SynOps)", casc, fs=8)
    fl = box(ax, 4.1, 0.55, 1.7, 0.8, "flag ~27%\nof beats", "#fff", fs=8)
    cf = box(ax, 6.5, 0.55, 2.6, 0.8, "3-seed ensemble confirmer\n(flagged beats only)", casc, fs=8)
    dec = box(ax, 9.9, 0.55, 1.7, 0.8, "VEB decision\n(both fire)", "#f0e0e0", fs=8)
    arrow(ax, scr["r"], fl["l"]); arrow(ax, fl["r"], cf["l"]); arrow(ax, cf["r"], dec["l"])

    save(fig, "fig1_protocol")


# ---------------------------------------------------------------- Figure 3
def figure3():
    gj = json.loads(Path("models/neurobeat-veb-v1/gated_ensemble.json").read_text())
    gated_syn = gj["synops"]["average_per_beat"]
    # (label, sens, ppv, ops, unit, marker, color)
    snn = [
        ("Single-stage SNN", 0.894, 0.490, 14200, "SynOps", "o", "#1f77b4"),
        ("Naive T32 cascade", 0.883, 0.622, 19400, "SynOps", "o", "#1f77b4"),
        ("Full ensemble (x5)", 0.932, 0.595, 71000, "SynOps", "s", "#1f77b4"),
        ("Gated-ensemble cascade", 0.923, 0.616, gated_syn, "SynOps", "*", "#d62728"),
    ]
    dense = [
        ("CNN (fp32)", 0.945, 0.434, 356000, "MACs", "^", "#2ca02c"),
        ("LSTM", 0.940, 0.178, 4260000, "MACs", "^", "#8c564b"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, s, p, ops, unit, mk, c in snn + dense:
        ax.scatter(ops, f1(s, p), s=190 if mk == "*" else 90, marker=mk, color=c,
                   edgecolor="black", linewidth=0.7, zorder=3)
        dy = 0.014 if lab != "Full ensemble (x5)" else -0.028
        ax.annotate(f"{lab}\nF1={f1(s,p):.2f}", (ops, f1(s, p)),
                    textcoords="offset points", xytext=(8, 6 if dy > 0 else -18),
                    fontsize=8)
    ax.axvline(25000, ls="--", color="#d62728", lw=1)
    ax.text(25000, 0.32, " 25k SynOps budget", color="#d62728", fontsize=8, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Operations per beat  (SNN: SynOps ; dense: MACs — different units)")
    ax.set_ylabel("DS2 VEB F1")
    ax.set_title("Accuracy vs compute (Pareto view)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_ylim(0.25, 0.80)
    save(fig, "fig3_pareto")


# ---------------------------------------------------------------- Figure 4
def figure4():
    dbs = ["DS2", "SVDB", "INCART"]
    sens = [0.923, 0.904, 0.901]
    ppv = [0.616, 0.377, 0.835]
    x = range(len(dbs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar([i - w / 2 for i in x], sens, w, label="VEB sensitivity", color="#1f77b4", edgecolor="black", lw=0.6)
    ax.bar([i + w / 2 for i in x], ppv, w, label="VEB PPV", color="#ff7f0e", edgecolor="black", lw=0.6)
    ax.axhline(0.90, ls="--", color="#1f77b4", lw=1, alpha=0.7)
    ax.axhline(0.60, ls="--", color="#ff7f0e", lw=1, alpha=0.7)
    for i, (s, p) in enumerate(zip(sens, ppv)):
        ax.text(i - w / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(dbs)
    ax.set_ylabel("Value"); ax.set_ylim(0, 1.0)
    ax.set_title("Gated cascade: cross-database VEB transfer (one frozen operating point)")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    save(fig, "fig4_crossdb")


# ---------------------------------------------------------------- Figure 2
def figure2():
    lab = np.load(fv.CACHE / "labels_ds2.npy")
    seeds = sorted(int(p.stem[4:]) for p in fv.CACHE.glob("seed*.pt"))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in seeds:
        lg = np.load(fv.CACHE / f"seed{s}_ds2.npy")
        S, P = [], []
        for bV in np.linspace(-2, 10, 90):
            pred = (lg + np.array([0.0, 0.0, bV, -12.0, -12.0])).argmax(1)
            vs, vp = fv._metrics(pred, lab, 2)
            S.append(vs); P.append(vp)
        ax.plot(S, P, alpha=0.55, lw=1.3, label=f"seed {s}")
    ops = {"sens-first": (0.905, 0.539), "balanced": (0.857, 0.679), "ppv-first": (0.845, 0.700)}
    for name, (s, p) in ops.items():
        ax.scatter(s, p, s=140, marker="X", zorder=5, edgecolor="black", color="#333")
        ax.annotate(name, (s, p), textcoords="offset points", xytext=(7, 6), fontsize=8.5)
    ax.axvline(0.90, ls="--", color="gray", lw=1); ax.axhline(0.60, ls="--", color="gray", lw=1)
    ax.text(0.902, 0.30, "sens 0.90", color="gray", fontsize=8, rotation=90)
    ax.set_xlabel("DS2 VEB sensitivity"); ax.set_ylabel("DS2 VEB PPV")
    ax.set_title("Single-stage sensitivity-PPV frontier (per seed, DS2 sweep)")
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.2, 0.85)
    ax.legend(fontsize=8, loc="lower left", ncol=2); ax.grid(ls=":", alpha=0.4)
    save(fig, "fig2_frontier")


# ---------------------------------------------------------------- Figure 5
def figure5():
    # SVEB specialist, DS2, per seed (single lead): (sensitivity, PPV)
    pts = [(0.2700, 0.0697), (0.1584, 0.1508), (0.2537, 0.0808),
           (0.9385, 0.0630), (0.8176, 0.0382)]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=95, color="#d62728",
               edgecolor="black", zorder=3, label="SVEB specialist on DS2 (single lead, per seed)")
    for i, (x, y) in enumerate(pts):
        ax.annotate(f"seed {i}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    # same architecture on 12-lead INCART (cross-database evaluation): SVEB 0.62 / 0.27
    ax.scatter([0.62], [0.27], s=200, marker="*", color="#2ca02c", edgecolor="black",
               zorder=4, label="same architecture on 12-lead INCART")
    ax.annotate("12-lead INCART\nSVEB 0.62 / 0.27", (0.62, 0.27),
                textcoords="offset points", xytext=(-10, 10), fontsize=8.5, color="#186a18")
    ax.set_xlabel("SVEB sensitivity"); ax.set_ylabel("SVEB PPV")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.4)
    ax.set_title("Single-lead SVEB is unstable and low-precision (negative result)")
    ax.legend(fontsize=8, loc="upper center"); ax.grid(ls=":", alpha=0.4)
    save(fig, "fig5_sveb")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print("done", flush=True)
