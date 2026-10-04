"""Figures of the paper and its supplementary material, drawn from ``results_v2``.

    python -m experiments.figures

Each model keeps one colour in every figure; bars carry their value, so no figure relies on
colour alone.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path("results_v2/figures")
COLOR = {"pinode": "#0072B2", "data": "#D55E00", "hybrid": "#009E73", "pikan": "#CC79A7",
         "data_gru": "#E69F00", "simple": "#8c8c8c"}
LABEL = {"pinode": "PI-NODE", "data": "DATA", "hybrid": "HYBRID", "pikan": "PI-KAN", "data_gru": "DATA-GRU",
         "Ridge on physical monomials": "Ridge", "Propeller law (constant K_Q)": "Propeller\nlaw",
         "B-series, constant wake": "B-series,\nconst. wake", "Fitted K_Q(J) quadratic": "Fitted\n$K_Q(J)$",
         "Gradient boosting": "Gradient\nboosting"}
VESSELS = ["A", "B", "C", "D"]

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


def summary():
    return json.loads(Path("results_v2/summary.json").read_text())["models"]


def color_of(key):
    return COLOR.get(key, COLOR["simple"])


# --------------------------------------------------------------------------- Fig. 1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc, ec, fs=7.4, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, linespacing=1.35)

    def arrow(x0, y0, x1, y1, color="#444444", style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=1.1, color=color, ls=ls))

    grey, blue, green = ("#f2f2f2", "#777777"), ("#dbe9f6", "#0072B2"), ("#d9f0e6", "#009E73")
    box(1, 33, 17, 12, "Calm-water inputs\n10 records, 1 min apart\nspeed, shaft speed,\ndrafts, trim", *grey, fs=6.8)
    box(1, 14, 17, 10, "Wind inputs\nlatest record", *grey, fs=6.8)
    box(23, 31, 25, 16, "ODE-RNN encoder\n" r"between records:  $\dot h = f_\theta(h)$" "\n(Runge-Kutta over elapsed time)\n"
        r"at each record:  $h \leftarrow \mathrm{GRU}(x_k, h)$", *blue, fs=6.8)
    box(23, 14, 25, 10, "Wind network (MLP)", *blue, fs=6.8)
    box(53, 24, 13, 13, "Effective\nwake\n" r"$w \in (0,\,0.75)$", *blue, fs=7.2, weight="bold")
    box(71, 17, 19, 28, "Propeller layer\n(closed form)\n\n" r"$J = V(1-w)/(nD)$" "\n\n" r"$K_Q(J)$: B-series" "\n" r"cubic $+\ \delta$" "\n\n"
        r"$P = 2\pi\rho\,n^3 D^5 K_Q$", *green, fs=6.9)
    box(93.5, 26, 6, 10, r"$\hat P$" "\n[kW]", *grey, fs=7.4)
    box(40, 1.5, 25, 9, "Measured at the latest record\n" r"$n$ (shaft speed),  $V$ (speed)", "#ffffff", "#777777", fs=6.4)
    box(69, 1.5, 30, 9, "Propeller particulars of the vessel\n" r"$D,\ P/D,\ A_E/A_0,\ Z$" "\n(swapped on transfer)", "#fff4dc", "#b8860b", fs=6.4)

    arrow(18.6, 39, 22.6, 39); arrow(18.6, 19, 22.6, 19)
    arrow(48.6, 39, 54, 37.6); arrow(48.6, 19, 54, 23.4)
    arrow(66.6, 30.5, 70.6, 30.5); arrow(90.6, 31, 93.1, 31)
    arrow(58, 10.9, 72.5, 16.6, color="#777777"); arrow(85, 10.9, 83, 16.6, color="#b8860b")
    ax.text(50, 48.2, "trained end to end on measured shaft power; gradients pass through the propeller layer",
            ha="center", va="center", fontsize=6.6, style="italic", color="#555555")
    save(fig, "F1_architecture")


# --------------------------------------------------------------------------- Fig. 2
def fig_source():
    m = summary()
    order = ["pinode", "data", "hybrid", "pikan", "data_gru", "Ridge on physical monomials", "Fitted K_Q(J) quadratic",
             "Propeller law (constant K_Q)", "B-series, constant wake"]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for i, k in enumerate(order):
        mean, sd = m[k]["test"]["rmse"]
        ax.bar(i, mean, 0.66, color=color_of(k), yerr=sd if sd else None, error_kw=dict(lw=0.9, capsize=2.5, ecolor="#333333"),
               hatch=None if k in COLOR else "///", edgecolor="white", linewidth=0.8)
        ax.text(i, mean + (sd or 0) + 9, f"{mean:.0f}", ha="center", va="bottom", fontsize=7.6)
    ax.axvline(4.5, color="#bbbbbb", lw=0.8, ls="--")
    ax.text(2.0, 610, "neural models (five seeds)", ha="center", fontsize=7.4, color="#555555")
    ax.text(6.9, 610, "closed-form and linear baselines", ha="center", fontsize=7.4, color="#555555")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LABEL.get(k, k) for k in order], fontsize=7.4)
    ax.set_ylabel("Test RMSE (kW)"); ax.set_ylim(0, 650); ax.grid(axis="x", visible=False)
    save(fig, "F2_source")


# --------------------------------------------------------------------------- Fig. 3
def fig_zeroshot():
    m = summary()
    order = ["pinode", "data", "hybrid", "pikan"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    width = 0.19
    for j, k in enumerate(order):
        vals = [m[k][f"zs_{v}"]["mape"][0] for v in VESSELS]
        sds = [m[k][f"zs_{v}"]["mape"][1] for v in VESSELS]
        x = np.arange(len(VESSELS)) + (j - (len(order) - 1) / 2) * (width + 0.012)
        ax.bar(x, vals, width, color=color_of(k), yerr=sds, error_kw=dict(lw=0.8, capsize=1.8, ecolor="#333333"),
               edgecolor="white", linewidth=0.6, label=LABEL[k])
        for xi, v, sd in zip(x, vals, sds):
            ax.text(xi, (v + sd) * 1.06, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=6.4)
    ax.set_yscale("log"); ax.set_ylim(2.5, 190); ax.set_yticks([3, 5, 10, 20, 50, 100]); ax.set_yticklabels(["3", "5", "10", "20", "50", "100"])
    ax.set_xticks(range(len(VESSELS))); ax.set_xticklabels([f"Vessel {v}" for v in VESSELS])
    ax.set_ylabel("Zero-shot MAPE (%, log scale)"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7.4, columnspacing=1.4, handlelength=1.2)
    save(fig, "F3_zeroshot")


# --------------------------------------------------------------------------- Fig. 4
def fig_fewshot():
    rows = sum([json.load(open(f)) for f in sorted(glob.glob("results_v2/fewshot_seed*.json"))], [])
    d = pd.DataFrame(rows)
    names = {"pinode:full": "pinode", "data:full": "data", "hybrid:full": "hybrid", "data_gru:full": "data_gru",
             "bseries_const_wake": "B-series, constant wake"}
    marks = {"pinode": "o", "data": "s", "hybrid": "^", "data_gru": "D", "B-series, constant wake": "x"}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), sharey=True)
    for ax, v in zip(axes, VESSELS):
        for model, key in names.items():
            g = d[(d.vessel == v) & (d.model == model)].groupby("frac").mape.mean()
            ax.plot(g.index * 100, g.values, marker=marks[key], ms=3.6, lw=1.4, color=color_of(key),
                    ls="--" if key not in COLOR else "-", label=LABEL.get(key, key).replace("\n", " "))
        ax.set_title(f"Vessel {v}", fontsize=8.5); ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 25]); ax.set_xticklabels(["1", "5", "10", "25"])
        ax.set_yscale("log"); ax.set_yticks([2, 5, 10, 20, 30]); ax.set_yticklabels(["2", "5", "10", "20", "30"]); ax.minorticks_off()
        ax.set_xlabel("Target data used (%)")
    axes[0].set_ylabel("MAPE after adaptation (%)")
    axes[0].legend(ncol=5, loc="upper left", bbox_to_anchor=(0.0, 1.36), fontsize=7.2, columnspacing=1.2, handlelength=1.8)
    save(fig, "F4_fewshot")


# --------------------------------------------------------------------------- Fig. 5
def fig_ablation():
    m = summary()
    order = [("pinode", "PI-NODE (proposed)"), ("pinode_gru", "GRU encoder"), ("pinode_static", "No memory"),
             ("pinode_relax", "Relaxation ODE"), ("pinode_fixedkq", "Fixed $K_Q$"), ("pinode_noweather", "No wind branch"),
             ("pinode_raw", "Dimensional inputs only"), ("pinode_nd", "Non-dimensional inputs only")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (getter, xlab, top, fmt) in zip(axes, [
            (lambda r: r["test"]["rmse"], "Source test RMSE (kW)", 400, "{:.0f}"),
            (lambda r: [np.mean([r[f"zs_{v}"]["mape"][0] for v in VESSELS]), np.mean([r[f"zs_{v}"]["mape"][1] for v in VESSELS])],
             "Zero-shot MAPE, mean of vessels A-D (%)", 15.5, "{:.1f}")]):
        for yi, (k, _) in zip(y, order):
            mean, sd = getter(m[k])
            ax.barh(yi, mean, 0.66, color=COLOR["pinode"] if k == "pinode" else "#9fc5e0", xerr=sd,
                    error_kw=dict(lw=0.8, capsize=2, ecolor="#333333"), edgecolor="white", linewidth=0.8)
            ax.text(mean + sd + top * 0.015, yi, fmt.format(mean), va="center", fontsize=7)
        ax.set_xlabel(xlab); ax.set_xlim(0, top); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([l for _, l in order], fontsize=7.4)
    fig.tight_layout(w_pad=1.4)
    save(fig, "F5_ablation")


# --------------------------------------------------------------------------- Fig. 6
def fig_calibration():
    u = json.loads(Path("results_v2/uncertainty.json").read_text())["chronological"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    methods = [("raw", "Raw ensemble", "#bdbdbd"), ("temperature", "Temperature scaling", "#7fb3d5"), ("conformal", "Split conformal", COLOR["pinode"])]
    for j, (key, lab, c) in enumerate(methods):
        vals = [u[f"{key}@{lv}"][0] for lv in (0.9, 0.95)]
        x = np.arange(2) + (j - 1) * 0.27
        ax.bar(x, vals, 0.25, color=c, label=lab, edgecolor="white", linewidth=0.8)
        for xi, v in zip(x, vals):
            ax.text(xi, v - 3.5, f"{v:.1f}", ha="center", va="top", fontsize=6.8, color="white" if key == "conformal" else "#222222")
    for i, lv in enumerate((90, 95)):
        ax.plot([i - 0.45, i + 0.45], [lv, lv], color="#333333", lw=1.0, ls="--")
    ax.text(1.47, 95, "target", fontsize=6.8, va="center", color="#333333")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["90 % interval", "95 % interval"]); ax.set_ylim(0, 112)
    ax.set_ylabel("Empirical coverage on test (%)"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, columnspacing=0.9, handlelength=1.0)
    save(fig, "F6_calibration")


# --------------------------------------------------------------------------- Graphical abstract
def fig_graphical_abstract():
    """Journal graphical abstract: 1980 x 792 px at 300 dpi, readable at 13 x 5 cm."""
    m = summary()
    fig = plt.figure(figsize=(6.6, 2.64))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc, ec, fs=6.4, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, linespacing=1.3)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=1.0, color="#444444"))

    ax.text(2, 37.2, "PI-NODE: the network predicts the wake, the propeller physics gives the power",
            fontsize=8.6, fontweight="bold", va="center", color="#111111")
    y0, h = 26.2, 7.0
    box(2, y0, 13, h, "Ten minutes of\noperational records", "#f2f2f2", "#777777")
    arrow(15.5, 18.5, y0 + h / 2)
    box(19, y0, 13, h, "Neural ODE\nencoder", "#dbe9f6", COLOR["pinode"], weight="bold")
    arrow(32.5, 35.5, y0 + h / 2)
    box(36, y0, 10, h, "Effective\nwake $w$", "#dbe9f6", COLOR["pinode"])
    arrow(46.5, 49.5, y0 + h / 2)
    box(50, y0, 24, h, "Propeller layer: B-series $K_Q(J)$\nof the vessel's own propeller\n" r"$P = 2\pi\rho\,n^3D^5K_Q$",
        "#d9f0e6", "#009E73", fs=6.0)
    arrow(74.5, 77.5, y0 + h / 2)
    box(78, y0, 9, h, "Shaft\npower", "#f2f2f2", "#777777")
    ax.text(93.5, y0 + h / 2, "new ship:\nswap four\npropeller\nnumbers", ha="center", va="center", fontsize=5.8,
            color="#8a6400", style="italic", linespacing=1.25)

    # Left panel: accuracy on the training ship.
    a1 = fig.add_axes([0.075, 0.17, 0.25, 0.40])
    keys = ["pinode", "data", "hybrid", "pikan"]
    vals = [m[k]["test"]["rmse"][0] for k in keys]
    a1.bar(range(4), vals, 0.68, color=[COLOR[k] for k in keys], edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        a1.text(i, v + 12, f"{v:.0f}", ha="center", va="bottom", fontsize=6.4)
    a1.set_xticks(range(4)); a1.set_xticklabels([LABEL[k] for k in keys], fontsize=6.0)
    a1.set_ylim(0, 560); a1.set_yticks([]); a1.spines["left"].set_visible(False); a1.grid(False)
    a1.set_title("Training ship: error (kW)", fontsize=6.8, pad=3)

    # Right panel: error on ships never seen in training.
    a2 = fig.add_axes([0.42, 0.17, 0.555, 0.40])
    width = 0.2
    for j, k in enumerate(keys):
        zs = [m[k][f"zs_{v}"]["mape"][0] for v in VESSELS]
        x = np.arange(4) + (j - 1.5) * (width + 0.012)
        a2.bar(x, zs, width, color=COLOR[k], edgecolor="white", linewidth=0.6, label=LABEL[k])
        for xi, v in zip(x, zs):
            a2.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.4)
    a2.set_ylim(0, 112); a2.set_yticks([])
    a2.spines["left"].set_visible(False); a2.grid(False)
    a2.set_xticks(range(4)); a2.set_xticklabels([f"Vessel {v}" for v in VESSELS], fontsize=6.0)
    a2.set_title("Four ships never seen in training: error (%), no retraining", fontsize=6.8, pad=3)
    a2.legend(ncol=4, fontsize=5.6, loc="upper left", bbox_to_anchor=(0.0, 1.0), columnspacing=0.9, handlelength=1.0, borderpad=0.1)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "graphical_abstract.pdf", bbox_inches=None)
    fig.savefig(FIG / "graphical_abstract.png", dpi=300, bbox_inches=None)
    plt.close(fig)
    print("saved graphical_abstract")


# --------------------------------------------------------------------------- Supplementary
def fig_long_training():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    styles = {"ode": ("PI-NODE (ODE encoder)", COLOR["pinode"], "-"), "gru": ("GRU encoder", "#555555", "--"), "static": ("No memory", "#999999", ":")}
    for enc, (lab, c, ls) in styles.items():
        curves = [json.load(open(f))["curve"] for f in sorted(glob.glob(f"results_v2/ode_search/long_{enc}_lr0.0001_seed*.json"))]
        ep = [p["epoch"] for p in curves[0]]
        for ax, key in zip(axes, ("val", "test")):
            ax.plot(ep, np.mean([[p[key] for p in c_] for c_ in curves], axis=0), color=c, ls=ls, lw=1.4, label=lab)
    for ax, t in zip(axes, ("Validation blocks", "Test period")):
        ax.set_title(t, fontsize=8.5); ax.set_xlabel("Epoch"); ax.set_ylim(220, 340)
    axes[0].set_ylabel("RMSE (kW)"); axes[0].legend(fontsize=7.2)
    save(fig, "S1_long_training")


def fig_kq():
    import torch
    from experiments.train import OUT
    from pinode.bseries import open_water_cubics, polyval_ascending
    from pinode.data import load_vessel
    v = load_vessel("source")
    _, kq = open_water_cubics(v.p_d, v.ae_a0, v.z)
    J = np.linspace(0.2, 0.75, 100)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(J, 10 * polyval_ascending(kq, J), color="#333333", lw=1.6, label="Wageningen B-series")
    for s in range(5):
        d = torch.load(OUT / f"pinode_seed{s}.pt")["propeller.delta"].numpy()
        ax.plot(J, 10 * polyval_ascending(kq + d, J), color=COLOR["pinode"], lw=0.9, alpha=0.8, label="Learned (five seeds)" if s == 0 else None)
    ax.set_xlabel("Advance coefficient $J$"); ax.set_ylabel("$10\\,K_Q$"); ax.legend(fontsize=7.2)
    save(fig, "S2_kq_curves")


def fig_wake():
    from experiments.summarise import seed_runs
    from experiments.train import load_everything
    from pinode.baselines import torque_identity_wake
    source, scaler, train, val, test, targets = load_everything()
    w_data = torque_identity_wake(test, source)
    w_net = np.mean([r["test_wake"] for r in seed_runs("pinode_fixedkq")], axis=0)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    hb = ax.hexbin(w_data, w_net, gridsize=45, cmap="Blues", mincnt=1, bins="log", linewidths=0)
    ax.plot([0.25, 0.65], [0.25, 0.65], color="#333333", lw=0.9, ls="--")
    ax.set_xlim(0.25, 0.65); ax.set_ylim(0.25, 0.65); ax.set_aspect("equal")
    ax.set_xlabel("Torque-identity wake from measured power"); ax.set_ylabel("Wake predicted by the network")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03).set_label("records per cell (log)", fontsize=7)
    save(fig, "S3_wake")


if __name__ == "__main__":
    for f in (fig_architecture, fig_source, fig_zeroshot, fig_fewshot, fig_ablation, fig_calibration,
              fig_graphical_abstract, fig_long_training, fig_kq, fig_wake):
        f()
