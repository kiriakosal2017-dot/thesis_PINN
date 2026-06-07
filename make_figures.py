"""
make_figures.py — generates all publication figures (F1–F10) for the PI-NODE paper.

Each fig_* function writes one figure to results/figures/ as both a vector PDF
(for LaTeX inclusion) and a 300-dpi PNG preview.  Loss-curve data is loaded from
saved CSV histories when available, falling back to regex-parsed training logs;
all other numeric results are stored as constants below with provenance comments.
"""
import os
import re
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Style / paths
# --------------------------------------------------------------------------- #
FIG_DIR = "results/figures"
HIST_DIR = "results/history"

# Consistent palette across all figures: green = physics-informed, purple = KAN,
# blue = data-only, red = hybrid.  ORDER drives the axis ordering in most plots.
COLORS = {"PI-NODE": "#1b7837", "PI-KAN": "#762a83", "DATA": "#2166ac", "HYBRID": "#b2182b"}
ORDER = ["PI-NODE", "DATA", "HYBRID"]

# Minimal rcParams: grid at 30% opacity keeps clutter low; spines removed on top/right
# for a cleaner half-frame look consistent with the paper's style guide.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig, name):
    # tight_layout before save prevents label clipping; close immediately to keep
    # memory flat when regenerating all ten figures in one pass.
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    png = os.path.join(FIG_DIR, name + ".png")
    fig.tight_layout()
    fig.savefig(pdf)            # vector, for LaTeX
    fig.savefig(png)            # preview
    plt.close(fig)
    print(f"  saved {pdf} + {png}")


# --------------------------------------------------------------------------- #
# Loss-history loading: prefer saved CSV, fall back to log parsing
# --------------------------------------------------------------------------- #
def _load_history_csv(path):
    # Structured CSV is the authoritative source when available; val_loss may
    # be absent in early-epoch rows, so .get() is used defensively.
    epochs, train, val = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]) if row["train_loss"] else None)
            val.append(float(row["val_loss"]) if row.get("val_loss") else None)
    return epochs, train, val


def _parse_log(path, pattern, section_start=None, section_end=None):
    """Extract (train, val) loss per epoch from a log file via regex.

    section_start / section_end: substrings bounding the relevant block
    (used to isolate DATA vs HYBRID inside baseline_rerun.log).
    """
    train, val = [], []
    if not os.path.exists(path):
        return train, val
    active = section_start is None
    with open(path, errors="ignore") as f:
        for line in f:
            if section_start and section_start in line:
                active = True
                continue
            if active and section_end and section_end in line:
                break
            if not active:
                continue
            m = pattern.search(line)
            if m:
                train.append(float(m.group("tr")))
                v = m.group("vl")
                val.append(float(v) if v not in (None, "") else None)
    return train, val


# Each model trainer prints epochs in a distinct format; separate patterns avoid
# false matches when DATA and HYBRID share a single log file.
# DATA:    "Epoch [163/1000], Training Loss: 0.0041, Validation Loss: 0.0545"
PAT_DATA = re.compile(
    r"Epoch \[\d+/\d+\], Training Loss:\s*(?P<tr>[\d.]+),\s*Validation Loss:\s*(?P<vl>[\d.]+)")
# HYBRID:  "Epoch 12: train_total=0.0312, val_data=0.0312"
PAT_HYBRID = re.compile(
    r"Epoch \d+: train_total=(?P<tr>[\d.]+), val_data=(?P<vl>[\d.]+)")
# PI-NODE: "Epoch 12 | Train Loss: 0.0187 | Val Loss: 0.0151 | Val RMSE: 202.12 kW"
PAT_PINODE = re.compile(
    r"Epoch \d+ \| Train Loss:\s*(?P<tr>[\d.]+) \| Val Loss:\s*(?P<vl>[\d.]+)")


def get_loss_history(model):
    """Return (train, val) lists. CSV history wins; else parse the log."""
    # CSV histories in results/history/ are written by training scripts and carry
    # the full epoch-by-epoch record.  Log files are the fallback for runs where
    # history saving was not configured.
    csv_path = os.path.join(HIST_DIR, f"{model.replace('-', '_')}_danae.csv")
    if os.path.exists(csv_path):
        _, tr, vl = _load_history_csv(csv_path)
        return tr, vl
    if model == "DATA":
        return _parse_log("baseline_rerun.log", PAT_DATA,
                          section_start="Training DATA Model",
                          section_end="Evaluating DATA Model")
    if model == "HYBRID":
        return _parse_log("baseline_rerun.log", PAT_HYBRID,
                          section_start="Training HYBRID Model",
                          section_end="Evaluating HYBRID Model")
    if model == "PI-NODE":
        return _parse_log("pinode_rerun.log", PAT_PINODE)
    return [], []


# --------------------------------------------------------------------------- #
# Numeric tables (provenance: docs/EXPERIMENT_RUNBOOK.md + training logs)
# --------------------------------------------------------------------------- #
# F2 source-domain test RMSE (kW) on the source vessel.
# PI-NODE/DATA/HYBRID are single canonical runs; PI-KAN is the 5-seed mean with
# ±std error bar sourced from results/multiseed_pikan_results.csv.
SOURCE_RMSE = {"PI-NODE": 312.52, "PI-KAN": 471.04, "DATA": 557.52, "HYBRID": 583.88}
SOURCE_ORDER = ["PI-NODE", "PI-KAN", "DATA", "HYBRID"]
SOURCE_RMSE_ERR = {"PI-KAN": 72.80}  # 5-seed std; others are single runs (no CI)

# F3 transient analysis (P75 |dV/dt| threshold)
TRANSIENT = {  # model: (steady_rmse, trans_rmse, steady_mape, trans_mape)
    "PI-NODE": (299.78, 347.80, 3.48, 3.86),
    "DATA":    (541.01, 604.34, 7.16, 8.36),
    "HYBRID":  (539.75, 699.74, 7.76, 10.91),
}

# F4 zero-shot transfer MAPE (%) per target vessel
TRANSFER = {  # ship: {model: mape}
    "KASTOR":   {"PI-NODE": 3.75,  "DATA": 9.47,  "HYBRID": 23.46},
    "MENELAOS": {"PI-NODE": 4.87,  "DATA": 39.72, "HYBRID": 35.05},
    "THALIA":   {"PI-NODE": 27.72, "DATA": 41.55, "HYBRID": 41.07},
    "THISSEAS": {"PI-NODE": 32.19, "DATA": 88.63, "HYBRID": 77.69},
}

# F5 few-shot MAPE (%) by training-data fraction.  DATA vs PI-NODE.
# A target vessel's 25% PI-NODE not yet available (run interrupted) -> None.
FEWSHOT = {
    "KASTOR": {
        "frac":    [1, 5, 10, 25],
        "DATA":    [9.29, 7.85, 9.05, 7.84],
        "PI-NODE": [5.56, 2.86, 2.82, 2.74],
    },
    "MENELAOS": {
        "frac":    [1, 5, 10, 25],
        "DATA":    [5.46, 3.09, 2.71, 2.50],
        "PI-NODE": [3.97, 3.02, 2.74, 2.66],
    },
}

# F7 ablation (source vessel) — source: results/ablation_results.csv
ABLATION = [  # (label, test_rmse_kw, mape)
    ("full",              312.52, 3.575),
    ("− neural ODE",      286.95, 3.670),
    ("− sea-state",       289.12, 3.235),
    ("frozen propeller",  941.06, 8.028),
    ("+ acceleration",    285.91, 3.418),
]

# F8 multi-seed (source vessel) — source: results/multiseed_results.csv
MULTISEED_RMSE = [290.59, 288.99, 289.97, 293.54, 268.99]  # seeds 0..4
MULTISEED_MEAN, MULTISEED_SD = 286.42, 9.89

# F9 uncertainty (source vessel) — source: uq_rerun.log
UQ = [  # (method, rmse_kw, mean_std_kw, coverage95_pct)
    ("MC-Dropout\n(K=30)", 310.98, 22.96, 15.6),
    ("Deep Ensemble\n(M=5)", 278.36, 62.13, 51.4),
]

# F10 calibration (source vessel, test eval-half) — source: results/calibration_results.csv
# Deep-Ensemble empirical coverage per method at each target level.
CALIBRATION = {
    0.90: {"raw": 0.4524, "temperature": 0.9566, "conformal": 0.9078},
    0.95: {"raw": 0.5150, "temperature": 0.9662, "conformal": 0.9548},
}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_loss_curves():
    """F1: training & validation loss per model (source domain)."""
    # sharey=False because loss scales differ substantially across model families;
    # log-scale y-axis makes convergence visible even when final loss is very small.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, model in zip(axes, ORDER):
        tr, vl = get_loss_history(model)
        if not tr:
            ax.set_title(f"{model} (no data)")
            continue
        ep = range(1, len(tr) + 1)
        ax.plot(ep, tr, color=COLORS[model], lw=1.6, label="Train")
        # Val loss may be logged less frequently than every epoch; skip None entries
        # rather than plotting zero or breaking the x-axis.
        vl_ep = [(i + 1, v) for i, v in enumerate(vl) if v is not None]
        if vl_ep:
            xs, ys = zip(*vl_ep)
            ax.plot(xs, ys, color=COLORS[model], lw=1.6, ls="--", alpha=0.7,
                    label="Validation")
        ax.set_title(model)
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_ylabel("Loss (MSE, scaled)")
    fig.suptitle("Training / validation loss (source vessel)", y=1.02)
    save(fig, "F1_loss_curves")


def fig_source_rmse():
    """F2: source-domain test RMSE bars comparing all four model families."""
    # PI-KAN carries a 5-seed std error bar; the other three have single-run values
    # so their error is 0 (no cap drawn).  ylim headroom leaves room for the labels.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = [SOURCE_RMSE[m] for m in SOURCE_ORDER]
    errs = [SOURCE_RMSE_ERR.get(m, 0.0) for m in SOURCE_ORDER]
    bars = ax.bar(SOURCE_ORDER, vals, color=[COLORS[m] for m in SOURCE_ORDER],
                  width=0.62, yerr=errs, capsize=4,
                  error_kw={"ecolor": "#333333", "elinewidth": 1.2})
    for b, v, e in zip(bars, vals, errs):
        label = f"{v:.0f}" + (f"±{e:.0f}" if e else "")
        ax.text(b.get_x() + b.get_width() / 2, v + e + 8, label,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_title("Source-domain accuracy")
    ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.15)
    save(fig, "F2_source_rmse")


def fig_transient():
    """F3: steady vs transient MAPE (grouped bars, P75 |dV/dt| threshold)."""
    # Steady bars are rendered at alpha=0.55 to visually distinguish them from the
    # opaque transient bars while keeping the same hue for each model.
    # MAPE (index 2/3 of the tuple) is used rather than RMSE because fractional
    # error is more interpretable across different power regimes.
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(ORDER))
    w = 0.38
    steady = [TRANSIENT[m][2] for m in ORDER]
    trans = [TRANSIENT[m][3] for m in ORDER]
    ax.bar([i - w / 2 for i in x], steady, w, label="Steady",
           color=[COLORS[m] for m in ORDER], alpha=0.55)
    ax.bar([i + w / 2 for i in x], trans, w, label="Transient",
           color=[COLORS[m] for m in ORDER])
    for i, m in enumerate(ORDER):
        ax.text(i - w / 2, steady[i] + 0.1, f"{steady[i]:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, trans[i] + 0.1, f"{trans[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ORDER)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Accuracy by operating regime (steady vs transient)")
    ax.legend()
    save(fig, "F3_transient_mape")


def fig_transfer():
    """F4: zero-shot transfer MAPE per target vessel (no fine-tuning)."""
    # The two nearer sister vessels show lower transfer error for PI-NODE;
    # the vessels with more distinct hull forms show a larger gap.
    # Width w=0.26 with j-1 centering keeps three bars tightly grouped under each ship label.
    ships = list(TRANSFER.keys())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(ships))
    w = 0.26
    for j, m in enumerate(ORDER):
        vals = [TRANSFER[s][m] for s in ships]
        ax.bar([i + (j - 1) * w for i in x], vals, w, label=m, color=COLORS[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels(ships)
    ax.set_ylabel("Zero-shot MAPE (%)")
    ax.set_title("Cross-vessel transfer (lower is better)")
    ax.legend()
    save(fig, "F4_zeroshot_transfer")


def fig_fewshot():
    """F5: few-shot MAPE vs fine-tuning data fraction — DATA vs PI-NODE per ship."""
    # sharey=True enables direct cross-ship comparison; None values are skipped so
    # an incomplete run does not break the line — a text annotation flags the gap.
    ships = list(FEWSHOT.keys())
    fig, axes = plt.subplots(1, len(ships), figsize=(6.5 * len(ships) / 2, 4),
                             sharey=True)
    if len(ships) == 1:
        axes = [axes]
    for ax, ship in zip(axes, ships):
        d = FEWSHOT[ship]
        for model in ("DATA", "PI-NODE"):
            pts = [(f, v) for f, v in zip(d["frac"], d[model]) if v is not None]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=COLORS[model], label=model, lw=1.8)
        # mark missing point if any
        if None in d["PI-NODE"]:
            ax.text(0.97, 0.95, "PI-NODE 25% pending", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="gray", style="italic")
        ax.set_title(ship)
        ax.set_xlabel("Fine-tuning data (%)")
        ax.legend()
    axes[0].set_ylabel("Few-shot MAPE (%)")
    fig.suptitle("Few-shot adaptation to unseen vessels", y=1.02)
    save(fig, "F5_fewshot")


def fig_ablation():
    """F7: ablation study — test RMSE per model variant, full model as reference."""
    # The "frozen propeller" bar is highlighted in red because its ~3× RMSE increase
    # (vs full) is the single most important finding: the learnable propeller map
    # is the architecture's decisive component.
    labels = [a[0] for a in ABLATION]
    rmse = [a[1] for a in ABLATION]
    colors = ["#1b7837" if lbl == "full" else "#7f7f7f" for lbl in labels]
    colors[3] = "#b2182b"  # frozen propeller — the critical degradation
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, rmse, color=colors, width=0.65)
    # Dashed reference line at the full-model RMSE aids direct visual comparison.
    ax.axhline(rmse[0], color="#1b7837", ls="--", lw=1, alpha=0.6,
               label="full (reference)")
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_title("Ablation: learnable propeller is decisive")
    ax.set_ylim(0, max(rmse) * 1.12)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, "F7_ablation")


def fig_multiseed():
    """F8: per-seed RMSE with mean±SD band, vs DATA/HYBRID baselines."""
    # The shaded band conveys variance at a glance; individual seed dots confirm no
    # outlier seeds.  ylim anchored to the HYBRID baseline keeps the scale readable
    # without wasting space above the worst baseline.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seeds = list(range(len(MULTISEED_RMSE)))
    ax.axhspan(MULTISEED_MEAN - MULTISEED_SD, MULTISEED_MEAN + MULTISEED_SD,
               color=COLORS["PI-NODE"], alpha=0.15, label="PI-NODE mean ± SD")
    ax.axhline(MULTISEED_MEAN, color=COLORS["PI-NODE"], lw=1.4)
    ax.plot(seeds, MULTISEED_RMSE, "o", color=COLORS["PI-NODE"], ms=9,
            label="PI-NODE per seed")
    # Baseline horizontals reuse SOURCE_RMSE so this figure stays consistent with F2.
    ax.axhline(SOURCE_RMSE["DATA"], color=COLORS["DATA"], ls="--", lw=1.3,
               label=f"DATA ({SOURCE_RMSE['DATA']:.0f})")
    ax.axhline(SOURCE_RMSE["HYBRID"], color=COLORS["HYBRID"], ls=":", lw=1.3,
               label=f"HYBRID ({SOURCE_RMSE['HYBRID']:.0f})")
    ax.text(0.02, MULTISEED_MEAN + MULTISEED_SD + 6,
            f"{MULTISEED_MEAN:.1f} ± {MULTISEED_SD:.1f} kW", color=COLORS["PI-NODE"],
            fontsize=10, va="bottom")
    ax.set_xticks(seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_ylim(0, SOURCE_RMSE["HYBRID"] * 1.1)
    ax.set_title("Multi-seed stability — PI-NODE far below baselines")
    ax.legend(loc="center right", fontsize=9)
    save(fig, "F8_multiseed")


def fig_uncertainty():
    """F9: empirical 95% prediction-interval coverage per UQ method (source vessel)."""
    # Both MC-Dropout and Deep Ensemble under-cover relative to the nominal 95%
    # line; the gap motivates the post-hoc recalibration shown in F10.
    # RMSE is annotated alongside coverage so the accuracy/uncertainty trade-off
    # is visible in a single bar.
    methods = [u[0] for u in UQ]
    cov = [u[3] for u in UQ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(methods, cov, color=["#7f7f7f", COLORS["PI-NODE"]], width=0.55)
    ax.axhline(95, color="#b2182b", ls="--", lw=1.4, label="nominal 95 %")
    for b, u in zip(bars, UQ):
        ax.text(b.get_x() + b.get_width() / 2, u[3] + 1.5,
                f"{u[3]:.1f}%\n(RMSE {u[1]:.0f} kW)", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylabel("Empirical 95 % interval coverage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("UQ calibration — both under-cover; ensemble closer")
    ax.legend(loc="upper left")
    save(fig, "F9_uncertainty")


def fig_calibration():
    """F10: empirical coverage before/after post-hoc recalibration (Deep Ensemble)."""
    # Three methods compared at both the 90% and 95% target levels.  The dashed
    # horizontal lines are the ideal targets; conformal prediction hits them by
    # construction on the calibration split, which is why it lands closest.
    # Source data: results/calibration_results.csv (Deep Ensemble, source vessel test eval-half).
    methods = ["raw", "temperature", "conformal"]
    colors = {"raw": "#b2182b", "temperature": "#f1a340", "conformal": "#1b7837"}
    levels = [0.90, 0.95]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(levels))
    w = 0.25
    for j, m in enumerate(methods):
        vals = [CALIBRATION[l][m] * 100 for l in levels]
        offs = (j - 1) * w
        bars = ax.bar([i + offs for i in x], vals, w, label=m, color=colors[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for l in levels:
        ax.axhline(l * 100, ls="--", color="#555555", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{int(l*100)}% target" for l in levels])
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Predictive-interval calibration (Deep Ensemble)")
    ax.legend(title="method")
    save(fig, "F10_calibration")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Generating paper figures -> results/figures/ (PDF + PNG)")
    fig_loss_curves()
    fig_source_rmse()
    fig_transient()
    fig_transfer()
    fig_fewshot()
    fig_ablation()
    fig_multiseed()
    fig_uncertainty()
    fig_calibration()
    print("Done.")
