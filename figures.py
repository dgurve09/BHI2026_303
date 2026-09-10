"""Paper figures. Run run_all.py extract first; the epoch example reads one raw EDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

import datasets as D
import evaluate as E
import features as F

RESULTS = Path(__file__).resolve().parent / "results"
N1C, REMC = "#C55A11", "#2E75B6"

# The three descriptors shown in Fig. 3.
FIG2 = [
    ("dyn_theta_alpha_mean", "Theta-alpha mean"),
    ("dyn_theta_alpha_crossing_rate", "Theta-alpha switching rate"),
    ("dyn_sigma_beta_std", "Sigma-beta variability"),
]
# The two example epochs used in Fig. 2.
FIG3_RECORD, FIG3_N1_EPOCH, FIG3_REM_EPOCH = "SC4821G0", 1799, 1679


def figure2():
    """Single-feature separability. AUCs are direction-adjusted, descriptive only."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    for r, subset in enumerate(["cassette", "telemetry"]):
        df = D.extract_sleep_edf(subset, task="n1_rem")
        y = (df.stage_name == "REM").astype(int).values
        for c, (col, label) in enumerate(FIG2):
            ax = axes[r, c]
            v = df[col].replace([np.inf, -np.inf], np.nan).values.astype(float)
            groups = [v[(y == 0) & np.isfinite(v)], v[(y == 1) & np.isfinite(v)]]
            parts = ax.violinplot(groups, positions=[0, 1], showextrema=False, widths=0.8)
            for body, colour in zip(parts["bodies"], [N1C, REMC]):
                body.set_facecolor(colour)
                body.set_alpha(0.45)
            for i, gvals in enumerate(groups):
                q1, med, q3 = np.percentile(gvals, [25, 50, 75])
                ax.vlines(i, q1, q3, color="0.2", lw=4)
                ax.hlines(med, i - 0.18, i + 0.18, color="white", lw=2, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["N1", "REM"])
            ax.set_title("%s\nAUC = %.3f" % (label, E.directional_auc(y, v)), fontsize=10)
            if c == 0:
                ax.set_ylabel("%s\n\nfeature value" % subset.capitalize(), fontsize=10)
            ax.grid(alpha=0.3, lw=0.6)
    fig.tight_layout()
    out = RESULTS / "fig2_feature_separability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out.name)


def figure3():
    """Worked example: two real epochs and their descriptors."""
    folder = D.SLEEP_EDF_DIR / D.subset_folder("cassette")
    if not folder.is_dir():
        folder = next(p for p in D.SLEEP_EDF_DIR.rglob(D.subset_folder("cassette")) if p.is_dir())
    raw = mne.io.read_raw_edf(folder / (FIG3_RECORD + "-PSG.edf"), preload=False, verbose="ERROR")
    ch = [c for c in raw.ch_names if "Fpz" in c][0]
    fs = int(raw.info["sfreq"])
    n = 30 * fs

    def epoch(i):
        return F.clean_epoch(raw.get_data(picks=[ch], start=i * n, stop=(i + 1) * n)[0], fs)

    ex = {"N1": (epoch(FIG3_N1_EPOCH), N1C, FIG3_N1_EPOCH),
          "REM": (epoch(FIG3_REM_EPOCH), REMC, FIG3_REM_EPOCH)}
    s = {}
    for k, (x, _, _) in ex.items():
        b = F.subwindow_spectra(x, fs)[0]
        ta = np.log((b[:, 1] + F.EPS) / (b[:, 2] + F.EPS))
        sb = np.log((b[:, 3] + F.EPS) / (b[:, 4] + F.EPS))
        d = np.diff(b, axis=0)
        den = np.linalg.norm(d[:-1], axis=1) * np.linalg.norm(d[1:], axis=1) + F.EPS
        turn = 1 - np.clip(np.sum(d[:-1] * d[1:], axis=1) / den, -1, 1)
        dom = ta > 0
        s[k] = dict(ta=ta, sb=sb, turn=turn, L=np.linalg.norm(d, axis=1).sum(), T=turn.mean(),
                    sw=int(np.sum(dom[1:] != dom[:-1])), sd=sb.std(),
                    swi=np.flatnonzero(dom[1:] != dom[:-1]) + 1)

    plt.rcParams.update({"font.size": 9.5, "font.family": "sans-serif", "axes.grid": True,
                         "grid.alpha": .25, "grid.linewidth": .5, "axes.linewidth": .8,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "legend.fontsize": 8.5, "legend.framealpha": .9})
    fig = plt.figure(figsize=(9.2, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[.80, 1, 1], hspace=.62, wspace=.26,
                          left=.085, right=.975, top=.93, bottom=.075)

    def panel(ax, letter, title):
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
        ax.text(-.105, 1.14, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="right")

    ax = fig.add_subplot(gs[0, :])
    for i, (k, (x, c, ep)) in enumerate(ex.items()):
        ax.plot(np.arange(len(x)) / fs, x / np.std(x) - 6.5 * i, lw=.55, color=c)
        ax.text(.35, -6.5 * i + 3.1, "%s epoch %d" % (k, ep), color=c, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 30)
    ax.grid(True, axis="x", alpha=.25, lw=.5)
    ax.set_yticks([])
    ax.set_xlabel("Time within epoch (s)")
    ax.set_ylabel("Fpz-Cz EEG\n($\\mu$V, offset)")
    panel(ax, "A", "Example 30-s Sleep-EDF Fpz-Cz epochs")

    ax = fig.add_subplot(gs[1, 0])
    lo, hi = -1.9, 2.5
    ax.axhspan(0, hi, color=N1C, alpha=.07)
    ax.axhspan(lo, 0, color=REMC, alpha=.07)
    for k, (_, c, _) in ex.items():
        ax.plot(range(1, 30), s[k]["ta"], "-o", ms=3.2, lw=1.1, color=c,
                label="%s (%d switches)" % (k, s[k]["sw"]))
        ax.plot(s[k]["swi"] + 1, s[k]["ta"][s[k]["swi"]], "o", ms=6.5, mfc="white", mec=c, mew=1.3)
    ax.axhline(0, ls="--", lw=1, color="0.4")
    ax.text(1.0, hi * .88, "theta dominant", fontsize=9, color="0.35")
    ax.text(1.0, lo * .86, "alpha dominant", fontsize=9, color="0.35", va="bottom")
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Window index")
    ax.set_ylabel(r"$r_{\theta\alpha}(k)$")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=.9)
    panel(ax, "B", "Theta-alpha dominance and switching")

    ax = fig.add_subplot(gs[1, 1])
    for k, (_, c, _) in ex.items():
        ax.plot(range(1, 30), s[k]["sb"], "-o", ms=3.2, lw=1.1, color=c,
                label="%s: SD=%.2f" % (k, s[k]["sd"]))
    for w in range(3, 29, 3):
        ax.annotate("", xy=(w, s["N1"]["sb"][w - 1]), xytext=(w, s["REM"]["sb"][w - 1]),
                    arrowprops=dict(arrowstyle="<->", lw=.9, color="0.6"))
    ax.set_xlabel("Window index")
    ax.set_ylabel(r"$r_{\sigma\beta}(k)$")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=.9)
    panel(ax, "C", "Band-ratio volatility")

    ax = fig.add_subplot(gs[2, 0])
    for k, (_, c, _) in ex.items():
        ax.plot(s[k]["sb"], s[k]["ta"], "-o", ms=2.8, lw=.9, color=c, alpha=.9,
                label="%s: $L$=%.2f" % (k, s[k]["L"]))
        ax.plot(s[k]["sb"].mean(), s[k]["ta"].mean(), "s", ms=7.5, color=c, mec="white", mew=1.1)
    ax.set_xlabel(r"$r_{\sigma\beta}(k)$")
    ax.set_ylabel(r"$r_{\theta\alpha}(k)$")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=.9)
    panel(ax, "D", "Path length $L$")

    ax = fig.add_subplot(gs[2, 1])
    for k, (_, c, _) in ex.items():
        ax.plot(range(2, 29), s[k]["turn"], "-o", ms=3.2, lw=1.1, color=c,
                label="%s: $T$=%.2f" % (k, s[k]["T"]))
        ax.axhline(s[k]["T"], ls="--", lw=1.3, color=c, alpha=.85)
    ax.set_ylim(-.08, 2.62)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Turn contribution")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=.9, columnspacing=1.4)
    panel(ax, "E", "Turning $T$")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "fig_real_epoch_feature_examples"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=.05)
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    print("wrote %s.pdf and .png" % out.name)
    print("  N1  L=%.2f T=%.2f switches=%d" % (s["N1"]["L"], s["N1"]["T"], s["N1"]["sw"]))
    print("  REM L=%.2f T=%.2f switches=%d" % (s["REM"]["L"], s["REM"]["T"], s["REM"]["sw"]))


# ----------------------------------------------------------------- Fig. 1, pipeline

GREY_PANEL, GREY_EDGE, WHITE_BOX = "#E9E9E9", "#6E6E6E", "#FFFFFF"


def _panel(ax, x, y, w, h, title):
    """A numbered stage box."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=GREY_PANEL, edgecolor=GREY_EDGE, lw=1.0, zorder=1))
    ax.text(x + 0.008, y + h + 0.022, title, fontsize=11, fontweight="bold", va="bottom", ha="left")


def _box(ax, x, y, w, h, lines, fontsize=8.2):
    """A sub-box with one or more lines of text."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.010",
                                facecolor=WHITE_BOX, edgecolor=GREY_EDGE, lw=0.9, zorder=2))
    if isinstance(lines, str):
        lines = [lines]
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - (i + 0.75) * h / (len(lines) + 0.5), t, fontsize=fontsize,
                ha="center", va="center", zorder=3,
                fontweight="bold" if i == 0 and len(lines) > 1 else "normal")


def _arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color="#3A3A3A", lw=1.4,
                                connectionstyle="arc3,rad=%.2f" % rad,
                                shrinkA=2, shrinkB=2))


def _inset(fig, ax, x, y, w, h):
    """Small inset axes, positioned in figure coordinates."""
    a = fig.add_axes([x, y, w, h], zorder=5)
    for side in a.spines.values():
        side.set_linewidth(0.6)
        side.set_color("#8A8A8A")
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_facecolor("white")
    return a


def _title(ax, x, y, w, h, text, fontsize=8.4):
    """Caption at the top of a sub-box, leaving room for an inset."""
    ax.text(x + w / 2, y + h - 0.030, text, fontsize=fontsize, fontweight="bold",
            ha="center", va="center", zorder=3)


def figure_pipeline():
    """Pipeline schematic."""
    folder = D.SLEEP_EDF_DIR / D.subset_folder("cassette")
    if not folder.is_dir():
        folder = next(p for p in D.SLEEP_EDF_DIR.rglob(D.subset_folder("cassette")) if p.is_dir())
    raw = mne.io.read_raw_edf(folder / (FIG3_RECORD + "-PSG.edf"), preload=False, verbose="ERROR")
    ch = [c for c in raw.ch_names if "Fpz" in c][0]
    fs = int(raw.info["sfreq"]); n = 30 * fs
    x = F.clean_epoch(raw.get_data(picks=[ch], start=FIG3_N1_EPOCH * n,
                                   stop=(FIG3_N1_EPOCH + 1) * n)[0], fs)
    bands = F.subwindow_spectra(x, fs)[0]

    plt.rcParams.update({"font.family": "sans-serif"})
    fig = plt.figure(figsize=(12.0, 5.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- 1. Data ----------------------------------------------------------
    _panel(ax, 0.015, 0.40, 0.150, 0.48, "1. Data")
    _box(ax, 0.027, 0.715, 0.126, 0.145, ["Sleep-EDF Expanded", "cassette 153, telemetry 44"], 8.0)
    _box(ax, 0.027, 0.565, 0.126, 0.135, ["UCDDB", "25 clinical records"], 8.0)
    _box(ax, 0.027, 0.425, 0.126, 0.125, ["one EEG channel", "Fpz-Cz or C3-A2"], 8.0)

    _box(ax, 0.176, 0.42, 0.030, 0.46, "")
    ax.text(0.191, 0.65, "30-s epochs", fontsize=8.6, fontweight="bold",
            rotation=90, ha="center", va="center", zorder=3)
    _arrow(ax, 0.165, 0.65, 0.176, 0.65)

    # ---- 2. Epoch processing ---------------------------------------------
    _panel(ax, 0.228, 0.40, 0.250, 0.48, "2. Epoch processing")
    _box(ax, 0.240, 0.795, 0.226, 0.062, "detrend, median centre, 0.3-45 Hz band-pass", 8.0)
    _box(ax, 0.240, 0.575, 0.109, 0.205, "")
    _title(ax, 0.240, 0.575, 0.109, 0.205, "2-s window, 1-s stride", 7.9)
    ax.text(0.2945, 0.712, "K = 29", fontsize=7.8, ha="center", va="center", zorder=3)
    _box(ax, 0.357, 0.575, 0.109, 0.205, "")
    _title(ax, 0.357, 0.575, 0.109, 0.205, "Welch PSD")
    ax.text(0.4115, 0.712, "5 relative band powers", fontsize=7.4, ha="center", va="center", zorder=3)
    _box(ax, 0.240, 0.425, 0.226, 0.130, "")
    _title(ax, 0.240, 0.425, 0.226, 0.130, "spectral trajectory over the epoch", 8.2)
    _arrow(ax, 0.206, 0.65, 0.228, 0.65)

    # ---- 3. Trajectory descriptors ---------------------------------------
    _panel(ax, 0.541, 0.40, 0.215, 0.48, "3. Trajectory descriptors")
    _box(ax, 0.553, 0.795, 0.191, 0.062, "band-ratio trajectories  (32)", 8.2)
    _box(ax, 0.553, 0.590, 0.092, 0.190, "")
    _title(ax, 0.553, 0.590, 0.092, 0.190, "path shape  (5)")
    _box(ax, 0.652, 0.590, 0.092, 0.190,
         ["band powers (20)", "window level (15)", "composite (1)"], 7.8)
    _box(ax, 0.553, 0.425, 0.191, 0.140, ["DYN", "73 interpretable features", ""], 8.8)
    _arrow(ax, 0.478, 0.65, 0.541, 0.65)

    # ---- 4. Model and evaluation -----------------------------------------
    _panel(ax, 0.786, 0.40, 0.199, 0.48, "4. Model and evaluation")
    _box(ax, 0.798, 0.760, 0.175, 0.097, ["feature sets", "BP 5, HC 16, DYN 73, HC+DYN 89"], 7.6)
    _box(ax, 0.798, 0.620, 0.175, 0.125, ["logistic regression", "class-balanced",
                                          "fit on training folds only"], 7.6)
    _box(ax, 0.798, 0.425, 0.175, 0.180, ["subject-grouped 5-fold CV", "pooled out-of-fold AUC",
                                          "2000-resample", "subject bootstrap"], 7.6)
    _arrow(ax, 0.756, 0.65, 0.786, 0.65)

    # ---- 5. Controls, on the same folds ----------------------------------
    _panel(ax, 0.228, 0.045, 0.528, 0.215, "5. Controls, on the same folds")
    _box(ax, 0.240, 0.070, 0.163, 0.150, ["size-matched", "shuffled features", "Gaussian noise"], 7.8)
    _box(ax, 0.413, 0.070, 0.163, 0.150, ["within-epoch", "window-order permutation",
                                          "order-free vs order-dependent"], 7.2)
    _box(ax, 0.586, 0.070, 0.158, 0.150, ["design", "2-s vs 4-s window", "Pz-Oz, interior epochs"], 7.6)
    _arrow(ax, 0.886, 0.40, 0.886, 0.152)
    _arrow(ax, 0.886, 0.152, 0.758, 0.152)

    # ---- real-data insets -------------------------------------------------
    a = _inset(fig, ax, 0.248, 0.588, 0.093, 0.098)
    a.plot(np.arange(len(x)) / fs, x, lw=0.35, color="#2A2A2A")
    for st in (2, 5, 8):
        a.axvspan(st, st + 2, color="#B4B4B4", alpha=0.8, lw=0)
    a.set_xlim(0, 30)

    a = _inset(fig, ax, 0.365, 0.588, 0.093, 0.098)
    for j, style in enumerate(["-", "-", "--", ":", "-."]):
        a.plot(bands[:, j], style, lw=0.8,
               color=["#1A1A1A", "#7A7A7A", "#1A1A1A", "#1A1A1A", "#7A7A7A"][j])
    a.set_xlim(0, 28)

    a = _inset(fig, ax, 0.561, 0.602, 0.076, 0.095)
    a.plot(bands[:, 1], bands[:, 2], "-o", ms=1.5, lw=0.7, color="#2A2A2A")

    a = _inset(fig, ax, 0.252, 0.437, 0.202, 0.056)
    a.imshow(bands.T, aspect="auto", cmap="Greys", interpolation="nearest")
    a.set_yticks(range(5))
    a.set_yticklabels([r"$\delta$", r"$\theta$", r"$\alpha$", r"$\sigma$", r"$\beta$"], fontsize=5.5)
    a.tick_params(length=0, pad=1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "fig_pipeline"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote %s.pdf and .png" % out.name)


if __name__ == "__main__":
    figure_pipeline()
    figure3()
    figure2()
