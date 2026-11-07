"""Pilot figures: results/fig_reference_set.png and results/fig_mix_sweep.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def reference_set():
    b = pd.read_csv(RESULTS / "pilot_B.csv")
    per_split = b.groupby(["split", "arm"]).accuracy.mean().unstack()
    arms = [("none", "No test-language\nstatistics"), ("reference", "Reference set\n(other prompts)"),
            ("batch", "Full test batch\n(original method)")]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (arm, label) in enumerate(arms):
        m, sd = per_split[arm].mean() * 100, per_split[arm].std() * 100
        ax.bar(i, m, width=0.55, color=BLUE, zorder=2)
        ax.errorbar(i, m, yerr=sd, color=INK, capsize=4, linewidth=1, zorder=3)
        ax.text(i, m + sd + 1.2, f"{m:.1f}%", ha="center", color=INK, fontsize=10)
    ax.axhline(20, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(2.45, 20.8, "chance 20%", ha="right", color=MUTED, fontsize=9)
    ax.set_xticks(range(3), [l for _, l in arms])
    ax.set_ylim(0, 60)
    ax.set_ylabel("Cross-language accuracy (%)")
    ax.set_title("Prompt-disjoint transfer, 42 language pairs × 50 splits", color=INK, fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_reference_set.png", dpi=200)


def mix_sweep():
    a = pd.read_csv(RESULTS / "pilot_A.csv")
    m = a[a.sweep == "mix"]
    series = [
        ("batch", "accuracy", "Batch statistics: accuracy", BLUE),
        ("batch", "balanced_accuracy", "Batch statistics: balanced accuracy", ORANGE),
        ("full", "accuracy", "Full-language statistics (original)", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for arm, metric, label, color in series:
        s = m[m.arm == arm].groupby("level")[metric].mean() * 100
        ax.plot(s.index * 100, s.values, color=color, linewidth=2, marker="o", markersize=6, label=label, zorder=3)
    ax.axhline(20, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(80, 21, "chance 20%", ha="right", color=MUTED, fontsize=9)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xlabel("Share of the 20-essay test batch from one model (%)")
    ax.set_ylabel("Cross-language score (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Lopsided test batches, same-prompt protocol", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_mix_sweep.png", dpi=200)


def reference_size():
    s = pd.read_csv(RESULTS / "followup_sizes.csv")
    base = pd.read_csv(RESULTS / "followup_length.csv")[["none", "batch"]].mean() * 100
    curve = s.groupby("k").arm_acc.mean() * 100
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(curve.index, curve.values, color=BLUE, linewidth=2, marker="o", markersize=6, zorder=3)
    for k, v in curve.items():
        ax.text(k, v - 3.2, f"{v:.1f}", ha="center", color=INK, fontsize=9)
    for y, label in ((base["batch"], f"Full test batch {base['batch']:.1f}%"),
                     (base["none"], f"No test-language statistics {base['none']:.1f}%"), (20, "chance 20%")):
        ax.axhline(y, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(61, y + 0.8, label, ha="right", color=MUTED, fontsize=9)
    ax.set_xticks([5, 10, 20, 40, 60])
    ax.set_xlabel("Reference essays per language (balanced across the 5 models)")
    ax.set_ylabel("Cross-language accuracy (%)")
    ax.set_ylim(0, 60)
    ax.set_title("How many reference essays are enough?", color=INK, fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_reference_size.png", dpi=200)


if __name__ == "__main__":
    reference_set()
    mix_sweep()
    reference_size()
