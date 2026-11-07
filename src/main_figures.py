"""Main-study figures: results/fig_main_reference_arms.png and results/fig_main_imbalance.png."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from figures import AQUA, BLUE, GRID, INK, MUTED, ORANGE, style
from common import RESULTS

S = json.load(open(RESULTS / "main_summary.json"))
ACC = pd.read_csv(RESULTS / "main_accuracy.csv")


def arms():
    a = ACC[(ACC.analysis == "R2") & (ACC["size"] == "all")].set_index("arm").accuracy * 100
    rows = [("none", "No statistics\n(single essay)", MUTED), ("ref_human", "Human text\n(Wikipedia)", ORANGE),
            ("ref_n2", "Outside models\n(Llama, Mistral, Qwen)", ORANGE), ("ref_original", "Same-date\ncandidates", BLUE),
            ("ref_n1", "Candidates,\n2 months later", BLUE), ("batch", "Full test batch\n(original method)", AQUA)]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, (k, lab, c) in enumerate(rows):
        ax.bar(i, a[k], width=0.6, color=c, zorder=2)
        ax.text(i, a[k] + 1, f"{a[k]:.1f}%", ha="center", color=INK, fontsize=9.5)
    ax.axhline(20, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(5.45, 20.8, "chance 20%", ha="right", color=MUTED, fontsize=9)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 55)
    ax.set_ylabel("Cross-language accuracy (%)")
    ax.set_title("Which reference set fixes single-essay attribution? (R2, 42 pairs × 50 splits)", color=INK, fontsize=10.5, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_main_reference_arms.png", dpi=200)


def imbalance():
    d = pd.read_csv(RESULTS / "main_imbalance.csv")
    g = d.groupby(["share", "arm"])[["accuracy", "balanced_accuracy"]].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for col, arm, c, lab in (("accuracy", "batch", BLUE, "Batch statistics, plain accuracy"),
                             ("balanced_accuracy", "batch", BLUE, "Batch statistics, balanced accuracy"),
                             ("accuracy", "ref_original", ORANGE, "Reference set, plain accuracy")):
        ls = "-" if col == "accuracy" else (0, (4, 3))
        y = g[(col, arm)]
        ax.plot(y.index, y.values, color=c, linewidth=2, linestyle=ls, marker="o", markersize=6, label=lab)
    ax.set_xticks([20, 40, 60, 80], ["20%\n(balanced)", "40%", "60%", "80%"])
    ax.set_xlabel("Share of the test batch from one model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(20, 55)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Lopsided batches hurt batch statistics, not a reference set (R3)", color=INK, fontsize=10.5, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_main_imbalance.png", dpi=200)


if __name__ == "__main__":
    arms()
    imbalance()
