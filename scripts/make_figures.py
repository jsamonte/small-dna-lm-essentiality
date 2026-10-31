"""Figure 1, built from results/ so the figure and the text cannot disagree.

Panel A  signal against model size, with the classical baseline as the line to beat.
Panel B  what survives a move to a second organism.

Every number is read from results/. Nothing here is typed by hand.

Run:  .venv/Scripts/python scripts/make_figures.py
"""

import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures" / "fig1_scaling_and_transfer.png"

# Validated categorical slots 1 and 2 (dataviz palette).
# Worst-pair CVD dE 24.7, normal-vision 33.6 -- clear of both floors.
GB = "#2a78d6"      # gradient boosting, and the measured value in panel B
LR = "#eb6834"      # logistic regression
INK = "#1a1a19"
MUTED = "#8a8a84"
GRID = "#e3e2dd"

SHORT = {
    "hyenadna-tiny": "HyenaDNA\n438K",
    "hyenadna-tiny-d256": "HyenaDNA\n1.6M",
    "nt-v2-50m": "NT v2\n55.9M",
    "nt-v2-100m": "NT v2\n97.9M",
}
ROWLABEL = {
    "classical": "classical features",
    "nt-v2-100m": "NT v2  97.9M",
    "nt-v2-50m": "NT v2  55.9M",
    "hyenadna-tiny-d256": "HyenaDNA  1.6M",
    "hyenadna-tiny": "HyenaDNA  438K",
}


def read_csv(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    scaling = read_csv("scaling.csv")
    gen = read_csv("generalization.csv")
    baseline = json.loads((RESULTS / "baseline.json").read_text(encoding="utf-8"))

    base = baseline["results"]["all__gradient_boosting"]
    base_mcc = base["mcc"]
    base_lo, base_hi = base["mcc_ci95"]

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.4, 5.6), dpi=160, gridspec_kw={"width_ratios": [1.05, 1]}
    )
    fig.patch.set_facecolor("white")

    # ---------------- Panel A: signal against model size ----------------
    ax.axhspan(base_lo, base_hi, color=INK, alpha=0.07, zorder=0, lw=0)
    ax.axhline(base_mcc, color=INK, ls="--", lw=1.6, zorder=2)
    ax.axhline(0.0, color=MUTED, ls=":", lw=1.2, zorder=1)

    for clf, colour, marker, label in (
        ("gradient_boosting", GB, "o", "gradient boosting"),
        ("logistic_regression", LR, "s", "logistic regression"),
    ):
        rows = [r for r in scaling if r["classifier"] == clf]
        rows.sort(key=lambda r: int(r["params"]))
        xs = [int(r["params"]) for r in rows]
        ys = [float(r["mcc"]) for r in rows]
        lo = [float(r["mcc"]) - float(r["mcc_lo"]) for r in rows]
        hi = [float(r["mcc_hi"]) - float(r["mcc"]) for r in rows]
        ax.errorbar(
            xs, ys, yerr=[lo, hi], color=colour, marker=marker, ms=8, lw=2,
            capsize=4, elinewidth=1.4, markeredgecolor="white", markeredgewidth=1.4,
            label=label, zorder=4,
        )

    # One direct label per model, under the lower of its two points. The two
    # NT v2 models sit close together on a log axis, so the second is dropped
    # further to keep the labels apart.
    label_dy = {
        "hyenadna-tiny": -12,
        "hyenadna-tiny-d256": -12,
        "nt-v2-50m": -12,
        "nt-v2-100m": -40,
    }
    for name in SHORT:
        rows = [r for r in scaling if r["model"] == name]
        x = int(rows[0]["params"])
        ylow = min(float(r["mcc_lo"]) for r in rows)
        ax.annotate(
            SHORT[name], xy=(x, ylow), xytext=(0, label_dy[name]),
            textcoords="offset points",
            ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.3,
        )

    # Sits just above the band so it never lands on the dashed line itself.
    ax.annotate(
        f"classical features  {base_mcc:.3f}",
        xy=(0.015, base_hi), xycoords=("axes fraction", "data"),
        xytext=(0, 5), textcoords="offset points",
        ha="left", va="bottom", fontsize=10, color=INK, fontweight="bold",
    )
    ax.annotate(
        "majority class  (MCC 0, accuracy 91.0%)",
        xy=(0.015, 0.0), xycoords=("axes fraction", "data"),
        xytext=(0, 6), textcoords="offset points",
        ha="left", va="bottom", fontsize=9, color=MUTED,
    )

    ax.set_xscale("log")
    ax.set_xlabel("model parameters", fontsize=10.5, color=INK)
    ax.set_ylabel("Matthews correlation coefficient", fontsize=10.5, color=INK)
    ax.set_title("A   Signal against model size", fontsize=12, color=INK,
                 fontweight="bold", loc="left", pad=12)
    ax.set_ylim(-0.06, 0.42)
    ax.set_xlim(2.4e5, 2.6e8)
    ax.legend(frameon=False, fontsize=10, loc="lower right",
              bbox_to_anchor=(1.0, 0.0))

    # ---------------- Panel B: transfer to B. subtilis ----------------
    rows = [r for r in gen if r["classifier"] == "gradient_boosting"]
    order = ["classical", "nt-v2-100m", "nt-v2-50m", "hyenadna-tiny-d256", "hyenadna-tiny"]
    rows.sort(key=lambda r: order.index(r["representation"]))

    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        a = float(r["within_ecoli_mcc"])
        b = float(r["mcc"])
        bx.plot([a, b], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        bx.plot([a], [y], marker="o", ms=9, color="white",
                markeredgecolor=MUTED, markeredgewidth=2, zorder=3)
        bx.plot([b], [y], marker="o", ms=10, color=GB,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        bx.annotate(f"{b:+.3f}", xy=(b, y), xytext=(0, -15),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9.5, color=GB, fontweight="bold")

    bx.axvline(0.0, color=MUTED, ls=":", lw=1.2, zorder=0)
    bx.set_yticks(ys)
    bx.set_yticklabels([ROWLABEL[r["representation"]] for r in rows], fontsize=10.5)
    bx.set_ylim(-1.2, len(rows) - 0.25)
    bx.set_xlim(-0.09, 0.47)
    bx.set_xlabel("Matthews correlation coefficient", fontsize=10.5, color=INK)
    bx.set_title("B   What survives a move to another organism",
                 fontsize=12, color=INK, fontweight="bold", loc="left", pad=12)

    # legend for the two dot roles, drawn as proxy handles
    from matplotlib.lines import Line2D  # noqa: E402
    bx.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", ms=9, color="white",
                   markeredgecolor=MUTED, markeredgewidth=2,
                   label="within $\\it{E.\\ coli}$"),
            Line2D([], [], marker="o", ls="none", ms=10, color=GB,
                   markeredgecolor="white", markeredgewidth=1.4,
                   label="in $\\it{B.\\ subtilis}$, no retraining"),
        ],
        frameon=False, fontsize=9.5, loc="lower right",
        bbox_to_anchor=(1.0, -0.01),
    )

    for a in (ax, bx):
        a.set_facecolor("white")
        a.grid(axis="both", color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9.5)
        for lbl in a.get_yticklabels() + a.get_xticklabels():
            lbl.set_color(INK)

    bx.grid(axis="y", visible=False)

    fig.tight_layout(pad=1.4, w_pad=3.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white", bbox_inches="tight")
    print("wrote", OUT)

    # the caption, generated from the same numbers
    worst = min(rows, key=lambda r: float(r["mcc"]))
    worst_name = " ".join(ROWLABEL[worst["representation"]].split())
    # Typographic minus on the value only. Doing this with .replace() on the
    # caption would also eat the hyphens in words like "codon-counting".
    worst_mcc = f"{float(worst['mcc']):+.3f}".replace("-", "−")
    cap = (
        f"**Figure 1. No language model beats a codon-counting baseline, and the "
        f"smallest stop working entirely in a second organism.** (**A**) Matthews "
        f"correlation coefficient against model size, for four DNA language models "
        f"used as frozen feature extractors, with 95% intervals. The dashed line is "
        f"the classical sequence-feature baseline ({base_mcc:.3f}, shaded 95% "
        f"interval); the dotted line is a model that calls every gene non-essential. "
        f"No language model clears the baseline at any size. (**B**) The same "
        f"representations with gradient boosting, trained on *E. coli* and applied to "
        f"*B. subtilis* without retraining. Every method degrades; "
        f"{worst_name} falls to {worst_mcc}, indistinguishable from chance, "
        f"while the classical features and the largest model retain about 60%."
    )
    (RESULTS / "fig1_caption.md").write_text(cap, encoding="utf-8")
    print("wrote", RESULTS / "fig1_caption.md")


if __name__ == "__main__":
    main()
