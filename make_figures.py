#!/usr/bin/env python3
"""
Generates the two figures used by sparsecomm_acl.tex:

    figures/cost_accuracy.pdf   -> Figure 2 (replaces the old auxiliary table)
    figures/scale_forest.pdf    -> Figure 3 (replaces factorial_scale_twinx.png
                                             and the per-scale coefficient table)

Every number below is copied from the manuscript's own reported values, or is
exact arithmetic on them (cell means from the saturated fit, t-intervals from
the reported CR1 standard errors with G-1 degrees of freedom).

    pip install matplotlib scipy
    python make_figures.py
"""
import os
import math
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito colour-blind-safe palette
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
BLUE, VERM, PURPLE, GREY = "#0072B2", "#D55E00", "#CC79A7", "#7F7F7F"


# =============================================================================
# Figure 2 : cost vs accuracy
# =============================================================================
# Protocol P1: (accuracy, sd) for NoComm / MAD / SparseComm, and comm tokens
P1 = {
    "GSM8K":    dict(nocomm=(87.00, 0.67), mad=(96.22, 0.19), ours=(95.00, 0.88),
                     mad_tok=530.8,  our_tok=52.2,  colour=BLUE,   marker="o"),
    "AQuA-RAT": dict(nocomm=(79.52, 5.46), mad=(85.34, 2.33), ours=(86.28, 0.31),
                     mad_tok=850.3,  our_tok=421.6, colour=GREEN,  marker="s"),
    "HotpotQA": dict(nocomm=(30.00, 2.33), mad=(34.00, 3.48), ours=(31.67, 4.26),
                     mad_tok=2914.1, our_tok=473.2, colour=VERM,   marker="^"),
}

# Protocol P3: fixed motifs on 150 GSM8K questions
P3 = [  # (name, accuracy %, comm tokens, is_learned)
    ("Fully connected", 98.67, 304.8, False),
    ("Small world",     96.00, 292.9, False),
    ("Ring",            95.33, 337.0, False),
    ("Learned (ours)",  95.33,  51.1, True),
    ("Star",            94.67, 203.9, False),
    ("Random",          94.67, 191.4, False),
    ("Scale-free",      92.67, 192.6, False),
]
N_P3 = 150

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 2.5))

# ---- panel (a) -------------------------------------------------------------
for name, d in P1.items():
    xs = [0.0, d["our_tok"], d["mad_tok"]]
    ys = [d["nocomm"][0], d["ours"][0], d["mad"][0]]
    es = [d["nocomm"][1], d["ours"][1], d["mad"][1]]
    axL.plot(xs, ys, color=d["colour"], lw=0.8, alpha=0.45, zorder=1)
    axL.errorbar(xs, ys, yerr=es, fmt="none", ecolor=d["colour"],
                 elinewidth=0.8, capsize=2, alpha=0.8, zorder=2)
    axL.scatter(xs[0], ys[0], s=26, facecolors="white", edgecolors=d["colour"],
                marker=d["marker"], linewidths=1.0, zorder=3)
    axL.scatter(xs[1], ys[1], s=42, color=d["colour"], marker=d["marker"],
                zorder=4, label=name)
    axL.scatter(xs[2], ys[2], s=26, facecolors="none", edgecolors=d["colour"],
                marker=d["marker"], linewidths=1.0, zorder=3)
    axL.annotate("", xy=(xs[1], ys[1]), xytext=(xs[2], ys[2]),
                 arrowprops=dict(arrowstyle="-|>", color=d["colour"],
                                 lw=0.9, alpha=0.75, shrinkA=3, shrinkB=4))

axL.set_xscale("symlog", linthresh=30)
axL.set_xlabel("transmitted inter-agent tokens per question")
axL.set_ylabel("accuracy (\\%)" if plt.rcParams["text.usetex"] else "accuracy (%)")
axL.set_ylim(22, 104)
axL.set_xlim(-3, 5200)
axL.set_xticks([0, 50, 200, 1000, 3000])
axL.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axL.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
axL.set_title("(a) three benchmarks (P1)", loc="left")
axL.legend(frameon=False, loc="center left", bbox_to_anchor=(0.30, 0.55), handletextpad=0.3)
axL.text(0.015, 0.035, "at 0 tokens: NoComm    hollow: MAD    filled: ours",
         transform=axL.transAxes, fontsize=6.3, color=GREY)

# ---- panel (b) -------------------------------------------------------------
for name, acc, tok, learned in P3:
    p = acc / 100.0
    ci = 1.96 * math.sqrt(p * (1 - p) / N_P3) * 100
    col = VERM if learned else GREY
    axR.errorbar(tok, acc, yerr=ci, fmt="none", ecolor=col,
                 elinewidth=0.8, capsize=2, alpha=0.9)
    axR.scatter(tok, acc, s=46 if learned else 28,
                color=col if learned else "white",
                edgecolors=col, linewidths=1.0, marker="D" if learned else "o",
                zorder=3)
    off = {"Fully connected": (-6, 8, "right"), "Small world": (0, 9, "center"),
           "Ring": (7, -2, "left"), "Learned (ours)": (0, 11, "center"),
           "Star": (9, 4, "left"), "Random": (-7, 2, "right"),
           "Scale-free": (0, -14, "center")}[name]
    axR.annotate(name, (tok, acc), textcoords="offset points",
                 xytext=(off[0], off[1]), ha=off[2],
                 fontsize=6.6, color=col)

axR.set_xlabel("transmitted inter-agent tokens per question")
axR.set_ylabel("accuracy (%)")
axR.set_xlim(-5, 430)
axR.set_ylim(88.5, 103)
axR.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
axR.set_title("(b) GSM8K fixed motifs (P3)", loc="left")
axR.text(0.98, 0.06, r"fixed motifs: $\rho_s=0.73$, $p=0.10$, $n=6$",
         transform=axR.transAxes, ha="right", fontsize=6.8, color=GREY)

fig.tight_layout(w_pad=1.6)
fig.savefig("figures/cost_accuracy.pdf")
plt.close(fig)
print("wrote figures/cost_accuracy.pdf")


# =============================================================================
# Figure 3 : training scale + factorial forest
# =============================================================================
# Cell means implied exactly by the saturated fit at each scale:
#   Arm00 = b0, Arm10 = b0+bC, Arm01 = b0+bT, Arm11 = b0+bC+bT+bCT
COEF = {  # n_train : (b0, bC, bT, bCT, SE_C, SE_T, SE_CT, n_questions)
    100: (0.5800,  0.0500,  0.0800, -0.0600, 0.0559, 0.0583, 0.0792, 100),
    200: (0.9800, -0.0800, -0.0600,  0.1400, 0.0485, 0.0447, 0.0644,  50),
    300: (0.9600, -0.0400, -0.0200,  0.0200, 0.0495, 0.0454, 0.0788,  50),
    400: (0.9200,  0.0400,  0.0400, -0.1000, 0.0495, 0.0403, 0.0594,  50),
}

# Arm-11 transmitted tokens per checkpoint.
ARM11_TOKENS = {100: 496.7, 200: 124.6, 300: 123.3, 400: 58.9}

scales = sorted(COEF)
arm = {"Arm 00  ring, full": [], "Arm 10  ring, compressed": [],
       "Arm 01  learned, full": [], "Arm 11  full SparseComm": []}
for s in scales:
    b0, bC, bT, bCT, *_ = COEF[s]
    arm["Arm 00  ring, full"].append(100 * b0)
    arm["Arm 10  ring, compressed"].append(100 * (b0 + bC))
    arm["Arm 01  learned, full"].append(100 * (b0 + bT))
    arm["Arm 11  full SparseComm"].append(100 * (b0 + bC + bT + bCT))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 2.7),
                               gridspec_kw={"width_ratios": [1.0, 1.25]})

# ---- panel (a) -------------------------------------------------------------
styles = [(GREY, "o", "--"), (SKY, "^", "-."), (GREEN, "s", ":"), (VERM, "D", "-")]
for (label, ys), (col, mk, ls) in zip(arm.items(), styles):
    axA.plot(scales, ys, ls=ls, marker=mk, ms=4, color=col, label=label,
             lw=1.5 if col == VERM else 1.0)
axA.set_xlabel("controller-training examples")
axA.set_ylabel("matched-question accuracy (%)")
axA.set_xticks(scales)
axA.set_ylim(52, 104)
axA.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
axA.legend(frameon=False, loc="lower right", handlelength=2.2)

axA2 = axA.twinx()
axA2.spines["right"].set_visible(True)
tk = [(s, v) for s, v in ARM11_TOKENS.items() if v is not None]
if tk:
    axA2.plot([s for s, _ in tk], [v for _, v in tk], ls="--", marker="x",
              ms=5, color=BLUE, lw=1.0)
    for s, v in tk:
        axA2.annotate(f"{v:.1f}", (s, v), textcoords="offset points",
                      xytext=(4, 4), fontsize=6.5, color=BLUE)
axA2.set_ylabel("arm-11 comm. tokens", color=BLUE)
axA2.tick_params(axis="y", colors=BLUE)
axA2.set_ylim(0, 550)
if len(tk) < len(ARM11_TOKENS):
    axA2.text(0.985, 0.985, "token series incomplete in archive", ha="right",
              va="top", transform=axA2.transAxes, fontsize=6.2, color=BLUE,
              alpha=0.85)
axA.set_title("(a) accuracy by arm and training scale", loc="left")

# ---- panel (b) : forest ----------------------------------------------------
terms = [("Interaction", 6, PURPLE), ("Topology", 5, GREEN), ("Compression", 4, ORANGE)]
mdes, ypos, ylabels = [], [], []
row = 0
for s in reversed(scales):
    b0, bC, bT, bCT, seC, seT, seCT, G = COEF[s]
    df = G - 1
    tcrit = stats.t.ppf(0.975, df)
    tpow = stats.t.ppf(0.80, df)
    for name, (b, se, col) in zip(["Interaction", "Topology", "Compression"],
                                  [(bCT, seCT, PURPLE), (bT, seT, GREEN),
                                   (bC, seC, ORANGE)]):
        lo, hi = b - tcrit * se, b + tcrit * se
        mdes.append((tcrit + tpow) * se)
        axB.plot([lo, hi], [row, row], color=col, lw=1.4, solid_capstyle="butt")
        axB.plot([lo, lo], [row - .16, row + .16], color=col, lw=1.0)
        axB.plot([hi, hi], [row - .16, row + .16], color=col, lw=1.0)
        axB.scatter([b], [row], s=22, color=col, zorder=4)
        ylabels.append(f"{name}")
        ypos.append(row)
        row += 1
    axB.axhline(row - 0.5, color="0.85", lw=0.6)
    axB.text(-0.315, row - 2, f"n={s}", fontsize=7, va="center", ha="left",
             color="0.25", rotation=0)
    row += 0.6

band = statistics.median(mdes)
axB.axvspan(-band, band, color="0.5", alpha=0.11, lw=0, zorder=0)
axB.axvline(0, color=BLACK, lw=0.8, zorder=1)
axB.set_yticks(ypos)
axB.set_yticklabels(ylabels)
axB.set_xlabel("estimated change in accuracy (proportion)")
axB.set_xlim(-0.33, 0.33)
axB.set_ylim(-0.8, row - 0.4)
axB.tick_params(axis="y", length=0)
axB.set_title("(b) factorial estimates, 95% question-clustered CI", loc="left", pad=12)
axB.text(0.5, 1.005, f"shaded band: median 80%-power detection limit "
         f"($\\pm${band:.2f})", transform=axB.transAxes, ha="center",
         va="bottom", fontsize=6.6, color="0.35")

fig.tight_layout(w_pad=2.0)
fig.savefig("figures/scale_forest.pdf")
plt.close(fig)
print("wrote figures/scale_forest.pdf")
