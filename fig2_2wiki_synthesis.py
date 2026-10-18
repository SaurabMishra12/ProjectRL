#!/usr/bin/env python3
"""
Publication-quality figure generator for WWW 2027.

Paper:
"Who Gets the Last Word? Disentangling Topology, Readout Duality,
and Volume Attribution in Multi-Agent Web Question Answering"

Design goals:
- Minimal, clean, research-oriented visual style.
- Landscape layout suitable for a two-column paper.
- Consistent typography and color palette.
- Carefully positioned annotations to minimize overlap.
- Audited experimental numbers and labels preserved without N-value labels.
- Vector PDF and high-resolution PNG output.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# =============================================================================
# GLOBAL STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5,

    "axes.titlesize": 8.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 7.5,
    "axes.labelweight": "medium",

    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.3,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,

    "axes.edgecolor": "#C8CED6",
    "axes.linewidth": 0.65,
    "axes.labelcolor": "#303846",
    "xtick.color": "#505967",
    "ytick.color": "#303846",

    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E5E9EF",
    "grid.alpha": 1.0,
    "grid.linestyle": "-",
    "grid.linewidth": 0.55,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Save alongside this script, with a fallback for notebooks
try:
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUT_DIR = os.getcwd()

# Restrained, colorblind-conscious research palette.
C_NAVY = "#315A7D"
C_TEAL = "#238A87"
C_ORANGE = "#D88945"
C_PURPLE = "#8174A8"
C_GREY = "#89939F"
C_LIGHTGREY = "#DCE2E8"
C_RED = "#BD6868"
C_GREEN = "#4D896B"

C_TEXT = "#303846"
C_GRID = "#E5E9EF"
C_WHITE = "#FFFFFF"


# =============================================================================
# SHARED HELPERS
# =============================================================================

def style_axis(ax, grid_axis="y"):
    """Apply consistent minimal styling to an axis."""
    ax.set_axisbelow(True)
    ax.grid(
        True,
        axis=grid_axis,
        color=C_GRID,
        linewidth=0.55,
        linestyle="-",
    )
    ax.tick_params(
        axis="both",
        which="major",
        length=2.5,
        width=0.6,
        pad=3,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#C8CED6")
        ax.spines[spine].set_linewidth(0.65)


def style_panel_title(ax, title):
    """Use consistent, understated panel headings without sample size suffixes."""
    ax.set_title(
        title,
        loc="left",
        pad=7,
        color=C_TEXT,
        fontsize=8.5,
        fontweight="semibold",
    )


def save_both(fig, basename):
    """Save a vector PDF and a high-resolution PNG."""
    pdf_path = os.path.join(OUT_DIR, f"{basename}.pdf")
    png_path = os.path.join(OUT_DIR, f"{basename}.png")

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# =============================================================================
# FIGURE 2
# 2WikiMultihopQA Synthesis Frontier & Mechanistic Dissection
# =============================================================================

def make_fig2():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.45),
        gridspec_kw={
            "width_ratios": [1.55, 1.0, 1.0],
            "wspace": 0.43,
        },
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.84,
    )

    # -------------------------------------------------------------------------
    # PANEL A: ACCURACY VS. LOGICAL TOKEN COST
    # -------------------------------------------------------------------------
    ax = axes[0]

    pts = [
        ("Full-Union (1 call/q)", 349.1, 80.24, C_GREEN, "D"),
        ("NoComm (3 calls/q)", 505.0, 22.17, C_GREY, "o"),
        ("Random (4 calls/q)", 697.1, 36.85, C_PURPLE, "s"),
        ("Complementary (3.9 calls/q)", 686.4, 45.44, C_ORANGE, "^"),
        ("Heuristic (4 calls/q)", 700.6, 50.91, C_NAVY, "v"),
        ("In-Star (Sink 1)", 708.5, 54.96, C_TEAL, "P"),
        ("In-Star (Fallback / Vote)", 708.5, 59.83, C_TEAL, "*"),
        ("Dense (Sink 1)", 1109.3, 54.96, C_ORANGE, "P"),
        ("Dense (Fallback)", 1109.3, 59.83, C_ORANGE, "^"),
        ("Dense (All-Agent Vote, 6 calls/q)", 1109.3, 77.26, C_ORANGE, "*"),
    ]

    for label, tok, f1, color, marker in pts:
        ax.scatter(
            tok,
            f1,
            s=65 if marker == "*" else 39,
            color=color,
            marker=marker,
            edgecolors=C_WHITE,
            linewidths=0.8,
            zorder=5,
        )

    annotation_specs = [
        ("Full-Union (1 call/q)", 349.1, 80.24, C_GREEN, 7, 1, "left"),
        ("NoComm (3 calls/q)", 505.0, 22.17, C_GREY, 7, -1, "left"),
        ("Random (4 calls/q)", 697.1, 36.85, C_PURPLE, 7, -8, "left"),
        ("Complementary (3.9 calls/q)", 686.4, 45.44, C_ORANGE, -8, -10, "right"),
        ("Heuristic (4 calls/q)", 700.6, 50.91, C_NAVY, -8, 8, "right"),
        ("In-Star (Sink 1)", 708.5, 54.96, C_TEAL, -8, -13, "right"),
        ("In-Star (Fallback / Vote)", 708.5, 59.83, C_TEAL, -8, 11, "right"),
        ("Dense (Sink 1)", 1109.3, 54.96, C_ORANGE, -8, -13, "right"),
        ("Dense (Fallback)", 1109.3, 59.83, C_ORANGE, -8, 8, "right"),
        ("Dense (All-Agent Vote, 6 calls/q)", 1109.3, 77.26, C_ORANGE, -8, 4, "right"),
    ]

    for label, x, y, color, dx, dy, ha in annotation_specs:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=5.7,
            color=color,
            annotation_clip=False,
            zorder=6,
        )

    # Parity comparison line
    ax.plot(
        [708.5, 1109.3],
        [59.83, 59.83],
        linestyle=(0, (3, 2)),
        color=C_TEAL,
        linewidth=1.0,
        alpha=0.9,
        zorder=2,
    )

    ax.annotate(
        "Exact $\\mathcal{R}_{\\mathrm{SP}}$ Parity (-36.1% tok)",
        xy=(910, 59.83),
        xytext=(0, -17),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=5.8,
        color=C_TEAL,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=C_WHITE,
            edgecolor="none",
            alpha=0.96,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=C_TEAL,
            linewidth=0.65,
        ),
        zorder=7,
    )

    # Clean newline without literal \n
    ax.annotate(
        "+17.4 F1\n(3x R1 upd.)",
        xy=(1109.3, 75.8),
        xytext=(20, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=5.8,
        color=C_ORANGE,
        fontweight="medium",
        arrowprops=dict(
            arrowstyle="->",
            color=C_ORANGE,
            linewidth=0.9,
            shrinkA=2,
            shrinkB=3,
        ),
        annotation_clip=False,
    )

    ax.set_xlabel("Known Logical Model Tokens (Thousands)", labelpad=5)
    ax.set_ylabel("Official Answer F1 (%)", labelpad=5)
    ax.set_xlim(290, 1290)
    ax.set_ylim(16, 87)

    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    style_axis(ax, grid_axis="y")
    style_panel_title(ax, "(a) 2Wiki Cost-Quality Frontier")

    # -------------------------------------------------------------------------
    # PANEL B: SINK-1 ABSTENTION AND FALLBACK
    # -------------------------------------------------------------------------
    ax = axes[1]

    categories = [
        "Sink 1 Speaks\n& Correct",
        "Sink 1 Speaks\n& Wrong",
        "Sink 1 Abstains,\nLeaves Abstain",
        "Sink 1 Abstains,\nLeaf R0 Recovers",
        "Sink 1 Abstains,\nLeaves Wrong",
    ]

    counts = [51.33, 10.67, 30.67, 4.00, 3.33]

    colors = [
        C_NAVY,
        C_GREY,
        C_LIGHTGREY,
        C_TEAL,
        "#B57373",
    ]

    y_pos = np.arange(len(categories))

    bars = ax.barh(
        y_pos,
        counts,
        color=colors,
        height=0.52,
        edgecolor=C_WHITE,
        linewidth=0.8,
        zorder=3,
    )

    for bar, pct in zip(bars, counts):
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.8,
            color=C_TEXT,
            fontweight="medium",
            annotation_clip=False,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=6.2)
    ax.invert_yaxis()

    ax.set_xlabel("Share of Pooled Queries (%)", labelpad=5)
    ax.set_xlim(0, 68)
    ax.xaxis.set_major_locator(MultipleLocator(20))

    ax.text(
        0.98,
        0.045,
        "+4.00 pp EM Lift\nat 0 Extra LLM Calls!",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=C_TEAL,
        fontweight="medium",
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="#F1F7F6",
            edgecolor="#C8E2DE",
            linewidth=0.65,
        ),
        zorder=5,
    )

    style_axis(ax, grid_axis="x")
    style_panel_title(ax, "(b) Sink-1 Abstention & Fallback")

    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=4)

    # -------------------------------------------------------------------------
    # PANEL C: WHY DENSE VOTING BEATS FIXED SINK 1
    # -------------------------------------------------------------------------
    ax = axes[2]

    win_cats = [
        "Updated Agent 2\nCorrect in R1",
        "Updated Agent 0\nCorrect in R1",
        "Recovered from\nUn-updated R0 Leaf",
        "Both Updated Leaves\nAgree & Correct",
    ]

    win_pcts = [71.43, 46.43, 17.86, 17.86]

    win_colors = [
        C_ORANGE,
        C_ORANGE,
        C_TEAL,
        C_RED,
    ]

    y2 = np.arange(len(win_cats))

    bars = ax.barh(
        y2,
        win_pcts,
        color=win_colors,
        height=0.52,
        edgecolor=C_WHITE,
        linewidth=0.8,
        zorder=3,
    )

    for bar, pct in zip(bars, win_pcts):
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.8,
            color=C_TEXT,
            fontweight="medium",
            annotation_clip=False,
        )

    ax.set_yticks(y2)
    ax.set_yticklabels(win_cats, fontsize=6.2)
    ax.invert_yaxis()

    ax.set_xlabel("Share of Discordant Dense Wins (%)", labelpad=5)
    ax.set_xlim(0, 95)
    ax.xaxis.set_major_locator(MultipleLocator(20))

    style_axis(ax, grid_axis="x")
    style_panel_title(ax, "(c) Why Dense Vote Beats Sink 1")

    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=4)

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    save_both(fig, "fig2_2wiki_synthesis")


if __name__ == "__main__":
    make_fig2()
