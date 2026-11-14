"""
Build the result figures from the CSVs written by experiments.py.

    python make_figures.py

Reads  results/training_log.csv, results/evaluation_results.csv,
       results/summary.csv
Writes figures/*.png at 200 dpi:
    1_learning_curves.png        cost, stockout rate, delivery efficiency
    2_evaluation_bars.png        greedy test results with baselines
    3_epsilon_sensitivity.png    fixed epsilon vs decaying epsilon
    4_exploration_over_time.png  epsilon schedule and non-greedy action rate

Colors are assigned once per strategy family and reused in every figure.
Variants inside a family (e.g. epsilon = 0.01 / 0.1 / 0.3) are lighter or
darker shades of the family color.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DPI = 200
MA = 100   # moving-average window, episodes

HEURISTIC = "Reorder-to-target heuristic (baseline)"
RANDOM = "Random policy (baseline)"

# ---------------------------------------------------------------------------
# Colors: one hue per family, fixed everywhere
# ---------------------------------------------------------------------------
FAMILY_COLOR = {
    "Fixed ε-greedy": "#2a78d6",    # blue
    "Decaying ε": "#eb6834",        # orange
    "Softmax": "#1baf7a",           # aqua
    "UCB": "#6250d6",               # violet
    "Heuristic": "#52514e",         # dark gray (baseline)
    "Random": "#a3a29c",            # light gray (baseline)
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def family(name: str) -> str:
    if name.startswith("Epsilon-greedy"):
        return "Fixed ε-greedy"
    if name.startswith("Decaying"):
        return "Decaying ε"
    if name.startswith("Softmax"):
        return "Softmax"
    if name.startswith("UCB"):
        return "UCB"
    return "Heuristic" if name == HEURISTIC else "Random"


def shade(hex_color: str, t: float) -> str:
    """t < 0 mixes toward white (lighter), t > 0 toward black (darker)."""
    c = np.array(to_rgb(hex_color))
    c = c + (1 - c) * (-t) if t < 0 else c * (1 - t)
    return to_hex(np.clip(c, 0, 1))


# lighter = less exploration, darker = more exploration
VARIANT_SHADE = {
    "Epsilon-greedy ε=0.01": -0.55, "Epsilon-greedy ε=0.1": -0.25,
    "Epsilon-greedy ε=0.3": 0.0,
    "Softmax τ=0.5": 0.0, "Softmax τ=5": 0.35,
    "UCB c=0.5": 0.0, "UCB c=2": 0.35,
}


def color(name: str) -> str:
    return shade(FAMILY_COLOR[family(name)], VARIANT_SHADE.get(name, 0.0))


def short(name: str) -> str:
    return {HEURISTIC: "Reorder heuristic", RANDOM: "Random policy",
            "Decaying ε 1→0.05": "Decaying ε (1→0.05)"}.get(name, name)


plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load():
    train = pd.read_csv(RESULTS / "training_log.csv")
    evals = pd.read_csv(RESULTS / "evaluation_results.csv")
    summary = pd.read_csv(RESULTS / "summary.csv", index_col=0)
    return train, evals, summary


def smoothed(train: pd.DataFrame, strategy: str, col: str):
    """100-episode moving average per seed, then mean and std across seeds."""
    d = train[train.strategy == strategy].pivot(index="episode", columns="run_seed", values=col)
    ma = d.rolling(MA, min_periods=MA).mean()
    return ma.index.to_numpy(), ma.mean(axis=1).to_numpy(), ma.std(axis=1).to_numpy()


def best_variants(summary: pd.DataFrame) -> list[str]:
    """Best variant of each learning family = lowest mean greedy test cost."""
    s = summary.drop([HEURISTIC, RANDOM])
    s = s.assign(fam=[family(n) for n in s.index])
    return [s[s.fam == f].final_cost_mean.idxmin()
            for f in ("Fixed ε-greedy", "Decaying ε", "Softmax", "UCB")]


def line_with_band(ax, x, m, sd, c, label, lw=1.8):
    ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.18, linewidth=0)
    ax.plot(x, m, color=c, lw=lw, label=label)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


money_fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
money_k_fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:g}k" if v else "$0")
pct_fmt = matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
ep_fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")


# ---------------------------------------------------------------------------
# Figure 1: learning curves
# ---------------------------------------------------------------------------
def fig_learning_curves(train, summary):
    best = best_variants(summary)
    panels = [("total_cost", "Monthly distribution cost", "Cost per 30-day month (USD)", money_fmt),
              ("stockout_rate", "Stockout rate", "Unmet demand (% of units)", pct_fmt),
              ("delivery_eff", "Delivery efficiency", "On-time deliveries (%)", pct_fmt)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        for name in best:
            x, m, sd = smoothed(train, name, col)
            line_with_band(ax, x, m, sd, color(name), short(name))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Training episode (30-day months)")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_formatter(ep_fmt)
        ax.set_xlim(MA, train.episode.max())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Learning curves during training (exploration on): best variant of each strategy",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.92, f"{MA}-episode moving average; line = mean of 5 seeds, band = ±1 std. "
             "Best variant = lowest mean greedy test cost within its family.",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "1_learning_curves.png")


# ---------------------------------------------------------------------------
# Figure 2: greedy evaluation bars
# ---------------------------------------------------------------------------
def hbar_panel(ax, names, means, stds, fmt_value, xfmt, title, xlabel):
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, color=[color(n) for n in names], height=0.68,
            error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=2.5), edgecolor="white",
            linewidth=1.5)
    ax.set_yticks(y, [short(n) for n in names])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    right = max(m + (s if np.isfinite(s) else 0) for m, s in zip(means, stds))
    for yi, m, s in zip(y, means, stds):
        ax.text(m + (s if np.isfinite(s) else 0) + right * 0.02, yi, fmt_value(m),
                va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, right * 1.22)


def fig_evaluation_bars(summary):
    names = list(summary.index)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    hbar_panel(axes[0], names, summary.final_cost_mean, summary.final_cost_std,
               lambda v: f"${v:,.0f}", money_k_fmt, "Distribution cost",
               "Mean cost per 30-day month (USD)")
    hbar_panel(axes[1], names, summary.stockout_rate_mean, summary.stockout_rate_std,
               lambda v: f"{v:.1%}", pct_fmt, "Stockout rate", "Unmet demand (% of units)")
    hbar_panel(axes[2], names, summary.delivery_eff_mean, summary.delivery_eff_std,
               lambda v: f"{v:.1%}", pct_fmt, "Delivery efficiency",
               "On-time deliveries (% of deliveries)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Greedy evaluation on 100 held-out months (no exploration)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.905, "Bars = mean of 5 training seeds, error bars = ±1 std across seeds. "
             "Gray bars are baselines; the heuristic is deterministic (std = 0).",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "2_evaluation_bars.png")


# ---------------------------------------------------------------------------
# Figure 3: epsilon sensitivity
# ---------------------------------------------------------------------------
def fig_epsilon_sensitivity(train, summary):
    names = ["Epsilon-greedy ε=0.01", "Epsilon-greedy ε=0.1",
             "Epsilon-greedy ε=0.3", "Decaying ε 1→0.05"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    ax = axes[0]
    for n in names:
        x, m, sd = smoothed(train, n, "total_cost")
        line_with_band(ax, x, m, sd, color(n), short(n))
    ax.set_title("Training cost (exploration on)", loc="left")
    ax.set_xlabel("Training episode (30-day months)")
    ax.set_ylabel("Cost per 30-day month (USD)")
    ax.yaxis.set_major_formatter(money_fmt)
    ax.xaxis.set_major_formatter(ep_fmt)
    ax.set_xlim(MA, train.episode.max())
    ax.legend(loc="upper right")

    s = summary.loc[names]
    hbar_panel(axes[1], names, s.final_cost_mean, s.final_cost_std,
               lambda v: f"${v:,.0f}", money_k_fmt, "Greedy test cost",
               "Mean cost per 30-day month (USD)")
    hbar_panel(axes[2], names, s.stockout_rate_mean, s.stockout_rate_std,
               lambda v: f"{v:.1%}", pct_fmt, "Greedy test stockout rate",
               "Unmet demand (% of units)")
    axes[2].set_yticklabels([])
    axes[2].tick_params(axis="y", length=0)
    fig.suptitle("Epsilon-greedy sensitivity: fixed ε vs. decaying ε",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.9, f"Left: {MA}-episode moving average, mean ± 1 std of 5 seeds. "
             "Right: 100 held-out months, mean ± 1 std across seeds.",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "3_epsilon_sensitivity.png")


# ---------------------------------------------------------------------------
# Figure 4: exploration over time
# ---------------------------------------------------------------------------
def fig_exploration(train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    dec = "Decaying ε 1→0.05"
    eps = train[(train.strategy == dec) & (train.run_seed == 0)]
    ax.plot(eps.episode, eps.epsilon, color=color(dec), lw=2, label="Decaying ε (1→0.05)")
    for n, v in [("Epsilon-greedy ε=0.3", 0.3), ("Epsilon-greedy ε=0.1", 0.1),
                 ("Epsilon-greedy ε=0.01", 0.01)]:
        ax.axhline(v, color=color(n), lw=1.3, ls="--", label=f"Fixed ε = {v:g}")
    ax.set_title("Exploration rate ε per episode", loc="left")
    ax.set_xlabel("Training episode (30-day months)")
    ax.set_ylabel("ε (probability of a random action)")
    ax.xaxis.set_major_formatter(ep_fmt)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, train.episode.max())
    ax.legend(loc="upper right")

    ax = axes[1]
    for n in ["Softmax τ=0.5", "Softmax τ=5", "UCB c=0.5", "UCB c=2"]:
        x, m, sd = smoothed(train, n, "non_greedy_rate")
        line_with_band(ax, x, m, sd, color(n), short(n))
    ax.set_title("Share of non-greedy actions: softmax and UCB", loc="left")
    ax.set_xlabel("Training episode (30-day months)")
    ax.set_ylabel("Actions not equal to best-known action (%)")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.xaxis.set_major_formatter(ep_fmt)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(MA, train.episode.max())
    ax.legend(loc="lower left", ncol=2)
    fig.suptitle("How much each strategy explores over time",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.9, "Left: ε schedule (identical for all seeds). Right: "
             f"{MA}-episode moving average, mean ± 1 std of 5 seeds.",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "4_exploration_over_time.png")


def main():
    train, evals, summary = load()
    fig_learning_curves(train, summary)
    fig_evaluation_bars(summary)
    fig_epsilon_sensitivity(train, summary)
    fig_exploration(train)


if __name__ == "__main__":
    main()
