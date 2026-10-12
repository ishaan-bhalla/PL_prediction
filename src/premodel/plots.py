"""Render the backtest summary figure (reports/figures/backtest_summary.png)."""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from premodel import config  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
MODEL = "#2a78d6"   # categorical slot 1
MARKET = "#eb6834"  # categorical slot 2
NEUTRAL = "#9a9893"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
})


def render():
    s = json.loads((config.REPORTS_DIR / "summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))

    # A. Accuracy vs benchmarks (lower RPS is better)
    ax = axes[0, 0]
    names = ["Naive base rates", "Dixon-Coles",
             "Market (opening, devigged)", "Market (closing, devigged)"]
    labels = ["Naive base rates", "Dixon-Coles (ours)", "Market – opening", "Market – closing"]
    vals = [s["benchmarks"][n]["rps"] for n in names]
    colors = [NEUTRAL, MODEL, MARKET, MARKET]
    y = np.arange(len(vals))[::-1]
    ax.hlines(y, 0.19, vals, color=GRID, linewidth=1.5)
    ax.scatter(vals, y, color=colors, s=90, zorder=3, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + 0.0012, yi, f"{v:.4f}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.19, 0.242)
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Ranked probability score (lower is better)")
    ax.set_title(f"A. Forecast accuracy, {s['n_test_matches']:,} test matches")

    # B. Per-season RPS
    ax = axes[0, 1]
    ps = pd.DataFrame(s["per_season"])
    x = np.arange(len(ps))
    ax.plot(x, ps.rps_model, color=MODEL, marker="o", markersize=5, label="Dixon-Coles")
    ax.plot(x, ps.rps_close, color=MARKET, marker="o", markersize=5, label="Market – closing")
    ax.set_xticks(x, [p.replace("20", "", 1) for p in ps.season], rotation=0, fontsize=8.5)
    ax.set_ylabel("RPS (lower is better)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0.170, 0.222)
    ax.set_title("B. The market wins every season")
    ax.text(x[-1], ps.rps_model.iloc[-1], f"  n={ps.n.iloc[-1]}",
            color=INK2, fontsize=8.5, va="center")

    # C. Calibration, home-win probability
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1, linestyle="--", label="Perfect calibration")
    for key, color, lab in (("calibration_home_model", MODEL, "Dixon-Coles"),
                            ("calibration_home_close", MARKET, "Market – closing")):
        c = pd.DataFrame(s[key])
        ax.plot(c.pred, c.actual, color=color, marker="o", markersize=5, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed home-win rate")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("C. Both are well calibrated – the market is just sharper")

    # D. Cumulative P&L of betting the model's 'edges' vs random bets
    ax = axes[1, 1]
    b = next(b for b in s["betting"] if abs(b["threshold"] - 0.05) < 1e-9)
    pnl = np.array(b["pnl_series"])
    rnd = np.array(s["random_bet_baseline"]["pnl_series"])
    ax.axhline(0, color=INK2, linewidth=1)
    ax.plot(np.arange(1, len(pnl) + 1), pnl, color=MODEL, label="Model edge > 5%")
    ax.plot(np.arange(1, len(rnd) + 1), rnd, color=NEUTRAL, label="Random outcome")
    ax.set_xlabel("Bets placed (1 unit each, opening odds)")
    ax.set_ylabel("Cumulative profit (units)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"D. Betting the model: ROI {b['roi']:+.1%}, CLV {b['mean_clv']:+.1%}")

    fig.suptitle("Dixon-Coles vs the betting market — EPL walk-forward backtest, 2018/19–2026/27",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "backtest_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out.relative_to(config.ROOT))
    return out
