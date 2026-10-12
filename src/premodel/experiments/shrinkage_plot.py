"""Figure for the shrinkage experiment (reports/experiments/shrinkage/shrinkage.png).

Reads only summary.json. Colours: categorical slots 1-3 of the project palette (validated
for all pairs, light surface); slot 3 sits below 3:1 contrast, so every series also has
its own marker shape and a direct label, and the write-up carries the numbers as tables.
"""
import json
from pathlib import Path

import numpy as np

from premodel.experiments.shrinkage import OUT_DIR
from premodel.plots import GRID, INK, INK2, NEUTRAL, plt

SERIES = {  # name: (label, colour, marker)
    "promoted_only": ("promoted-only", "#2a78d6", "o"),
    "uniform": ("uniform", "#eb6834", "s"),
    "uniform_zero_prior": ("all priors 0 (diagnostic)", "#1baf7a", "^"),
}
INF_X = 1024.0  # where lam = inf is drawn on the log axis


def _lam_x(lam) -> float:
    return INF_X if lam in ("inf", float("inf")) else float(lam)


def _ci(ax, y, block, colour, marker, label=None):
    lo, hi = block["ci95"]
    ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round")
    ax.plot(block["mean"], y, marker=marker, ms=8, color=colour, mec="white", mew=1.5,
            ls="none", label=label)


def render(summary_path: Path = OUT_DIR / "summary.json", out: Path = OUT_DIR / "shrinkage.png"):
    s = json.loads(summary_path.read_text())
    chosen = s["chosen"]["variant"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # A. Validation tuning curves
    ax = axes[0]
    base = s["tuning"][chosen][0]["rps"]  # lam = 0 row: dc_v1
    ax.axhline(base, color=NEUTRAL, lw=1.5, ls="--")
    ax.text(0.27, base + 0.00008, "dc_v1 (λ = 0)", color=INK2, fontsize=9)
    for name, (label, colour, marker) in SERIES.items():
        rows = [r for r in s["tuning"][name] if r["lam"] not in (0, 0.0)]
        x = [_lam_x(r["lam"]) for r in rows]
        y = [r["rps"] for r in rows]
        ax.plot(x, y, color=colour, marker=marker, ms=6, mec="white", mew=1.2, label=label)
        best = min(rows, key=lambda r: r["rps"])
        if name == chosen:
            ax.plot(_lam_x(best["lam"]), best["rps"], marker="o", ms=14, mfc="none",
                    mec=INK, mew=1.5)
            ax.annotate(f"selected: λ = {best['lam']:g}", (_lam_x(best["lam"]), best["rps"]),
                        xytext=(0, 16), textcoords="offset points", ha="center",
                        fontsize=9, color=INK)
    ax.axvline(32, color=GRID, lw=1.5, zorder=0)
    ax.text(36, 0.19985, "run 1 grid ended at λ = 32", ha="left", va="top",
            fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 4, 16, 64, 256, INF_X],
                  ["0.25", "1", "4", "16", "64", "256", "∞"])
    ax.minorticks_off()
    ax.set_ylim(0.1958, 0.2010)
    ax.set_xlabel("λ (shrinkage strength)")
    ax.set_ylabel("validation RPS (2015/16–2017/18)")
    ax.set_title("A. Tuning on validation only")
    ax.text(0.99, 0.62, "uniform curves exit\nabove λ = 16\n(RPS 0.224, 0.229 at ∞)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # B. Per-season RPS difference, selected model vs dc_v1
    ax = axes[1]
    comp = s["primary_test"]["comparison"]
    label, colour, marker = SERIES[chosen]
    seasons = comp["per_season"]
    ys = np.arange(len(seasons))[::-1] + 1.5
    for y, r in zip(ys, seasons, strict=True):
        _ci(ax, y, r["rps_diff"], colour, marker)
    _ci(ax, 0, comp["rps_diff"], INK, "D")
    ax.axhline(0.75, color=GRID, lw=1)
    ax.axvline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.set_yticks([*ys, 0], [f"{r['season']} (n={r['n']})" for r in seasons] + ["all test"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RPS difference vs dc_v1 (← better), 95% CI")
    ax.set_title(f"B. Test: {label}, λ = {s['chosen']['lam']:g}")

    # C. Subgroups, all three models
    ax = axes[2]
    groups = [("promoted_first_10_games", "promoted team,\nfirst 10 games"),
              ("all_other_matches", "all other\nmatches")]
    blocks = {chosen: comp, **s["secondary_test"], **s["diagnostic_test"]}
    for g, (key, _) in enumerate(groups):
        for k, (name, (lab, col, mk)) in enumerate(SERIES.items()):
            sub = blocks[name]["subgroups"][key]
            _ci(ax, (1 - g) * 1.6 + 0.35 - 0.35 * k, sub["rps_diff"], col, mk,
                label=lab if g == 0 else None)
    ns = [blocks[chosen]["subgroups"][k]["n"] for k, _ in groups]
    ax.set_yticks([1.6, 0], [f"{t}\n(n={n})" for (_, t), n in zip(groups, ns, strict=True)])
    ax.axvline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RPS difference vs dc_v1 (← better), 95% CI")
    ax.set_title("C. Test subgroups")
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    fig.suptitle("Shrinking low-sample team ratings: no detectable overall RPS gain on test",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, "Test = the published 3,090 matches (2018/19 to 20 Sep 2026). "
             "CIs from a paired bootstrap over match dates.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    print(f"wrote {render()}")


if __name__ == "__main__":
    main()
