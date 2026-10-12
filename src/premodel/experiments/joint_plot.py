"""Figure for the joint multi-league experiment (reports/experiments/joint/joint.png).

Reads only summary.json. E0 / E1 use categorical slots 1-2 of the project palette (as in
the championship figure); pooled results are drawn in ink; markers differ per series and
every point is labelled, and the write-up carries the numbers as tables.
"""
import json
from pathlib import Path

from premodel.experiments.joint import OUT_DIR
from premodel.plots import GRID, INK, INK2, NEUTRAL, plt

LEAGUE = {"E0": ("Premier League (E0)", "#2a78d6", "o"),
          "E1": ("Championship (E1)", "#eb6834", "s")}
MUTED = "#9a9893"


def _ci(ax, y, block, colour, marker, ms=8, filled=True):
    lo, hi = block["ci95"]
    ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round")
    ax.plot(block["mean"], y, marker=marker, ms=ms, ls="none", mew=1.8,
            color=colour, mfc=colour if filled else "white", mec=colour if not filled
            else "white")


def render(summary_path: Path = OUT_DIR / "summary.json", out: Path = OUT_DIR / "joint.png"):
    s = json.loads(summary_path.read_text())
    p = s["primary"]
    lg = p["leagues"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    # A. Pre-registered questions, then the non-decisive breakdown
    ax = axes[0]
    sub = {code: {k: v["rps_diff"] for k, v in lg[code]["entrant_subgroups"].items()}
           for code in ("E0", "E1")}
    rows = [("Q1  E0, all matches", lg["E0"]["rps_diff"], LEAGUE["E0"][1], "o", True),
            ("Q2  E1, all matches", lg["E1"]["rps_diff"], LEAGUE["E1"][1], "s", True),
            ("Q3  new teams, first 10 games\n(E0 + E1 pooled)", p["q3_pooled_new_teams"], INK,
             "D", True),
            ("E1: relegated from the PL", sub["E1"]["from_above"],
             LEAGUE["E1"][1], "s", False),
            ("E1: promoted from League One", sub["E1"]["from_below"],
             LEAGUE["E1"][1], "s", False),
            ("E0: promoted from the Championship", sub["E0"]["from_below"],
             LEAGUE["E0"][1], "o", False)]
    ys = [6.4, 5.4, 4.3, 2.4, 1.4, 0.4]
    for y, (_, b, col, mk, decisive) in zip(ys, rows, strict=True):
        _ci(ax, y, b, col, mk, filled=decisive)
        ax.text(max(b["ci95"][1], 0) + 0.0012, y, f"{b['mean']:+.4f}", va="center",
                fontsize=9, color=INK2)
    ax.axhline(3.35, color=GRID, lw=1)
    ax.text(-0.045, 3.1, "breakdown of Q3, reported, not decisive", fontsize=8.5,
            color=INK2, va="center")
    ax.axvline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(-0.3, 7.0)
    ax.set_xlim(-0.047, 0.022)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RPS difference, dc_joint − dc_v1 (← better), 95% CI")
    ax.set_title("A. Pre-registered questions")

    # B. Per-season RPS difference, both leagues
    ax = axes[1]
    seasons = [r["season"] for r in lg["E0"]["per_season"]]
    for code, off in (("E0", 0.17), ("E1", -0.17)):
        label, col, mk = LEAGUE[code]
        for i, r in enumerate(lg[code]["per_season"]):
            _ci(ax, len(seasons) - i + off, r["rps_diff"], col, mk, ms=6)
        ax.plot([], [], color=col, marker=mk, ls="-", label=label)
    ax.axvline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.set_yticks([len(seasons) - i for i in range(len(seasons))],
                  [f"{x} (n={lg['E0']['per_season'][i]['n']}/{lg['E1']['per_season'][i]['n']})"
                   for i, x in enumerate(seasons)], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RPS difference, dc_joint − dc_v1 (← better), 95% CI")
    ax.set_title("B. By season (n = E0/E1)")
    ax.set_ylim(0.4, len(seasons) + 1.3)  # empty band on top for the legend
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)

    # C. Still behind the market: gap to the closing line, dc_v1 vs dc_joint
    ax = axes[2]
    ys = {"E0": (3.3, 2.7), "E1": (1.3, 0.7)}
    for code, (y_v1, y_j) in ys.items():
        label, col, mk = LEAGUE[code]
        g = lg[code]["gap_to_close"]
        _ci(ax, y_v1, g["v1"], col, mk, filled=False)
        _ci(ax, y_j, g["joint"], col, mk)
        for y, b in ((y_v1, g["v1"]), (y_j, g["joint"])):
            ax.text(b["ci95"][1] + 0.0003, y, f"{b['mean']:+.4f}", va="center", fontsize=9,
                    color=INK2)
    ax.axvline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.text(0.0002, 3.75, "← as good as the closing line", fontsize=8.5, color=INK2)
    ax.set_yticks([3.3, 2.7, 1.3, 0.7], ["E0  dc_v1", "E0  dc_joint", "E1  dc_v1",
                                          "E1  dc_joint"])
    ax.set_ylim(0.2, 4.0)
    ax.set_xlim(-0.0015, 0.0125)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RPS gap to devigged closing odds (→ worse), 95% CI")
    ax.set_title("C. Gap to the market")

    fig.suptitle("Joint E0/E1/E2 model: better in the Championship and for new teams, "
                 "not in the Premier League overall", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    n = (lg["E0"]["n"], lg["E1"]["n"])
    fig.text(0.01, 0.905, f"dc_joint (plain, ξ = {p['config']['xi']}) vs dc_v1 on the same "
             f"test matches, 2018/19 to 20 Sep 2026: E0 {n[0]:,}, E1 {n[1]:,}. Filled = "
             "decision or dc_joint; hollow = not decisive or dc_v1. CIs: paired bootstrap "
             "over match dates.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"wrote {render()}")
