"""Figure for the Championship experiment (reports/experiments/championship/championship.png).

Reads only summary.json. Two series, categorical slots 1-2 of the project palette
(validated all-pairs on the light surface); each also has its own marker shape and a
direct label, and the write-up carries every number as a table.
"""
import json
from pathlib import Path

from premodel.experiments.championship import OUT_DIR
from premodel.plots import GRID, INK, INK2, NEUTRAL, plt

LEAGUE = {"E0": ("Premier League (E0)", "#2a78d6", "o"),
          "E1": ("Championship (E1)", "#eb6834", "s")}


def _ci(ax, y, block, colour, marker, ms=8):
    lo, hi = block["ci95"]
    ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round")
    ax.plot(block["mean"], y, marker=marker, ms=ms, color=colour, mec="white", mew=1.5,
            ls="none")


def render(summary_path: Path = OUT_DIR / "summary.json",
           out: Path = OUT_DIR / "championship.png") -> Path:
    s = json.loads(summary_path.read_text())
    q = s["primary"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # A. Q1: RPS gap to the closing market, and the difference
    ax = axes[0]
    rows = [("E0", q["Q1"]["gap_E0"]), ("E1", q["Q1"]["gap_E1"])]
    for y, (lg, b) in zip((2.2, 1.2), rows, strict=True):
        label, col, mk = LEAGUE[lg]
        _ci(ax, y, b, col, mk)
        ax.text(b["ci95"][1] + 0.0004, y, f"{b['mean']:+.4f}", va="center", fontsize=9,
                color=INK2)
    d = q["Q1"]["difference_E1_minus_E0_joint"]
    _ci(ax, 0, d, INK, "D")
    ax.text(d["ci95"][1] + 0.0004, 0, f"{d['mean']:+.4f}", va="center", fontsize=9,
            color=INK2)
    ax.axhline(0.6, color=GRID, lw=1)
    ax.axvline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.set_yticks([2.2, 1.2, 0], [LEAGUE["E0"][0], LEAGUE["E1"][0], "E1 − E0\n(joint dates)"])
    ax.set_ylim(-0.6, 2.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RPS gap: model − devigged close (→ model worse), 95% CI")
    ax.set_title("Q1. Gap to the closing market")

    # B. Q2: CLV of model-edge bets minus a random bet on the same match
    ax = axes[1]
    for y, lg in zip((1.2, 0.2), ("E0", "E1"), strict=True):
        label, col, mk = LEAGUE[lg]
        b = q["Q2"][lg]
        _ci(ax, y, b, col, mk)
        ax.text(b["ci95"][1] + 0.0003, y, f"{b['mean']:+.2%}", va="center", fontsize=9,
                color=INK2)
    ax.axvline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.text(0.0002, 1.62, "softer market →", fontsize=8.5, color=INK2)
    ax.set_yticks([1.2, 0.2], [LEAGUE["E0"][0], LEAGUE["E1"][0]])
    ax.set_ylim(-0.4, 1.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1%}"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("CLV of model bets − random bet, same match, 95% CI")
    ax.set_title("Q2. Do the model's bets beat a random bet?")

    # C. Q3: blend weight on validation (RPS relative to the open market alone)
    ax = axes[2]
    for lg in ("E0", "E1"):
        label, col, mk = LEAGUE[lg]
        curve = q["Q3"][lg]["validation_curve"]
        base = curve[0]["rps"]
        ax.plot([c["w"] for c in curve], [c["rps"] - base for c in curve], color=col,
                marker=mk, ms=5, mec="white", mew=1, label=f"{label}: chosen w = "
                f"{q['Q3'][lg]['w']:g}")
    ax.axhline(0, color=NEUTRAL, lw=1.5, ls="--")
    ax.set_xlabel("weight w on the model (rest: open market)")
    ax.set_ylabel("validation RPS − open market alone")
    ax.set_title("Q3. Does blending the model in help?")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(0.03, 0.62, "w = 0 in both leagues:\nthe test blend is the open market",
            transform=ax.transAxes, ha="left", va="center", fontsize=8.5, color=INK2)

    fig.suptitle("The Championship market is not softer for this model",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    n = s["test_window"]["n"]
    fig.text(0.01, 0.9, f"Same dc_v1 model in both leagues. Test 2018/19 to 20 Sep 2026: "
             f"E0 {n['E0']:,} matches, E1 {n['E1']:,}. CIs: paired bootstrap over match "
             "dates. Decision rules fixed before the run.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"wrote {render()}")
