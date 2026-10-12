# EPL Match Forecasting: Dixon-Coles vs the Betting Market

A walk-forward backtest of the Dixon-Coles (1997) football model on 12+ seasons of
Premier League data, benchmarked against bookmaker opening and closing odds, with
closing-line value (CLV) analysis.

![Backtest summary](reports/figures/backtest_summary.png)

## Results (test period 2018/19 – 2026/27, 3,090 matches)

| Forecaster | RPS ↓ | Brier ↓ | Log-loss ↓ |
|---|---|---|---|
| Naive base rates | 0.2344 | 0.6472 | 1.0694 |
| **Dixon-Coles (this repo)** | **0.2016** | **0.5792** | **0.9804** |
| Market – opening (devigged) | 0.1967 | 0.5690 | 0.9600 |
| Market – closing (devigged) | 0.1955 | 0.5665 | 0.9564 |

Betting simulation (flat 1-unit stakes at opening odds, one bet per match when
the model's expected value exceeds the threshold):

| Edge threshold | Bets | ROI | 95% bootstrap CI | Mean CLV | Bets beating close |
|---|---|---|---|---|---|
| > 0% | 2,880 | −3.7% | −10.0% … +2.9% | −3.0% | 31% |
| > 5% | 2,276 | −3.7% | −10.7% … +3.7% | −2.9% | 33% |
| > 10% | 1,757 | −4.7% | −13.7% … +4.3% | −2.8% | 34% |

For reference, betting a **random** outcome on every match gives CLV −2.9%
(that is just the bookmaker's margin).

### What this shows

1. **The model learns real signal.** It cuts RPS by 14% versus naive base
   rates and is well calibrated.
2. **It does not beat the market.** The closing line is sharper in every
   single season.
3. **Its "edges" are noise.** Bets on the model's disagreements with the market
   have CLV indistinguishable from random bets. Blending the model into the
   opening odds was also tested: the optimal weight chosen on validation
   seasons was **0**, meaning the model adds no information the market doesn't
   already price.
4. **Profit and loss is the wrong yardstick.** Even across ~2,300 bets the ROI
   confidence interval spans roughly ±7%. CLV detects the absence of edge far
   faster than P&L does.

This is the expected result for a goals-only model and it establishes the
baseline any richer model (xG, Elo, lineups) has to beat.

## Method

- **Model**: home goals ~ Poisson(exp(γ + att_home + def_away)), away goals ~
  Poisson(exp(att_away + def_home)), with the Dixon-Coles ρ correction on the
  0-0 / 0-1 / 1-0 / 1-1 cells and exponential time-decay weights exp(−ξ·days).
  Fitted by weighted maximum likelihood (L-BFGS-B, analytic gradient,
  gradient-checked), small ridge penalty, sum-to-zero attack constraint.
- **Promoted teams** with no top-flight data get the mean ratings of the three
  weakest fitted teams.
- **Walk-forward**: refit before every match date on the previous 3 years of
  matches only. No future information is used.
- **Splits**: 2014/15 warm-up · 2015/16–2017/18 validation (choose ξ) ·
  2018/19–2026/27 test. Chosen ξ = 0.003/day (half-life ≈ 231 days).
- **Odds**: Pinnacle (the sharpest book) for opening and closing prices; the
  market average where Pinnacle is unavailable (later 2025/26 and 2026/27,
  220 test matches). Margins removed proportionally.
- **CLV** = opening odds × devigged closing probability − 1, i.e. the
  bet's expected value if the closing line is the truth.

## Project structure

```
Prem_prediction/
├── data/
│   ├── raw/                 # football-data.co.uk CSVs (git-ignored; `make data`)
│   └── processed/           # regenerated outputs, e.g. test_predictions.csv
├── notebooks/               # exploration
├── reports/
│   ├── summary.json         # every metric from the latest backtest
│   └── figures/             # charts used in this README
├── src/premodel/
│   ├── config.py            # paths, splits, decay grid, thresholds
│   ├── data.py              # download, load, de-vig
│   ├── metrics.py           # RPS, Brier, log-loss, calibration
│   ├── models/dixon_coles.py
│   ├── backtest.py          # walk-forward, betting sim, CLV
│   ├── plots.py
│   └── __main__.py          # CLI: python -m premodel <command>
├── tests/                   # model, metrics, and no-leakage tests
├── Makefile
└── pyproject.toml
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
make setup      # pip install -e ".[dev]"
make data       # download season CSVs into data/raw/
make all        # tune decay, walk-forward backtest, render charts (~30 s)
make test       # 12 tests incl. a check that no fit ever sees future matches
```

Data: [football-data.co.uk](https://www.football-data.co.uk) Premier League
CSVs.

## Next steps

- [ ] Gradient-boosted model on Dixon-Coles ratings + xG (the 2026/27 files
      now include `HxG`/`AxG`), Elo and rest days; same walk-forward harness
- [ ] Weekly forward pipeline logging predictions and CLV live
- [ ] Commit-reveal of each week's predictions on Solana (Memo program) so the
      forward track record is verifiably pre-kickoff
