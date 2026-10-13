# PL_prediction — project context for Claude

## What this is
Premier League match-outcome forecasting (home/draw/away), benchmarked against
the betting market. A CV portfolio project: rigor and honesty matter more than
flashy claims. Owner: Ishaan (MSc Data Science, Bristol).

## Layout
- `src/premodel/` — package. `config.py` (paths, splits, constants), `data.py`
  (download/load/devig), `metrics.py` (RPS, Brier, log-loss, calibration),
  `models/dixon_coles.py`, `backtest.py` (walk-forward + betting sim + CLV),
  `plots.py`, `__main__.py` (CLI: `python -m premodel <cmd>`)
- `data/raw/` — football-data.co.uk `E0_<code>.csv` files (git-ignored)
- `data/processed/` — regenerated outputs (git-ignored)
- `reports/` — `summary.json` + `figures/` (committed)
- `tests/` — pytest

## Commands
- `make setup` / `make test` / `make lint` / `make all` (backtest + plots)
- `make data` re-downloads raw CSVs

## Rules
- **No leakage, ever.** Any model fit for a match may only use matches played
  strictly before that match's date. `tests/test_data_and_backtest.py` enforces
  this for the backtest; new pipelines need equivalent tests.
- Odds: "open" = Friday/Tuesday pre-match price (what you could take), "close" =
  closing price. Prefer Pinnacle (PS*), fall back to market average (Avg*).
  De-vig proportionally (`data.devig`).
- Evaluate with proper scoring rules (RPS primary) and CLV, never P&L alone.
- `make test` and `make lint` (ruff, line length 100) must pass at the end of
  every piece of work.
- **Never run `git commit`, `git push`, `git reset`, `git rebase` or any command
  that changes git history.** Ishaan commits himself. At the end of each piece
  of work, stop and give: a short summary of what changed, the list of files
  touched, and a suggested commit message. `git status` / `git diff` are fine.
- **Never commit or write secrets into the repo**: wallet keypairs, API keys,
  `.env`. Keypairs live outside the repo (e.g. `~/.config/solana/`). Never
  print or log a private key. Never touch Solana mainnet unless Ishaan
  explicitly asks in that session.
- Don't change backtest behaviour or reported numbers unless asked; if a change
  alters results, say so and regenerate `reports/`.
- Suggested commit messages must be plain: no Co-Authored-By or any AI/Claude
  attribution lines.

## Current baseline (test 2018/19–2026/27, 3,090 matches)
Dixon-Coles RPS 0.2016 vs market closing 0.1955; naive 0.2344. Betting the
model's edges: CLV ≈ −2.9%, same as random bets → no edge over market.

## Roadmap
1. Weekly forward pipeline + Solana commit-reveal (verifiable track record)
2. Gradient-boosted model: DC ratings + xG + Elo + rest days, same harness.
   Shrink low-sample team ratings toward the promoted-team prior (evaluate in
   backtest first; currently a team with few matches gets extreme ratings, e.g.
   Coventry attack −1.50 after 5 games).
3. Solana AI agent project (separate repo)
