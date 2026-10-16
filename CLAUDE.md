# PL_prediction — project context for Claude

## What this is
Premier League match-outcome forecasting (home/draw/away), benchmarked against
the betting market. A CV portfolio project: rigor and honesty matter more than
flashy claims. Owner: Ishaan (MSc Data Science, Bristol).

## Layout
- `src/premodel/` — package.
  - Core: `config.py` (paths, splits, constants, live-model settings), `data.py`
    (download/load/devig, `league=` E0/E1/E2), `metrics.py` (RPS, Brier,
    log-loss, calibration), `stats.py` (date-block bootstrap), `plots.py`,
    `__main__.py` (CLI: `python -m premodel <cmd>`).
  - Models: `models/dixon_coles.py` (dc_v1), `models/dixon_coles_joint.py`
    (dc_joint: one rating per team across E0/E1/E2).
  - Backtest: `backtest.py` (dc_v1 walk-forward + betting sim + CLV).
  - Live pipeline: `forward.py` (predict both models, schema v2),
    `results.py` (football-data.org PL + Championship results), `commitment.py`
    (canonical JSON, commit/verify/audit, `fixture_probs` for v1/v2),
    `onchain.py` (only solana/solders importer), `track_record.py` (reveal,
    score, dc_joint vs dc_v1 head-to-head).
  - `experiments/` — shrinkage, championship, joint; each writes
    `reports/experiments/<name>/` (summary.json, figure, README note).
- `data/raw/` — football-data.co.uk `E0/E1/E2_<code>.csv` + `fixtures.csv`
  (git-ignored)
- `data/processed/` — regenerated outputs and API caches (git-ignored)
- `reports/` — `summary.json` + `figures/` (published dc_v1 backtest) and
  `experiments/` (committed)
- `predictions/` — `pending/` (git-ignored until reveal), `revealed/` and
  `track_record.csv` (committed by Ishaan)
- `tests/` — pytest; `pytest -m slow` re-runs the E0 backtest byte-for-byte

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
dc_v1 (published backtest): RPS 0.2016 vs market closing 0.1955; naive 0.2344.
Betting the model's edges: CLV ≈ −2.9%, same as random bets → no edge over
market. Experiments (reports/experiments/): shrinkage gave no detectable RPS
gain; the Championship market is not softer; dc_joint improves E1 (−0.0031) and
newly arrived teams (−0.0152), no detectable E0 change, still no market edge.

## Live models (model_version 0.2.0)
- Primary: dc_joint (plain, xi 0.005, `config.FORWARD_JOINT_XI`, drift-tested
  against the joint experiment). Shadow: dc_v1 (xi 0.003), unchanged.
- Every round's `predictions.json` (schema v2) holds both models' probabilities,
  so one commitment covers both. `E0-2026-10-12` is schema v1 (dc_v1 only).
- Adoption was decided after seeing the joint experiment's results (disclosed
  in README). Pre-registered live rule in `config.LIVE_EVALUATION`: after ≥150
  v2 matches and the 2026/27 season end, revert to dc_v1 only if the 95% CI of
  RPS(dc_joint) − RPS(dc_v1) is entirely above 0. Low power (≈ ±0.007 RPS).
- Data for predict: E0 CSV + API (completeness check blocks), E1 CSV + API
  (warning only), E2 CSV only.

## Roadmap
1. Weekly forward pipeline + Solana commit-reveal (verifiable track record)
2. Gradient-boosted model: DC ratings + xG + Elo + rest days, same harness.
   Shrink low-sample team ratings toward the promoted-team prior (evaluate in
   backtest first; currently a team with few matches gets extreme ratings, e.g.
   Coventry attack −1.50 after 5 games).
3. Solana AI agent project (separate repo)
