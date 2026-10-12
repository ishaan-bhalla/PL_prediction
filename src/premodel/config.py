"""Project paths and experiment settings — the single place to change them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = ROOT / "predictions"
PENDING_DIR = PREDICTIONS_DIR / "pending"      # git-ignored until reveal
REVEALED_DIR = PREDICTIONS_DIR / "revealed"    # committed to git after reveal
TRACK_RECORD_PATH = PREDICTIONS_DIR / "track_record.csv"

# football-data.co.uk season codes: "1415" = 2014/15
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2014, 2027)]
# football-data.co.uk league codes: E0 = Premier League, E1 = Championship, E2 = League One
LEAGUES = ["E0", "E1", "E2"]  # E2 = League One
DEFAULT_LEAGUE = "E0"
DOWNLOAD_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
CURRENT_SEASON = SEASONS[-1]
CURRENT_SEASON_LABEL = f"20{CURRENT_SEASON[:2]}/{CURRENT_SEASON[2:]}"  # "2026/27"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_PATH = RAW_DIR / "fixtures.csv"
LEAGUE_DIV = "E0"
KICKOFF_TZ = "Europe/London"  # football-data.co.uk times are UK local time

# Walk-forward design
WARMUP_SEASONS = ["2014/15"]                        # training only, never scored
VALID_SEASONS = ["2015/16", "2016/17", "2017/18"]   # used to choose xi
TEST_START = "2018/19"                              # everything from here is test
LOOKBACK_DAYS = 3 * 365
XI_GRID = [0.0, 0.001, 0.0019, 0.003, 0.005]        # decay per day

# Betting simulation
EDGE_THRESHOLDS = [0.0, 0.03, 0.05, 0.10]
BOOTSTRAP_SAMPLES = 5000
SEED = 42

# Forward predictions. xi is frozen at the backtest's validation choice
# (reports/summary.json "chosen_xi"); a test fails if the two ever drift apart.
MODEL_NAME = "dixon-coles"
FORWARD_XI = 0.003
# predict refuses if the newest training match is more than this many days before the
# first kickoff (stale data download); `--allow-stale` overrides it.
STALE_DAYS = 16

# On-chain commitments (Solana devnet only; there is deliberately no mainnet path)
PREDICTIONS_TEST_DIR = PREDICTIONS_DIR / "test"  # --test dry runs, git-ignored
CLUSTER = "devnet"
DEVNET_RPC_URL = "https://api.devnet.solana.com"
DEVNET_GENESIS_HASH = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG"
MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
EXPLORER_TX_URL = "https://explorer.solana.com/tx/{sig}?cluster=devnet"
KEYPAIR_PATH = Path("~/.config/solana/premodel-devnet.json")            # env SOLANA_KEYPAIR
TEST_KEYPAIR_PATH = Path("~/.config/solana/premodel-devnet-test.json")  # SOLANA_TEST_KEYPAIR
# Public key of the wallet whose memos form the official record. Empty until set;
# commit (real), verify and audit refuse to run without it.
CANONICAL_SIGNER = "EKjxGc617c1YqVYGPGGNZutuLcHHnoEGR1Ku5iFjJSxo"
REVEAL_DELAY_HOURS = 3  # reveal allowed from last kickoff + this
# audit: an unrevealed round is "awaiting reveal" until 00:00 UTC on its round_id date
# + this many days, then "overdue". Derived from the memo alone, so anyone can apply it.
REVEAL_DEADLINE_DAYS = 10

# Results fallback: football-data.org API v4 (football-data.co.uk's CSV lags a few days).
# Key from env FOOTBALL_DATA_API_KEY (or .env); it is only ever sent as a request header.
ENV_PATH = ROOT / ".env"
# Whole current-season schedule (all statuses): finished rows fill CSV gaps, and the
# schedule tells predict which matches must already be in the training data.
RESULTS_API_URL = "https://api.football-data.org/v4/competitions/PL/matches"
RESULTS_API_CACHE = PROCESSED_DIR / "football_data_org_pl_matches.json"
SEASON_MATCHES = 380  # a complete schedule: 20 teams x 19 opponents x home/away
# football-data.org team shortName -> football-data.co.uk name. Unmapped names fail loudly.
API_TEAM_NAMES = {
    # 2026/27 clubs, checked against the live API on 2026-10-12
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton Hove": "Brighton",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Man United": "Man United",
    "Newcastle": "Newcastle",
    "Nottingham": "Nott'm Forest",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    # recent PL clubs, not in the current API response (unverified until they return)
    "Burnley": "Burnley",
    "Leicester City": "Leicester",
    "Southampton": "Southampton",
    "West Ham": "West Ham",
    "Wolverhampton": "Wolves",
}

# Championship results from football-data.org (code ELC). shortName -> football-data.co.uk
# name, checked against the live API on 2026-10-12; unmapped names fail loudly.
API_TEAM_NAMES_E1 = {
    "Birmingham": "Birmingham", "Blackburn": "Blackburn", "Bolton": "Bolton",
    "Bristol City": "Bristol City", "Burnley": "Burnley", "Cardiff": "Cardiff",
    "Charlton": "Charlton", "Derby County": "Derby", "Lincoln City": "Lincoln",
    "Middlesbrough": "Middlesbrough", "Millwall": "Millwall", "Norwich": "Norwich",
    "Portsmouth": "Portsmouth", "Preston NE": "Preston", "QPR": "QPR",
    "Sheffield Utd": "Sheffield United", "Southampton": "Southampton", "Stoke": "Stoke",
    "Swansea": "Swansea", "Watford": "Watford", "West Brom": "West Brom",
    "West Ham": "West Ham", "Wolverhampton": "Wolves", "Wrexham": "Wrexham",
}
# Per football-data.org competition: endpoint, cumulative cache, name map, league code and
# matches in a complete season (used by the schedule-completeness check).
RESULTS_API = {
    "PL": {"url": RESULTS_API_URL, "cache": RESULTS_API_CACHE, "names": API_TEAM_NAMES,
           "league": "E0", "season_matches": SEASON_MATCHES},
    "ELC": {"url": "https://api.football-data.org/v4/competitions/ELC/matches",
            "cache": PROCESSED_DIR / "football_data_org_elc_matches.json",
            "names": API_TEAM_NAMES_E1, "league": "E1", "season_matches": 552},
}

# --- Live models (model_version 0.2.0) ---------------------------------------------------
# Primary: dc_joint (plain variant, one rating per team across E0/E1/E2), xi as selected in
# reports/experiments/joint/summary.json (a drift test keeps the two equal). Shadow: dc_v1,
# unchanged. Both are committed in every round's predictions.json (schema v2).
# Adoption was decided AFTER seeing the joint experiment's results; no adoption rule was
# pre-registered for that experiment.
PRIMARY_MODEL = "dc_joint"
FORWARD_JOINT_XI = 0.005
JOINT_LEAGUES = ["E0", "E1", "E2"]
MODELS = {
    "dc_joint": {"version": "0.2.0", "variant": "plain", "xi": FORWARD_JOINT_XI},
    "dc_v1": {"version": "0.1.0", "xi": FORWARD_XI},
}

# Pre-registered live evaluation rule (written 2026-10-12, before any schema v2 match).
LIVE_EVALUATION = {
    "min_v2_matches": 150,
    "season": "2026/27",
    "season_end": "2027-05-30",  # football-data.org PL 2026/27 endDate
    "rule": ("Evaluated once at least 150 schema-v2 matches are scored AND the last E0 "
             "2026/27 fixture (season end 2027-05-30) is scored, whichever is later. If the "
             "95% CI (paired date-block bootstrap) of RPS(dc_joint) - RPS(dc_v1) on live "
             "matches lies entirely above 0, dc_joint is worse: revert the primary model to "
             "dc_v1. Otherwise keep dc_joint. The result is reported either way."),
    "power": ("Low by design: at about 150 matches the CI half-width is roughly +/-0.007 "
              "RPS (against +/-0.0015 on the 3,090-match backtest), so the rule can only "
              "detect a large deterioration and cannot confirm an improvement."),
}
