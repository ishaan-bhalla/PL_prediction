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

# football-data.co.uk season codes: "1415" = 2014/15
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2014, 2027)]
DOWNLOAD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
CURRENT_SEASON = SEASONS[-1]
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
CANONICAL_SIGNER = ""
REVEAL_DELAY_HOURS = 3  # reveal allowed from last kickoff + this
# audit: an unrevealed round is "awaiting reveal" until 00:00 UTC on its round_id date
# + this many days, then "overdue". Derived from the memo alone, so anyone can apply it.
REVEAL_DEADLINE_DAYS = 10
