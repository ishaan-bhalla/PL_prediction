.PHONY: setup data coverage backtest plots all test lint experiment-shrinkage experiment-championship \
        experiment-joint \
        wallet predict commit reveal score verify audit

setup:          ## install the package, dev tools and Solana extras into the active environment
	pip install -e ".[dev,onchain]"

data:           ## download season CSVs into data/raw/
	python -m premodel download

coverage:       ## matches and odds sources per season
	python -m premodel coverage

backtest:       ## tune decay + walk-forward backtest -> reports/summary.json
	python -m premodel backtest

plots:          ## render reports/figures/backtest_summary.png
	python -m premodel plots

all: backtest plots

experiment-shrinkage:  ## dc_shrink vs dc_v1 -> reports/experiments/shrinkage/
	python -m premodel experiment shrinkage

experiment-championship:  ## is the E1 market softer than E0? -> reports/experiments/championship/
	python -m premodel experiment championship

experiment-joint:  ## joint E0/E1/E2 model vs dc_v1 -> reports/experiments/joint/
	python -m premodel experiment joint

test:
	pytest -q

lint:
	ruff check src tests

# --- Verifiable track record (Solana devnet; needs pip install -e ".[onchain]") ----------
# Optional: ROUND=E0-YYYY-MM-DD selects a round; ARGS passes extra flags, e.g.
#   make commit ARGS=--test      make reveal ARGS=--force      make predict ARGS=--allow-stale
ROUND_FLAG = $(if $(ROUND),--round $(ROUND))

wallet:         ## create/show the devnet keypair, request an airdrop
	python -m premodel wallet $(ARGS)

predict:        ## refresh data, predict the next round -> predictions/pending/
	python -m premodel predict $(ARGS)

commit:         ## post the round's salted hash as a devnet memo (before first kickoff)
	python -m premodel commit $(ROUND_FLAG) $(ARGS)

reveal:         ## move a finished committed round to predictions/revealed/
	python -m premodel reveal $(ROUND_FLAG) $(ARGS)

score:          ## score revealed rounds -> predictions/track_record.csv
	python -m premodel score $(ROUND_FLAG) $(ARGS)

verify:         ## PASS/FAIL a round against its on-chain commitment (needs ROUND=)
	python -m premodel verify --round $(ROUND) $(ARGS)

audit:          ## every premodel:v1 memo by the canonical signer; flags unrevealed rounds
	python -m premodel audit $(ARGS)
