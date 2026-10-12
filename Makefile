.PHONY: setup data coverage backtest plots all test lint

setup:          ## install the package and dev tools into the active environment
	pip install -e ".[dev]"

data:           ## download season CSVs into data/raw/
	python -m premodel download

coverage:       ## matches and odds sources per season
	python -m premodel coverage

backtest:       ## tune decay + walk-forward backtest -> reports/summary.json
	python -m premodel backtest

plots:          ## render reports/figures/backtest_summary.png
	python -m premodel plots

all: backtest plots

test:
	pytest -q

lint:
	ruff check src tests
