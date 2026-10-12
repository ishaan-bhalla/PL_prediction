"""Command-line entry point:  python -m premodel <command>

Commands
  download   fetch season CSVs from football-data.co.uk into data/raw/
  coverage   show matches and odds-source coverage per season
  backtest   tune decay, run the walk-forward backtest, write reports/summary.json
  plots      render reports/figures/backtest_summary.png
  all        backtest + plots
"""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="premodel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "coverage", "backtest", "plots", "all"])
    parser.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    if args.command == "download":
        from premodel.data import download
        download(overwrite=args.overwrite)
    elif args.command == "coverage":
        from premodel.data import coverage_report, load_matches
        print(coverage_report(load_matches()).to_string())
    elif args.command in ("backtest", "all"):
        from premodel.backtest import run
        run()
    if args.command in ("plots", "all"):
        from premodel.plots import render
        render()


if __name__ == "__main__":
    main()
