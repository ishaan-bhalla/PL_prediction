"""Command-line entry point:  python -m premodel <command>

Commands
  download   fetch season CSVs from football-data.co.uk into data/raw/
  coverage   show matches and odds-source coverage per season
  backtest   tune decay, run the walk-forward backtest, write reports/summary.json
  plots      render reports/figures/backtest_summary.png
  all        backtest + plots
  predict    refresh data, predict upcoming fixtures -> predictions/pending/<round>/
"""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="premodel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    dl = sub.add_parser("download", help="fetch season CSVs into data/raw/")
    dl.add_argument("--overwrite", action="store_true", help="re-download existing files")
    sub.add_parser("coverage", help="matches and odds sources per season")
    sub.add_parser("backtest", help="tune decay + walk-forward backtest")
    sub.add_parser("plots", help="render reports/figures/backtest_summary.png")
    sub.add_parser("all", help="backtest + plots")
    pr = sub.add_parser("predict", help="predict upcoming fixtures")
    pr.add_argument("--no-refresh", action="store_true",
                    help="use the existing season and fixtures files (no download)")
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
    elif args.command == "predict":
        from premodel.forward import run_predict
        run_predict(refresh=not args.no_refresh)
    if args.command in ("plots", "all"):
        from premodel.plots import render
        render()


if __name__ == "__main__":
    main()
