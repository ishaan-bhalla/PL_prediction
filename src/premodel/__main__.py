"""Command-line entry point:  python -m premodel <command>

Commands
  download   fetch season CSVs from football-data.co.uk into data/raw/
  coverage   show matches and odds-source coverage per season
  backtest   tune decay, run the walk-forward backtest, write reports/summary.json
  plots      render reports/figures/backtest_summary.png
  all        backtest + plots
  predict    refresh data, predict upcoming fixtures -> predictions/pending/<round>/
  wallet     create the devnet keypair if missing, show its pubkey, request an airdrop
  commit     post the round's salted hash as a devnet memo -> commit.json
  verify     check a round's files against its on-chain commitment (PASS / FAIL)
  audit      list every premodel:v1 memo by CANONICAL_SIGNER and flag unrevealed ones
  reveal     move a finished, committed round to predictions/revealed/ (commit it to git)
  score      score revealed rounds -> predictions/track_record.csv, print the summary
  experiment run a model experiment, e.g. `experiment shrinkage`
              -> reports/experiments/<name>/

On-chain commands need  pip install -e ".[onchain]"  and only ever talk to devnet.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

from premodel import config


def _onchain():
    try:
        from premodel import onchain
    except ImportError as e:
        raise SystemExit(f'{e}\ninstall the on-chain extra:  pip install -e ".[onchain]"') from None
    return onchain


def _client():
    from premodel.commitment import CommitRefused, require_devnet
    client = _onchain().SolanaClient()
    try:
        require_devnet(client)
    except CommitRefused as e:
        raise SystemExit(str(e)) from None
    return client


def cmd_wallet(args):
    onchain = _onchain()
    path = onchain.keypair_path(args.test)
    if onchain.create_keypair(path):
        print(f"created {path} (0600)")
    signer = onchain.load_signer(path)
    print(f"pubkey  {signer.pubkey}")
    client = _client()
    sol = client.balance(signer.pubkey) / onchain.LAMPORTS_PER_SOL
    print(f"balance {sol:.4f} SOL (devnet)")
    if sol >= 1:
        print("balance >= 1 SOL; skipping airdrop")
        return
    try:
        print(f"airdrop {client.airdrop(signer.pubkey)}")
    except Exception as e:  # the public faucet is often rate-limited
        print(f"airdrop failed ({type(e).__name__}: {e}); try https://faucet.solana.com")


def cmd_commit(args):
    from premodel.commitment import (
        PRED_FILE,
        CommitRefused,
        commit_round,
        kickoffs,
        parse_time,
        resolve_round,
    )
    from premodel.forward import git_info
    onchain = _onchain()
    try:
        round_dir = resolve_round(args.round, committed=False)
        signer = onchain.load_signer(onchain.keypair_path(args.test))
        rec = commit_round(round_dir, _client(), signer, datetime.now(timezone.utc),
                           git_dirty=git_info()[1], test=args.test,
                           allow_dirty=args.allow_dirty)
    except (CommitRefused, ValueError, FileNotFoundError, PermissionError) as e:
        raise SystemExit(f"commit refused: {e}") from None
    print(f"committed {round_dir.name}{' (TEST)' if args.test else ''}")
    for k in ("memo", "signer", "signature", "slot", "block_time", "explorer_url"):
        print(f"  {k:<12} {rec[k]}")
    first = kickoffs(json.loads((round_dir / PRED_FILE).read_bytes()))[0]
    if rec["block_time"] is None or parse_time(rec["block_time"]) >= first:
        print("  WARNING: block time is missing or not before the first kickoff; "
              "this round will fail verify")


def cmd_verify(args):
    from premodel.commitment import verify_round
    if args.test:
        d = config.PREDICTIONS_TEST_DIR / args.round
        signer = args.signer or _onchain().load_signer(_onchain().keypair_path(True)).pubkey
    else:
        d = next((p for p in (config.REVEALED_DIR / args.round, config.PENDING_DIR / args.round)
                  if p.is_dir()), config.REVEALED_DIR / args.round)
        signer = args.signer or config.CANONICAL_SIGNER
    ok, reason = verify_round(d, _client(), signer=signer, test=args.test)
    print(f"PASS: {reason}" if ok else f"FAIL: {reason}")
    sys.exit(0 if ok else 1)


def cmd_reveal(args):
    from premodel.commitment import resolve_round
    from premodel.track_record import RevealRefused, reveal_round
    try:
        dest = reveal_round(resolve_round(args.round, committed=True),
                            datetime.now(timezone.utc), force=args.force)
    except (RevealRefused, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"reveal refused: {e}") from None
    print(f"revealed -> {dest.relative_to(config.ROOT)}  (commit this folder to git)")


def cmd_score(args):
    from premodel.data import download, load_matches
    from premodel.results import get_api_results, merge_results
    from premodel.track_record import run_score
    if not args.no_refresh:
        download([config.CURRENT_SEASON], overwrite=True)
    api, status = get_api_results(not args.no_refresh)
    print(status)
    results = merge_results(load_matches(), api.rows if api else None)
    try:
        run_score(results, datetime.now(timezone.utc), round_id=args.round)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"score: {e}") from None


def cmd_audit(args):
    from premodel.commitment import audit, audit_ok
    signer = args.signer or config.CANONICAL_SIGNER
    if not signer:
        raise SystemExit("CANONICAL_SIGNER is not set (or pass --signer)")
    rows = audit(_client(), datetime.now(timezone.utc), signer=signer)
    print(f"premodel:v1 memos by {signer}: {len(rows)}")
    for r in rows:
        print(f"  {r.round_id:<16} {r.status:<16} {r.block_time or '-':<21} {r.signature}")
    ok = audit_ok(rows)
    print("OK" if ok else "FLAGGED: see statuses above")
    sys.exit(0 if ok else 1)


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
    pr.add_argument("--allow-stale", action="store_true",
                    help=f"predict even if the newest played match is more than "
                         f"{config.STALE_DAYS} days before the first kickoff")
    wa = sub.add_parser("wallet", help="create/show the devnet keypair, request an airdrop")
    wa.add_argument("--test", action="store_true", help="use the test keypair")
    co = sub.add_parser("commit", help="commit a pending round on devnet")
    co.add_argument("--round", help="round id (default: the only uncommitted pending round)")
    co.add_argument("--test", action="store_true",
                    help="dry run: premodel-test memo, test keypair, predictions/test/")
    co.add_argument("--allow-dirty", action="store_true",
                    help="commit even if src/ or pyproject.toml has uncommitted changes")
    ve = sub.add_parser("verify", help="verify a round against its on-chain commitment")
    ve.add_argument("--round", required=True)
    ve.add_argument("--test", action="store_true", help="verify a --test dry run")
    ve.add_argument("--signer", help="expected signer pubkey (default: CANONICAL_SIGNER)")
    au = sub.add_parser("audit", help="flag committed rounds that were never revealed")
    au.add_argument("--signer", help="signer pubkey to audit (default: CANONICAL_SIGNER)")
    rv = sub.add_parser("reveal", help="move a finished committed round to revealed/")
    rv.add_argument("--round", help="round id (default: the only committed pending round)")
    rv.add_argument("--force", action="store_true",
                    help=f"reveal before last kickoff + {config.REVEAL_DELAY_HOURS}h")
    sc = sub.add_parser("score", help="score revealed rounds -> predictions/track_record.csv")
    sc.add_argument("--round", help="score only this revealed round (default: all)")
    sc.add_argument("--no-refresh", action="store_true",
                    help="use existing CSVs and the cached results API response")
    ex = sub.add_parser("experiment", help="run a model experiment")
    ex.add_argument("name", choices=["shrinkage"])
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
        from premodel.forward import NoFixturesError, StaleDataError, run_predict
        try:
            run_predict(refresh=not args.no_refresh, allow_stale=args.allow_stale)
        except (NoFixturesError, StaleDataError) as e:
            raise SystemExit(f"predict: {e}") from None
    elif args.command == "experiment":
        from premodel.experiments import shrinkage
        shrinkage.run()
    elif args.command in ("wallet", "commit", "verify", "audit", "reveal", "score"):
        {"wallet": cmd_wallet, "commit": cmd_commit, "verify": cmd_verify,
         "audit": cmd_audit, "reveal": cmd_reveal, "score": cmd_score}[args.command](args)
    if args.command in ("plots", "all"):
        from premodel.plots import render
        render()


if __name__ == "__main__":
    main()
