"""Canonical serialisation of a prediction round, so the commitment is reproducible.

The bytes of predictions.json are what gets hashed, so the same predictions must
always serialise to exactly the same bytes:
- object keys sorted, 2-space indent, "\\n" line endings, one trailing newline,
  no trailing whitespace, UTF-8;
- every float written with exactly 6 decimals (0.4521 -> 0.452100), NaN/inf rejected.
Reading the file back with `json.loads` and re-serialising gives identical bytes.

Commitment scheme
- salt.hex:   32 random bytes as 64 lowercase hex chars + "\n";
- commitment: sha256(salt bytes + predictions.json bytes), lowercase hex;
- memo:       "premodel:v1:<round_id>:<commitment>" posted through the SPL Memo program
              ("premodel-test:v1:..." for dry runs; the track record ignores those).
The salt stays private until reveal, so the memo gives nothing away beforehand.

Everything here is pure logic: the chain is reached only through a `ChainClient`
(premodel.onchain.SolanaClient when live, a fake in tests).
"""
import hashlib
import json
import math
import re
import secrets
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from premodel import config

FLOAT_DECIMALS = 6
PRED_FILE = "predictions.json"
SALT_FILE = "salt.hex"
COMMIT_FILE = "commit.json"
PREFIX, TEST_PREFIX, MEMO_VERSION = "premodel", "premodel-test", "v1"
_MEMO_RE = re.compile(r"(premodel|premodel-test):v1:([A-Z0-9]+-\d{4}-\d{2}-\d{2}):([0-9a-f]{64})")
_SALT_RE = re.compile(rb"[0-9a-f]{64}\n")
_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _scalar(o) -> str:
    if o is None:
        return "null"
    if isinstance(o, (bool, np.bool_)):  # before int: bool is a subclass of int
        return "true" if o else "false"
    if isinstance(o, (int, np.integer)):
        return str(int(o))
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if not math.isfinite(x):
            raise ValueError(f"cannot serialise non-finite float {x!r}")
        s = f"{x:.{FLOAT_DECIMALS}f}"
        return "0.000000" if s == "-0.000000" else s
    if isinstance(o, str):
        return json.dumps(o, ensure_ascii=False)
    raise TypeError(f"cannot serialise {type(o).__name__} canonically")


def _encode(o, level: int) -> str:
    pad, end = "  " * (level + 1), "  " * level
    if isinstance(o, dict):
        if not o:
            return "{}"
        if not all(isinstance(k, str) for k in o):
            raise TypeError("object keys must be strings")
        items = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {_encode(o[k], level + 1)}"
                 for k in sorted(o)]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    if isinstance(o, (list, tuple)):
        if not o:
            return "[]"
        return "[\n" + ",\n".join(pad + _encode(v, level + 1) for v in o) + "\n" + end + "]"
    return _scalar(o)


def canonical_json(obj) -> bytes:
    """Deterministic UTF-8 JSON bytes for `obj` (dicts, lists, str, int, float, bool, None)."""
    return (_encode(obj, 0) + "\n").encode("utf-8")


# --- salt, hash, memo -------------------------------------------------------------------

def new_salt() -> str:
    return secrets.token_hex(32)


def read_salt(path: Path) -> str:
    raw = path.read_bytes()
    if not _SALT_RE.fullmatch(raw):
        raise ValueError(f"{path.name} must be 64 lowercase hex characters and one newline")
    return raw[:64].decode("ascii")


def compute_commitment(salt_hex: str, predictions_bytes: bytes) -> str:
    return hashlib.sha256(bytes.fromhex(salt_hex) + predictions_bytes).hexdigest()


@dataclass(frozen=True)
class Memo:
    test: bool
    round_id: str
    commitment: str


def build_memo(round_id: str, commitment: str, test: bool = False) -> str:
    text = f"{TEST_PREFIX if test else PREFIX}:{MEMO_VERSION}:{round_id}:{commitment}"
    if parse_memo(text) is None:
        raise ValueError(f"malformed memo {text!r}")
    return text


def parse_memo(text: str) -> Memo | None:
    m = _MEMO_RE.fullmatch(text)
    return Memo(m[1] == TEST_PREFIX, m[2], m[3]) if m else None


# --- chain interface --------------------------------------------------------------------

@dataclass(frozen=True)
class TxInfo:
    """What the commitment logic needs from a confirmed transaction."""
    signature: str
    slot: int
    block_time: int | None      # unix seconds, as reported by the cluster
    succeeded: bool
    fee_payer: str
    signers: tuple[str, ...]
    memos: tuple[str, ...]      # UTF-8 text of every Memo-program instruction


@dataclass(frozen=True)
class Signer:
    path: Path
    pubkey: str
    keypair: object = field(repr=False)  # opaque to this module; never printed


class ChainClient(Protocol):
    def genesis_hash(self) -> str: ...
    def send_memo(self, memo: str, signer: Signer) -> str: ...  # returns once confirmed
    def get_transaction(self, signature: str) -> TxInfo | None: ...
    def signatures_for_address(self, address: str) -> list[str]: ...  # full history


def signed_txs(client: ChainClient, address: str) -> Iterator[TxInfo]:
    """Successful transactions paid for and signed by `address` alone.

    Anyone can include an address as a non-signing account, so only transactions the
    wallet actually signed can count towards (or against) its record.
    """
    for sig in client.signatures_for_address(address):
        tx = client.get_transaction(sig)
        if tx and tx.succeeded and tx.fee_payer == address and set(tx.signers) == {address}:
            yield tx


def round_memos(client: ChainClient, address: str, round_id: str,
                test: bool = False) -> list[tuple[TxInfo, Memo]]:
    out = []
    for tx in signed_txs(client, address):
        for text in tx.memos:
            m = parse_memo(text)
            if m and m.test == test and m.round_id == round_id:
                out.append((tx, m))
    return out


def require_devnet(client: ChainClient):
    """Hard guard: there is no mainnet path. Raises unless the RPC node is devnet."""
    genesis = client.genesis_hash()
    if genesis != config.DEVNET_GENESIS_HASH:
        raise CommitRefused(f"RPC genesis hash {genesis} is not devnet's "
                            f"({config.DEVNET_GENESIS_HASH}); refusing")


# --- rounds on disk ---------------------------------------------------------------------

def parse_time(s: str) -> datetime:
    return datetime.strptime(s, _TIME_FMT).replace(tzinfo=timezone.utc)


def format_time(t: datetime | int) -> str:
    if isinstance(t, int):
        t = datetime.fromtimestamp(t, tz=timezone.utc)
    return t.astimezone(timezone.utc).strftime(_TIME_FMT)


def kickoffs(pred: dict) -> list[datetime]:
    return sorted(parse_time(f["kickoff_utc"]) for f in pred["fixtures"])


def resolve_round(round_id: str | None, pending_dir: Path = config.PENDING_DIR,
                  committed: bool | None = None) -> Path:
    """The pending round folder to act on. Without `round_id` there must be exactly one
    candidate (only uncommitted rounds if committed=False, only committed if True)."""
    if round_id:
        d = pending_dir / round_id
        if not (d / PRED_FILE).exists():
            raise FileNotFoundError(f"no pending round {round_id} in {pending_dir}")
        return d
    cands = sorted(d for d in pending_dir.glob("*") if (d / PRED_FILE).exists()
                   and (committed is None or (d / COMMIT_FILE).exists() == committed))
    if len(cands) != 1:
        found = ", ".join(d.name for d in cands) or "none"
        raise ValueError(f"expected exactly one pending round, found {found}; pass --round")
    return cands[0]


def commit_record(tx: TxInfo, memo: str, commitment: str, signer: str) -> dict:
    return {
        "signature": tx.signature,
        "slot": tx.slot,
        "block_time": format_time(tx.block_time) if tx.block_time is not None else None,
        "commitment": commitment,
        "cluster": config.CLUSTER,
        "memo": memo,
        "signer": signer,
        "explorer_url": config.EXPLORER_TX_URL.format(sig=tx.signature),
    }


# --- commit -----------------------------------------------------------------------------

class CommitRefused(RuntimeError):
    """A safety check stopped the commit before anything was sent."""


def commit_round(round_dir: Path, client: ChainClient, signer: Signer, now: datetime, *,
                 git_dirty: bool, test: bool = False, allow_dirty: bool = False,
                 repo_root: Path = config.ROOT,
                 canonical_signer: str = config.CANONICAL_SIGNER,
                 test_dir: Path = config.PREDICTIONS_TEST_DIR) -> dict:
    """Salt, hash and post the memo for a pending round; write and return commit.json.

    A real commit writes salt.hex and commit.json into the pending round folder. A --test
    dry run works on a copy in `test_dir`, so the real round can still be committed.
    An existing salt.hex is reused, and the signer's history is checked first: if an
    earlier attempt already posted this exact commitment, it is adopted, not re-sent.
    """
    rid = round_dir.name
    raw = (round_dir / PRED_FILE).read_bytes()
    pred = json.loads(raw)
    out = test_dir / rid if test else round_dir

    if canonical_json(pred) != raw:
        raise CommitRefused(f"{PRED_FILE} is not canonical JSON; regenerate it with predict")
    if pred.get("round_id") != rid:
        raise CommitRefused(f"{PRED_FILE} round_id {pred.get('round_id')!r} != folder {rid}")
    if (out / COMMIT_FILE).exists():
        raise CommitRefused(f"{out / COMMIT_FILE} already exists; round {rid} is committed")
    first = kickoffs(pred)[0]
    if now >= first:
        raise CommitRefused(f"now ({format_time(now)}) is not before the first kickoff "
                            f"({format_time(first)}); a late commitment proves nothing")
    if signer.path.expanduser().resolve().is_relative_to(repo_root.resolve()):
        raise CommitRefused(f"keypair {signer.path} is inside the repository; move it out "
                            "(e.g. ~/.config/solana/)")
    if (git_dirty or pred.get("git_dirty")) and not allow_dirty:
        raise CommitRefused("src/ or pyproject.toml has uncommitted changes (or did when the "
                            "predictions were made); commit first or pass --allow-dirty")
    if not test and not canonical_signer:
        raise CommitRefused("config.CANONICAL_SIGNER is not set")
    if not test and signer.pubkey != canonical_signer:
        raise CommitRefused(f"signer {signer.pubkey} is not CANONICAL_SIGNER "
                            f"{canonical_signer}")
    require_devnet(client)

    out.mkdir(parents=True, exist_ok=True)
    if test:
        (out / PRED_FILE).write_bytes(raw)
    salt_path = out / SALT_FILE
    if not salt_path.exists():
        salt_path.write_text(new_salt() + "\n", encoding="ascii")
    commitment = compute_commitment(read_salt(salt_path), raw)
    memo = build_memo(rid, commitment, test)

    earlier = round_memos(client, signer.pubkey, rid, test)
    if earlier:
        if len(earlier) > 1 or earlier[0][1].commitment != commitment:
            sigs = ", ".join(tx.signature for tx, _ in earlier)
            raise CommitRefused(f"round {rid} already has on-chain commitment(s) that do not "
                                f"match the local files: {sigs}")
        tx = earlier[0][0]
    else:
        sig = client.send_memo(memo, signer)
        tx = client.get_transaction(sig)
        if tx is None or not tx.succeeded or tx.memos != (memo,):
            raise RuntimeError(f"transaction {sig} was sent but is missing, failed or does "
                               "not carry the memo; inspect it before retrying")
    record = commit_record(tx, memo, commitment, signer.pubkey)
    (out / COMMIT_FILE).write_bytes(canonical_json(record))
    return record


# --- verify -----------------------------------------------------------------------------

class _Fail(Exception):
    pass


def verify_round(round_dir: Path, client: ChainClient,
                 signer: str = config.CANONICAL_SIGNER, test: bool = False) -> tuple[bool, str]:
    """(passed, reason). Recomputes the commitment from the folder's files and checks it
    against the on-chain memo, signer, timing and the signer's full history."""
    try:
        return True, _verify(round_dir, client, signer, test)
    except _Fail as e:
        return False, str(e)


def _verify(round_dir: Path, client: ChainClient, signer: str, test: bool) -> str:
    rid = round_dir.name
    if not signer:
        raise _Fail("CANONICAL_SIGNER is not set")
    for name in (PRED_FILE, SALT_FILE, COMMIT_FILE):
        if not (round_dir / name).exists():
            raise _Fail(f"{round_dir / name} is missing")
    raw = (round_dir / PRED_FILE).read_bytes()
    try:
        pred = json.loads(raw)
        salt = read_salt(round_dir / SALT_FILE)
        rec = json.loads((round_dir / COMMIT_FILE).read_bytes())
        sig = rec["signature"]
        first = kickoffs(pred)[0]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise _Fail(f"unreadable round files: {e}") from None
    if pred.get("round_id") != rid:
        raise _Fail(f"{PRED_FILE} round_id {pred.get('round_id')!r} != folder {rid}")
    commitment = compute_commitment(salt, raw)

    tx = client.get_transaction(sig)
    if tx is None:
        raise _Fail(f"transaction {sig} not found on {config.CLUSTER}")
    if not tx.succeeded:
        raise _Fail(f"transaction {sig} failed")
    if len(tx.memos) != 1:
        raise _Fail(f"transaction has {len(tx.memos)} memos, expected exactly one")
    m = parse_memo(tx.memos[0])
    if m is None or m.test != test:
        raise _Fail(f"memo {tx.memos[0]!r} is not a {TEST_PREFIX if test else PREFIX}:"
                    f"{MEMO_VERSION} commitment")
    if m.round_id != rid:
        raise _Fail(f"memo is for round {m.round_id}, folder is {rid}")
    if m.commitment != commitment:
        raise _Fail(f"files hash to {commitment} but the memo commits to {m.commitment}: "
                    f"{PRED_FILE} or {SALT_FILE} differs from what was committed")
    if tx.fee_payer != signer or set(tx.signers) != {signer}:
        raise _Fail(f"transaction signed by {', '.join(tx.signers)} (fee payer "
                    f"{tx.fee_payer}), expected {signer}")
    if tx.block_time is None:
        raise _Fail("transaction has no block time")
    if tx.block_time >= first.timestamp():
        raise _Fail(f"committed at {format_time(tx.block_time)}, not before the first "
                    f"kickoff {format_time(first)}")
    expected = commit_record(tx, tx.memos[0], commitment, signer)
    wrong = sorted(k for k in expected if k != "explorer_url" and rec.get(k) != expected[k])
    if wrong:
        raise _Fail(f"{COMMIT_FILE} disagrees with the chain on: {', '.join(wrong)}")
    others = [t.signature for t, _ in round_memos(client, signer, rid, test)
              if t.signature != sig]
    if others:
        raise _Fail(f"signer posted other commitments for round {rid}: {', '.join(others)}")
    return (f"round {rid} committed {commitment[:16]}... in slot {tx.slot} at "
            f"{format_time(tx.block_time)}, before first kickoff {format_time(first)}")


# --- audit ------------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditRow:
    round_id: str
    signature: str
    block_time: str | None
    status: str  # see AUDIT_OK / audit()


AWAITING, OVERDUE = "awaiting reveal", "overdue"
AUDIT_OK = frozenset({"revealed", AWAITING})


def reveal_deadline(round_id: str, days: int = config.REVEAL_DEADLINE_DAYS) -> datetime:
    """00:00 UTC on the round_id date + `days`: needs nothing but the memo."""
    d = datetime.strptime(round_id[-10:], "%Y-%m-%d")  # "E0-2026-10-17" -> 2026-10-17
    return d.replace(tzinfo=timezone.utc) + timedelta(days=days)


def audit_ok(rows: list[AuditRow]) -> bool:
    return all(r.status in AUDIT_OK for r in rows)


def audit(client: ChainClient, now: datetime, signer: str = config.CANONICAL_SIGNER,
          revealed_dir: Path = config.REVEALED_DIR,
          deadline_days: int = config.REVEAL_DEADLINE_DAYS) -> list[AuditRow]:
    """Every premodel:v1 memo the signer has posted, checked against revealed/ folders.

    Statuses: revealed; awaiting reveal (not revealed, deadline not yet reached);
    overdue (not revealed by the deadline); mismatch (revealed files do not hash to the
    memo); duplicate (several memos for one round); malformed (unparseable premodel:
    memo). Test-prefix memos are ignored. Anything except revealed / awaiting reveal
    counts against the record: a commitment that is never revealed could be a hidden
    bad prediction.
    """
    if not signer:
        raise ValueError("CANONICAL_SIGNER is not set")
    by_round: dict[str, list[tuple[TxInfo, Memo]]] = defaultdict(list)
    rows = []
    for tx in signed_txs(client, signer):
        bt = format_time(tx.block_time) if tx.block_time is not None else None
        for text in tx.memos:
            m = parse_memo(text)
            if m is None:
                if text.startswith(f"{PREFIX}:"):
                    rows.append(AuditRow(text[:40], tx.signature, bt, "malformed"))
            elif not m.test:
                by_round[m.round_id].append((tx, m))
    for rid, items in by_round.items():
        for tx, m in items:
            bt = format_time(tx.block_time) if tx.block_time is not None else None
            rows.append(AuditRow(rid, tx.signature, bt,
                                 "duplicate" if len(items) > 1
                                 else _reveal_status(revealed_dir / rid, tx, m, now,
                                                     deadline_days)))
    return sorted(rows, key=lambda r: (r.round_id, r.block_time or "", r.signature))


def _reveal_status(d: Path, tx: TxInfo, m: Memo, now: datetime, deadline_days: int) -> str:
    if not d.is_dir():
        return AWAITING if now < reveal_deadline(m.round_id, deadline_days) else OVERDUE
    try:
        rec = json.loads((d / COMMIT_FILE).read_bytes())
        ok = (rec["signature"] == tx.signature and m.commitment == compute_commitment(
            read_salt(d / SALT_FILE), (d / PRED_FILE).read_bytes()))
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    return "revealed" if ok else "mismatch"


# --- schemas ----------------------------------------------------------------------------

V1_MODEL = "dc_v1"  # schema v1 rounds hold only dc_v1 probabilities, at fixture level


def fixture_probs(pred: dict) -> list[dict]:
    """Per fixture: {model: (p_home, p_draw, p_away)}, for schema v1 and v2 rounds."""
    schema = pred.get("schema", "")
    out = []
    for f in pred["fixtures"]:
        if schema.endswith("/v1"):
            out.append({V1_MODEL: (f["p_home"], f["p_draw"], f["p_away"])})
        elif schema.endswith("/v2"):
            out.append({name: (b["p_home"], b["p_draw"], b["p_away"])
                        for name, b in f["probs"].items()})
        else:
            raise ValueError(f"unknown predictions schema {schema!r}")
    return out


def primary_model(pred: dict) -> str:
    return pred.get("primary_model", V1_MODEL)
