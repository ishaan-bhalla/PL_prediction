import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from fake_chain import MAINNET_GENESIS, FakeSolanaClient, make_signer
from premodel import commitment as cm
from premodel.commitment import canonical_json

CANON = "Canon1111111111111111111111111111111111111"
OTHER = "Other1111111111111111111111111111111111111"
RID = "E0-2026-10-17"
FIRST_KICKOFF = datetime(2026, 10, 17, 11, 30, tzinfo=timezone.utc)
NOW = FIRST_KICKOFF - timedelta(days=1)
PAYLOAD = {
    "round_id": RID,
    "git_dirty": False,
    "fixtures": [
        {"home": "T2", "away": "T3", "kickoff_utc": "2026-10-17T11:30:00Z",
         "p_home": 0.2, "p_draw": 0.25, "p_away": 0.55},
        {"home": "T0", "away": "T1", "kickoff_utc": "2026-10-18T15:30:00Z",
         "p_home": 0.712345, "p_draw": 0.181818, "p_away": 0.105837},
    ],
}


@pytest.fixture
def env(tmp_path):
    pending = tmp_path / "pending"
    rd = pending / RID
    rd.mkdir(parents=True)
    (rd / cm.PRED_FILE).write_bytes(canonical_json(PAYLOAD))
    client = FakeSolanaClient(block_time=int(NOW.timestamp()) + 30)
    return {"tmp": tmp_path, "round": rd, "client": client, "test_dir": tmp_path / "test",
            "revealed": tmp_path / "revealed"}


def commit(env, now=NOW, signer=CANON, **kw):
    kw = {"git_dirty": False, "repo_root": env["tmp"] / "repo",
          "canonical_signer": CANON, "test_dir": env["test_dir"], **kw}
    return cm.commit_round(env["round"], env["client"], make_signer(signer), now, **kw)


def rewrite(path, **changes):
    data = json.loads(path.read_bytes())
    data.update(changes)
    path.write_bytes(canonical_json(data))


# --- commitment -------------------------------------------------------------------------

@pytest.mark.parametrize("i", [0, 1])
@pytest.mark.parametrize("key", ["p_home", "p_draw", "p_away"])
def test_commitment_changes_when_any_probability_moves_by_1e6(i, key):
    salt = "ab" * 32
    base = cm.compute_commitment(salt, canonical_json(PAYLOAD))
    changed = json.loads(json.dumps(PAYLOAD))
    changed["fixtures"][i][key] = round(changed["fixtures"][i][key] + 1e-6, 6)
    assert cm.compute_commitment(salt, canonical_json(changed)) != base


def test_commitment_changes_with_salt_and_matches_spec():
    raw = canonical_json(PAYLOAD)
    a, b = "00" * 32, "00" * 31 + "01"
    assert cm.compute_commitment(a, raw) != cm.compute_commitment(b, raw)
    assert cm.compute_commitment(a, raw) == hashlib.sha256(bytes(32) + raw).hexdigest()


def test_memo_round_trip_and_strict_parsing():
    c = "f" * 64
    assert cm.build_memo(RID, c) == f"premodel:v1:{RID}:{c}"
    assert cm.parse_memo(cm.build_memo(RID, c, test=True)) == cm.Memo(True, RID, c)
    for bad in [f"premodel:v2:{RID}:{c}", f"premodel:v1:{RID}:{c.upper()}",
                f"premodel:v1:{RID}:{c} ", f"premodel:v1:{RID}:{c[:-1]}", "hello"]:
        assert cm.parse_memo(bad) is None


def test_salt_file_format(tmp_path):
    p = tmp_path / "salt.hex"
    p.write_text(cm.new_salt() + "\n")
    assert len(cm.read_salt(p)) == 64
    for bad in ["ab" * 32, "AB" * 32 + "\n", "ab" * 32 + "\n\n"]:
        p.write_text(bad)
        with pytest.raises(ValueError):
            cm.read_salt(p)


# --- commit -----------------------------------------------------------------------------

def test_commit_writes_salt_and_canonical_commit_json(env):
    rec = commit(env)
    rd = env["round"]
    salt = cm.read_salt(rd / cm.SALT_FILE)
    expected = cm.compute_commitment(salt, (rd / cm.PRED_FILE).read_bytes())
    assert rec["commitment"] == expected
    assert rec["memo"] == f"premodel:v1:{RID}:{expected}"
    assert env["client"].sent == [rec["memo"]]
    assert rec["signer"] == CANON and rec["cluster"] == "devnet"
    assert rec["explorer_url"] == (
        f"https://explorer.solana.com/tx/{rec['signature']}?cluster=devnet")
    assert rec["block_time"] == "2026-10-16T11:30:30Z"
    raw = (rd / cm.COMMIT_FILE).read_bytes()
    assert raw == canonical_json(rec) == canonical_json(json.loads(raw))


@pytest.mark.parametrize("now", [FIRST_KICKOFF, FIRST_KICKOFF + timedelta(minutes=1)])
def test_commit_refuses_at_or_after_first_kickoff(env, now):
    with pytest.raises(cm.CommitRefused, match="first kickoff"):
        commit(env, now=now)
    assert not env["client"].sent and not (env["round"] / cm.COMMIT_FILE).exists()


def test_commit_refuses_dirty_tree_unless_allowed(env):
    with pytest.raises(cm.CommitRefused, match="allow-dirty"):
        commit(env, git_dirty=True)
    assert not env["client"].sent
    commit(env, git_dirty=True, allow_dirty=True)
    assert len(env["client"].sent) == 1


def test_commit_refuses_predictions_made_from_dirty_code(env):
    rewrite(env["round"] / cm.PRED_FILE, git_dirty=True)
    with pytest.raises(cm.CommitRefused, match="allow-dirty"):
        commit(env)


def test_commit_refuses_keypair_inside_repo(env):
    repo = env["tmp"] / "repo"
    signer = make_signer(CANON, path=repo / "secrets" / "key.json")
    with pytest.raises(cm.CommitRefused, match="inside the repository"):
        cm.commit_round(env["round"], env["client"], signer, NOW, git_dirty=False,
                        repo_root=repo, canonical_signer=CANON)
    assert not env["client"].sent


def test_commit_refuses_non_devnet_genesis_hash(env):
    env["client"].genesis = MAINNET_GENESIS
    with pytest.raises(cm.CommitRefused, match="not devnet"):
        commit(env)
    assert not env["client"].sent and not (env["round"] / cm.SALT_FILE).exists()


def test_commit_refuses_twice_and_non_canonical_signer(env):
    with pytest.raises(cm.CommitRefused, match="CANONICAL_SIGNER"):
        commit(env, signer=OTHER)
    with pytest.raises(cm.CommitRefused, match="not set"):
        commit(env, canonical_signer="")
    commit(env)
    with pytest.raises(cm.CommitRefused, match="already exists"):
        commit(env)
    assert len(env["client"].sent) == 1


def test_commit_refuses_non_canonical_predictions_file(env):
    p = env["round"] / cm.PRED_FILE
    p.write_text(json.dumps(PAYLOAD))
    with pytest.raises(cm.CommitRefused, match="canonical"):
        commit(env)


def test_test_commit_uses_test_prefix_and_leaves_real_round_untouched(env):
    rec = commit(env, test=True, signer=OTHER)
    t = env["test_dir"] / RID
    assert rec["memo"].startswith(f"premodel-test:v1:{RID}:")
    assert (t / cm.COMMIT_FILE).exists() and (t / cm.SALT_FILE).exists()
    assert (t / cm.PRED_FILE).read_bytes() == (env["round"] / cm.PRED_FILE).read_bytes()
    assert sorted(p.name for p in env["round"].iterdir()) == [cm.PRED_FILE]
    assert cm.verify_round(t, env["client"], signer=OTHER, test=True)[0]
    real = commit(env)  # the real commit is still possible afterwards
    assert real["memo"].startswith("premodel:v1:") and real["commitment"] != rec["commitment"]


def test_commit_adopts_an_earlier_identical_memo_instead_of_resending(env):
    # Simulate a crash after the memo landed but before commit.json was written.
    commit(env)
    (env["round"] / cm.COMMIT_FILE).unlink()
    rec = commit(env)
    assert len(env["client"].sent) == 1
    assert rec["signature"] == "sig0001"
    assert cm.verify_round(env["round"], env["client"], signer=CANON)[0]


def test_commit_refuses_when_an_earlier_memo_does_not_match(env):
    commit(env)
    (env["round"] / cm.COMMIT_FILE).unlink()
    (env["round"] / cm.SALT_FILE).write_text(cm.new_salt() + "\n")
    with pytest.raises(cm.CommitRefused, match="do not match"):
        commit(env)
    assert len(env["client"].sent) == 1


# --- verify -----------------------------------------------------------------------------

def verify(env, **kw):
    return cm.verify_round(env["round"], env["client"], signer=kw.pop("signer", CANON), **kw)


def test_verify_passes_a_correct_round(env):
    commit(env)
    ok, reason = verify(env)
    assert ok, reason
    revealed = env["revealed"] / RID  # still passes once moved to revealed/
    shutil.copytree(env["round"], revealed)
    assert cm.verify_round(revealed, env["client"], signer=CANON)[0]


def test_verify_fails_on_tampered_predictions(env):
    commit(env)
    p = env["round"] / cm.PRED_FILE
    data = json.loads(p.read_bytes())
    data["fixtures"][0]["p_home"], data["fixtures"][0]["p_draw"] = 0.200001, 0.249999
    p.write_bytes(canonical_json(data))
    ok, reason = verify(env)
    assert not ok and "differs from what was committed" in reason


def test_verify_fails_on_wrong_salt(env):
    commit(env)
    (env["round"] / cm.SALT_FILE).write_text(cm.new_salt() + "\n")
    ok, reason = verify(env)
    assert not ok and "differs from what was committed" in reason


@pytest.mark.parametrize("memos, match", [
    (("premodel:v1:E0-2026-10-24:" + "a" * 64,), "is for round E0-2026-10-24"),
    (("premodel-test:v1:E0-2026-10-17:" + "a" * 64,), "is not a premodel:v1"),
    (("premodel:v1:E0-2026-10-17:" + "a" * 64,), "memo commits to"),
    (("premodel:v1:E0-2026-10-17:" + "a" * 64, "extra"), "expected exactly one"),
    (("hello",), "is not a premodel:v1"),
])
def test_verify_fails_on_memo_mismatch(env, memos, match):
    commit(env)
    sig = env["client"].post(*memos, payer=CANON)
    rewrite(env["round"] / cm.COMMIT_FILE, signature=sig)
    ok, reason = verify(env)
    assert not ok and match in reason, reason


def test_verify_fails_when_block_time_is_after_first_kickoff(env):
    env["client"].block_time = int(FIRST_KICKOFF.timestamp()) + 5  # confirmed too late
    commit(env)
    ok, reason = verify(env)
    assert not ok and "not before the first kickoff" in reason


def test_verify_fails_on_wrong_signer(env):
    commit(env, signer=OTHER, canonical_signer=OTHER)
    ok, reason = verify(env)  # expects CANON
    assert not ok and "expected " + CANON in reason


@pytest.mark.parametrize("same_hash", [True, False])
def test_verify_fails_on_duplicate_commitment_for_the_same_round(env, same_hash):
    rec = commit(env)
    c = rec["commitment"] if same_hash else "b" * 64
    env["client"].post(cm.build_memo(RID, c), payer=CANON)
    ok, reason = verify(env)
    assert not ok and "other commitments for round" in reason


def test_verify_ignores_memos_in_transactions_the_signer_did_not_sign(env):
    commit(env)
    # A stranger copies a memo into a tx that merely references the canonical address.
    env["client"].post(cm.build_memo(RID, "b" * 64), payer=OTHER, accounts=(CANON,))
    assert verify(env)[0]


def test_verify_fails_on_failed_or_missing_tx_and_missing_files(env):
    commit(env)
    rewrite(env["round"] / cm.COMMIT_FILE, signature="nope")
    assert "not found" in verify(env)[1]
    (env["round"] / cm.SALT_FILE).unlink()
    assert "salt.hex is missing" in verify(env)[1]


def test_verify_fails_when_commit_json_disagrees_with_chain(env):
    commit(env)
    rewrite(env["round"] / cm.COMMIT_FILE, block_time="2026-10-10T00:00:00Z")
    ok, reason = verify(env)
    assert not ok and "disagrees with the chain on: block_time" in reason


# --- audit ------------------------------------------------------------------------------

DEADLINE = datetime(2026, 10, 27, tzinfo=timezone.utc)  # RID date + REVEAL_DEADLINE_DAYS


def run_audit(env, now=NOW):
    rows = cm.audit(env["client"], now, signer=CANON, revealed_dir=env["revealed"])
    return [(r.round_id, r.status) for r in rows], cm.audit_ok(rows)


def test_reveal_deadline_comes_from_the_round_id_alone():
    assert cm.config.REVEAL_DEADLINE_DAYS == 10
    assert cm.reveal_deadline(RID) == DEADLINE


@pytest.mark.parametrize("now, status, ok", [
    (NOW, "awaiting reveal", True),
    (DEADLINE - timedelta(seconds=1), "awaiting reveal", True),
    (DEADLINE, "overdue", False),
    (DEADLINE + timedelta(days=30), "overdue", False),
])
def test_audit_unrevealed_round_is_awaiting_then_overdue(env, now, status, ok):
    commit(env)
    assert run_audit(env, now) == ([(RID, status)], ok)


def test_audit_ignores_test_memos_and_passes_once_revealed(env):
    commit(env)
    commit(env, test=True)
    env["client"].post(cm.build_memo("E0-2026-10-24", "c" * 64, test=True), payer=CANON)
    assert run_audit(env, DEADLINE) == ([(RID, "overdue")], False)

    shutil.copytree(env["round"], env["revealed"] / RID)
    assert run_audit(env, DEADLINE) == ([(RID, "revealed")], True)


def test_audit_flags_mismatched_reveals_duplicates_and_malformed_memos(env):
    commit(env)
    shutil.copytree(env["round"], env["revealed"] / RID)
    (env["revealed"] / RID / cm.SALT_FILE).write_text(cm.new_salt() + "\n")
    env["client"].post("premodel:v1:garbage", payer=CANON)
    env["client"].post(cm.build_memo("E0-2026-10-24", "d" * 64), payer=CANON)
    env["client"].post(cm.build_memo("E0-2026-10-24", "e" * 64), payer=CANON)
    env["client"].post(cm.build_memo("E0-2026-10-31", "f" * 64), payer=CANON,
                       succeeded=False)  # failed txs post nothing
    rows, ok = run_audit(env)
    assert not ok
    assert sorted(rows) == [
        (RID, "mismatch"), ("E0-2026-10-24", "duplicate"), ("E0-2026-10-24", "duplicate"),
        ("premodel:v1:garbage", "malformed")]


# --- round selection --------------------------------------------------------------------

def test_resolve_round_defaults_to_the_only_candidate(env):
    pending = env["round"].parent
    assert cm.resolve_round(None, pending) == env["round"]
    other = pending / "E0-2026-10-24"
    other.mkdir()
    (other / cm.PRED_FILE).write_text("{}")
    with pytest.raises(ValueError, match="pass --round"):
        cm.resolve_round(None, pending)
    (env["round"] / cm.COMMIT_FILE).write_text("{}")
    assert cm.resolve_round(None, pending, committed=False) == other
    assert cm.resolve_round(None, pending, committed=True) == env["round"]
    assert cm.resolve_round("E0-2026-10-24", pending) == other
    with pytest.raises(FileNotFoundError):
        cm.resolve_round("E0-2099-01-01", pending)
