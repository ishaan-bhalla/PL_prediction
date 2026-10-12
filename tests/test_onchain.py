"""Offline tests of the Solana adapter. Skipped if the onchain extra is not installed."""
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("solders")
pytest.importorskip("solana")

from solders.hash import Hash  # noqa: E402
from solders.instruction import AccountMeta, Instruction  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from solders.message import Message  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402
from solders.transaction import Transaction, VersionedTransaction  # noqa: E402

from fake_chain import MAINNET_GENESIS, FakeSolanaClient  # noqa: E402
from premodel import __main__ as cli  # noqa: E402
from premodel import config, onchain  # noqa: E402


def encoded(tx: Transaction, err=None, block_time=1_800_000_000):
    """The shape of an EncodedConfirmedTransactionWithStatusMeta (base64) value."""
    return SimpleNamespace(slot=7, block_time=block_time, transaction=SimpleNamespace(
        transaction=VersionedTransaction.from_legacy(tx), meta=SimpleNamespace(err=err)))


def signed(ixs, *kps):
    bh = Hash.default()
    return Transaction(list(kps), Message.new_with_blockhash(ixs, kps[0].pubkey(), bh), bh)


def test_tx_info_extracts_payer_signers_and_top_level_memos():
    payer, other = Keypair(), Keypair()
    memo = Instruction(onchain.MEMO_PROGRAM, b"premodel:v1:x", [])
    noise = Instruction(Pubkey.default(), b"not a memo",
                        [AccountMeta(other.pubkey(), is_signer=True, is_writable=False)])
    info = onchain.tx_info("sig", encoded(signed([noise, memo], payer, other)))
    assert info.fee_payer == str(payer.pubkey())
    assert info.signers == (str(payer.pubkey()), str(other.pubkey()))
    assert info.memos == ("premodel:v1:x",)
    assert info.succeeded and info.slot == 7 and info.block_time == 1_800_000_000


def test_tx_info_marks_failed_transactions():
    kp = Keypair()
    tx = signed([Instruction(onchain.MEMO_PROGRAM, b"m", [])], kp)
    assert not onchain.tx_info("sig", encoded(tx, err="InstructionError")).succeeded


def test_keypair_created_0600_loaded_and_never_shown(tmp_path):
    path = tmp_path / "solana" / "k.json"
    assert onchain.create_keypair(path) and not onchain.create_keypair(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    signer = onchain.load_signer(path)
    secret = path.read_text()
    assert len(signer.pubkey) in (43, 44)
    assert secret not in repr(signer) and "keypair=" not in repr(signer)


def test_load_signer_refuses_group_readable_keypair(tmp_path):
    path = tmp_path / "k.json"
    onchain.create_keypair(path)
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        onchain.load_signer(path)


def test_keypair_and_rpc_defaults_and_env(monkeypatch, tmp_path):
    for var in ("SOLANA_KEYPAIR", "SOLANA_TEST_KEYPAIR", "SOLANA_RPC_URL"):
        monkeypatch.delenv(var, raising=False)
    assert onchain.keypair_path() == Path("~/.config/solana/premodel-devnet.json").expanduser()
    assert onchain.keypair_path(test=True).name == "premodel-devnet-test.json"
    assert onchain.rpc_url() == "https://api.devnet.solana.com"
    monkeypatch.setenv("SOLANA_TEST_KEYPAIR", str(tmp_path / "t.json"))
    assert onchain.keypair_path(test=True) == tmp_path / "t.json"


@pytest.mark.parametrize("command", [["wallet"], ["audit", "--signer", "x"],
                                     ["verify", "--round", "E0-2026-10-17", "--signer", "x"]])
def test_cli_refuses_non_devnet_rpc(monkeypatch, tmp_path, command):
    monkeypatch.setenv("SOLANA_KEYPAIR", str(tmp_path / "k.json"))
    monkeypatch.setattr(onchain, "SolanaClient",
                        lambda: FakeSolanaClient(0, genesis=MAINNET_GENESIS))
    with pytest.raises(SystemExit, match="not devnet"):
        cli.main(command)


def test_onchain_is_the_only_module_importing_solana():
    pattern = re.compile(r"^\s*(from|import)\s+(solana|solders)\b", re.M)
    users = sorted(p.name for p in (config.ROOT / "src" / "premodel").rglob("*.py")
                   if pattern.search(p.read_text()))
    assert users == ["onchain.py"]
