"""Solana devnet access: the ONLY module that imports solana / solders.

Install with  pip install -e ".[onchain]".  Everything here is a thin adapter:
the commitment logic (premodel.commitment) talks to the chain through the
`ChainClient` interface, which `SolanaClient` implements for a live RPC node.

Private keys never leave this module: they are read from a Solana-CLI-style
keypair file (JSON array of 64 bytes, 0600) and only public keys are printed.
"""
import asyncio
import json
import os
from pathlib import Path

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.instruction import Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import Transaction

from premodel import config
from premodel.commitment import Signer, TxInfo

LAMPORTS_PER_SOL = 1_000_000_000
MEMO_PROGRAM = Pubkey.from_string(config.MEMO_PROGRAM_ID)


# --- keypairs ---------------------------------------------------------------------------

def keypair_path(test: bool = False) -> Path:
    env, default = (("SOLANA_TEST_KEYPAIR", config.TEST_KEYPAIR_PATH) if test
                    else ("SOLANA_KEYPAIR", config.KEYPAIR_PATH))
    return Path(os.environ.get(env) or default).expanduser()


def rpc_url() -> str:
    return os.environ.get("SOLANA_RPC_URL") or config.DEVNET_RPC_URL


def create_keypair(path: Path) -> bool:
    """Create a new keypair file with 0600 permissions if missing. True if created."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    kp = Keypair()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(list(bytes(kp)), f)
    return True


def load_signer(path: Path) -> Signer:
    if not path.exists():
        raise FileNotFoundError(f"no keypair at {path}; create one with `premodel wallet`")
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"keypair {path} is readable by other users; chmod 600 it")
    kp = Keypair.from_bytes(bytes(json.loads(path.read_text())))
    return Signer(path=path, pubkey=str(kp.pubkey()), keypair=kp)


# --- RPC client -------------------------------------------------------------------------

class SolanaClient:
    """Synchronous wrapper over solana-py's AsyncClient (the sync client was removed)."""

    def __init__(self, url: str | None = None, timeout: float = 30.0):
        self.url = url or rpc_url()
        self.timeout = timeout

    def _run(self, fn):
        async def go():
            async with AsyncClient(self.url, commitment=Confirmed, timeout=self.timeout) as c:
                return await fn(c)
        return asyncio.run(go())

    def genesis_hash(self) -> str:
        return str(self._run(lambda c: c.get_genesis_hash()).value)

    def balance(self, pubkey: str) -> int:
        return self._run(lambda c: c.get_balance(Pubkey.from_string(pubkey))).value

    def airdrop(self, pubkey: str, lamports: int = LAMPORTS_PER_SOL) -> str:
        async def go(c):
            sig = (await c.request_airdrop(Pubkey.from_string(pubkey), lamports)).value
            await c.confirm_transaction(sig, Confirmed)
            return str(sig)
        return self._run(go)

    def send_memo(self, memo: str, signer: Signer) -> str:
        """Post one Memo-program instruction paid and signed by `signer`; wait for confirmed."""
        kp: Keypair = signer.keypair
        ix = Instruction(MEMO_PROGRAM, memo.encode("utf-8"), [])

        async def go(c):
            bh = (await c.get_latest_blockhash()).value
            msg = Message.new_with_blockhash([ix], kp.pubkey(), bh.blockhash)
            tx = Transaction([kp], msg, bh.blockhash)
            sig = (await c.send_raw_transaction(bytes(tx))).value  # preflight at confirmed
            await c.confirm_transaction(sig, Confirmed,
                                        last_valid_block_height=bh.last_valid_block_height)
            return str(sig)
        return self._run(go)

    def get_transaction(self, signature: str) -> TxInfo | None:
        resp = self._run(lambda c: c.get_transaction(
            Signature.from_string(signature), encoding="base64", commitment=Confirmed,
            max_supported_transaction_version=0))
        return tx_info(signature, resp.value) if resp.value is not None else None

    def signatures_for_address(self, address: str) -> list[str]:
        """Full confirmed history of `address`, newest first (pages of 1000)."""
        async def go(c):
            out, before = [], None
            while True:
                page = (await c.get_signatures_for_address(
                    Pubkey.from_string(address), before=before, limit=1000)).value
                out += [str(s.signature) for s in page]
                if len(page) < 1000:
                    return out
                before = page[-1].signature
        return self._run(go)


def tx_info(signature: str, value) -> TxInfo:
    """Reduce an EncodedConfirmedTransactionWithStatusMeta (base64) to TxInfo.

    Only top-level Memo instructions count; premodel never posts memos via CPI.
    """
    vtx = value.transaction.transaction
    meta = value.transaction.meta
    msg = vtx.message
    keys = [str(k) for k in msg.account_keys]
    memos = tuple(bytes(ix.data).decode("utf-8", errors="replace")
                  for ix in msg.instructions
                  if keys[ix.program_id_index] == config.MEMO_PROGRAM_ID)
    return TxInfo(
        signature=signature,
        slot=value.slot,
        block_time=value.block_time,
        succeeded=meta is not None and meta.err is None,
        fee_payer=keys[0],
        signers=tuple(keys[:msg.header.num_required_signatures]),
        memos=memos,
    )
