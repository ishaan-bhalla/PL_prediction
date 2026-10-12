"""In-memory stand-in for premodel.onchain.SolanaClient. No network."""
from collections import defaultdict
from pathlib import Path

from premodel import config
from premodel.commitment import Signer, TxInfo

MAINNET_GENESIS = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"


def make_signer(pubkey: str, path: Path | None = None) -> Signer:
    return Signer(path=path or Path("/nonexistent/outside/repo/key.json"), pubkey=pubkey,
                  keypair=object())


class FakeSolanaClient:
    def __init__(self, block_time: int, genesis: str = config.DEVNET_GENESIS_HASH):
        self.genesis = genesis
        self.block_time = block_time  # block time given to the next transaction
        self.txs: dict[str, TxInfo] = {}
        self.history: dict[str, list[str]] = defaultdict(list)  # newest first
        self.sent: list[str] = []

    def genesis_hash(self) -> str:
        return self.genesis

    def send_memo(self, memo: str, signer: Signer) -> str:
        self.sent.append(memo)
        return self.post(memo, payer=signer.pubkey)

    def post(self, *memos: str, payer: str, signers: tuple[str, ...] | None = None,
             accounts: tuple[str, ...] = (), succeeded: bool = True,
             block_time: int | None = None) -> str:
        """Record a transaction directly (also used to stage adversarial history)."""
        sig = f"sig{len(self.txs) + 1:04d}"
        signers = signers or (payer,)
        self.txs[sig] = TxInfo(sig, 1000 + len(self.txs), block_time or self.block_time,
                               succeeded, payer, signers, memos)
        for a in dict.fromkeys((payer, *signers, *accounts)):
            self.history[a].insert(0, sig)
        return sig

    def get_transaction(self, signature: str) -> TxInfo | None:
        return self.txs.get(signature)

    def signatures_for_address(self, address: str) -> list[str]:
        return list(self.history[address])
