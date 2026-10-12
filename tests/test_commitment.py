import json

import numpy as np
import pytest

from premodel.commitment import canonical_json

PAYLOAD = {
    "round_id": "E0-2026-10-17",
    "xi": 0.003,
    "git_dirty": False,
    "training": {"n_matches": 1140, "last_match_date": "2026-10-12"},
    "fixtures": [
        {"home": "Nott'm Forest", "away": "Man City", "p_home": 0.2, "p_draw": 0.25,
         "p_away": 0.55, "odds_open": {"home": 5.1, "draw": 4.0, "away": 1.67},
         "odds_source": "market_avg"},
        {"home": "Arsenal", "away": "Leeds", "p_home": 0.712345, "p_draw": 0.181818,
         "p_away": 0.105837, "odds_open": None, "odds_source": None},
    ],
}


def reorder(o):
    """Same content, reversed key insertion order at every level."""
    if isinstance(o, dict):
        return {k: reorder(o[k]) for k in reversed(list(o))}
    if isinstance(o, list):
        return [reorder(v) for v in o]
    return o


def test_canonical_json_is_byte_stable_across_runs_and_key_order():
    a = canonical_json(PAYLOAD)
    assert a == canonical_json(PAYLOAD)
    assert a == canonical_json(reorder(PAYLOAD))
    assert a == canonical_json(json.loads(json.dumps(PAYLOAD)))


def test_canonical_json_round_trips_through_json_loads():
    a = canonical_json(PAYLOAD)
    assert canonical_json(json.loads(a)) == a


def test_canonical_json_layout():
    text = canonical_json(PAYLOAD).decode("utf-8")
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert "\r" not in text
    assert all(line == line.rstrip() for line in text.split("\n"))
    assert '"xi": 0.003000' in text
    assert '"p_home": 0.200000' in text
    assert '"n_matches": 1140' in text and '"git_dirty": false' in text
    keys = [line.strip().split('"')[1] for line in text.split("\n") if line.startswith('  "')]
    assert keys == sorted(keys)


def test_canonical_json_scalars():
    assert canonical_json(-0.0) == b"0.000000\n"
    assert canonical_json(np.float64(0.1234565)) == canonical_json(0.1234565)
    assert canonical_json(np.int64(3)) == b"3\n"
    assert canonical_json(np.bool_(True)) == b"true\n"
    assert canonical_json("Málaga") == '"Málaga"\n'.encode()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "x"}, {"a": {1, 2}}])
def test_canonical_json_rejects_unserialisable(bad):
    with pytest.raises((ValueError, TypeError)):
        canonical_json(bad)
