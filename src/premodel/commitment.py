"""Canonical serialisation of a prediction round, so the commitment is reproducible.

The bytes of predictions.json are what gets hashed, so the same predictions must
always serialise to exactly the same bytes:
- object keys sorted, 2-space indent, "\\n" line endings, one trailing newline,
  no trailing whitespace, UTF-8;
- every float written with exactly 6 decimals (0.4521 -> 0.452100), NaN/inf rejected.
Reading the file back with `json.loads` and re-serialising gives identical bytes.
"""
import json
import math

import numpy as np

FLOAT_DECIMALS = 6


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
