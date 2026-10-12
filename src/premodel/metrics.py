"""Proper scoring rules and calibration for 1X2 forecasts (lower is better)."""
import numpy as np


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score: respects the ordering home > draw > away."""
    onehot = np.eye(3)[y]
    cp, co = np.cumsum(p, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2))


def score_all(p: np.ndarray, y: np.ndarray) -> dict:
    return {"log_loss": log_loss(p, y), "brier": brier(p, y), "rps": rps(p, y)}


def calibration_table(p: np.ndarray, y: np.ndarray, outcome: int = 0, bins: int = 10,
                      min_count: int = 15) -> list[dict]:
    prob = p[:, outcome]
    hit = (y == outcome).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() >= min_count:
            rows.append({"pred": float(prob[mask].mean()),
                         "actual": float(hit[mask].mean()), "n": int(mask.sum())})
    return rows
