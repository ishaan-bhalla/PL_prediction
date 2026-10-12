"""Download and load football-data.co.uk Premier League CSVs.

Odds conventions used throughout the project
- "open"  = odds collected Friday/Tuesday afternoon before the match (the price
            you could realistically take when weekly predictions are made).
- "close" = closing odds just before kickoff (the market's sharpest estimate).
Pinnacle (PS*) is preferred because it is the sharpest bookmaker; where its
columns are missing (later 2025/26, 2026/27) the market average (Avg*) is used.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from premodel.config import DEFAULT_LEAGUE, DOWNLOAD_URL, RAW_DIR, SEASONS

RESULT_LABELS = {0: "H", 1: "D", 2: "A"}


def season_of(date: pd.Timestamp) -> str:
    """Season label in the same format as load_season, e.g. "2026/27"."""
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y}/{(y + 1) % 100:02d}"


def season_path(code: str, league: str = DEFAULT_LEAGUE, raw_dir: Path = RAW_DIR) -> Path:
    """data/raw/<league>_<code>.csv, e.g. E0_1415.csv."""
    return raw_dir / f"{league}_{code}.csv"


def download(seasons: list[str] = SEASONS, dest: Path = RAW_DIR, overwrite: bool = False,
             league: str = DEFAULT_LEAGUE):
    """Fetch <league>.csv for each season code (e.g. "1415") into data/raw/<league>_<code>.csv.

    league: football-data.co.uk code, "E0" (Premier League, default) or "E1" (Championship).
    """
    dest.mkdir(parents=True, exist_ok=True)
    for code in seasons:
        path = season_path(code, league, dest)
        if path.exists() and not overwrite:
            print(f"skip   {path.name} (exists)")
            continue
        urllib.request.urlretrieve(DOWNLOAD_URL.format(code=code, league=league), path)
        print(f"saved  {path.name}")


def _season_label(path: Path) -> str:
    code = path.stem.split("_")[1]  # "1415" -> "2014/15"
    return f"20{code[:2]}/{code[2:]}"


def _pick_odds(df: pd.DataFrame, preferred: list[str], fallback: list[str]):
    """Row-wise 3-way odds, preferring `preferred` columns, else `fallback`."""
    out = np.full((len(df), 3), np.nan)
    src = np.full(len(df), "", dtype=object)
    for cols, name in ((preferred, "pinnacle"), (fallback, "market_avg")):
        if all(c in df.columns for c in cols):
            vals = df[cols].to_numpy(dtype=float)
            fill = np.isnan(out).any(axis=1) & ~np.isnan(vals).any(axis=1)
            out[fill] = vals[fill]
            src[fill] = name
    return out, src


def open_odds(df: pd.DataFrame):
    """Pre-match 3-way odds: Pinnacle where available, else market average."""
    return _pick_odds(df, ["PSH", "PSD", "PSA"], ["AvgH", "AvgD", "AvgA"])


def close_odds(df: pd.DataFrame):
    """Closing 3-way odds: Pinnacle where available, else market average."""
    return _pick_odds(df, ["PSCH", "PSCD", "PSCA"], ["AvgCH", "AvgCD", "AvgCA"])


def load_season(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    open_odds_, open_src = open_odds(df)
    close_odds_, close_src = close_odds(df)
    return pd.DataFrame({
        "season": _season_label(path),
        "date": df["Date"].to_numpy(),
        "home": df["HomeTeam"].str.strip().to_numpy(),
        "away": df["AwayTeam"].str.strip().to_numpy(),
        "hg": df["FTHG"].to_numpy(),
        "ag": df["FTAG"].to_numpy(),
        "open_h": open_odds_[:, 0], "open_d": open_odds_[:, 1], "open_a": open_odds_[:, 2],
        "close_h": close_odds_[:, 0], "close_d": close_odds_[:, 1], "close_a": close_odds_[:, 2],
        "open_src": open_src, "close_src": close_src,
    })


def load_matches(raw_dir: Path = RAW_DIR, league: str = DEFAULT_LEAGUE) -> pd.DataFrame:
    """All played matches of one league, sorted by date, with result coded 0=H 1=D 2=A."""
    paths = sorted(raw_dir.glob(f"{league}_*.csv"))
    if not paths:
        flag = "" if league == DEFAULT_LEAGUE else f" --league {league}"
        raise FileNotFoundError(f"No {league}_*.csv files in {raw_dir}. "
                                f"Run `python -m premodel download{flag}` first.")
    m = pd.concat([load_season(p) for p in paths], ignore_index=True)
    m = m.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    m = m[m["hg"].notna() & m["ag"].notna()].copy()
    m[["hg", "ag"]] = m[["hg", "ag"]].astype(int)
    m["result"] = np.select([m.hg > m.ag, m.hg == m.ag], [0, 1], 2)
    return m.reset_index(drop=True)


def devig(odds: np.ndarray) -> np.ndarray:
    """Proportional (multiplicative) margin removal -> fair probabilities."""
    implied = 1.0 / np.asarray(odds, dtype=float)
    return implied / implied.sum(axis=1, keepdims=True)


def coverage_report(m: pd.DataFrame) -> pd.DataFrame:
    return m.groupby("season").agg(
        matches=("home", "size"),
        pinnacle_close=("close_src", lambda s: int((s == "pinnacle").sum())),
        avg_close=("close_src", lambda s: int((s == "market_avg").sum())),
    )
