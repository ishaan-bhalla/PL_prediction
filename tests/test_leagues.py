"""League generalisation (E0 / E1) must leave every E0 path, URL and output unchanged."""
import json
import shutil

import pandas as pd
import pytest

from premodel import backtest, config, data
from premodel.experiments import championship as ch

needs = {lg: pytest.mark.skipif(not any(config.RAW_DIR.glob(f"{lg}_*.csv")),
                                reason=f"no {lg} data; run `python -m premodel download`")
         for lg in ("E0", "E1")}
HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,PSH,PSD,PSA,PSCH,PSCD,PSCA\n"


def test_e0_url_and_paths_are_unchanged():
    assert config.DEFAULT_LEAGUE == "E0"
    assert config.DOWNLOAD_URL.format(code="1415", league="E0") == \
        "https://www.football-data.co.uk/mmz4281/1415/E0.csv"
    assert data.season_path("1415") == config.RAW_DIR / "E0_1415.csv"
    assert data.season_path("1415", "E1") == config.RAW_DIR / "E1_1415.csv"


def test_download_fetches_the_requested_league(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data.urllib.request, "urlretrieve",
                        lambda url, path: calls.append((url, path.name)))
    data.download(["2526"], dest=tmp_path)
    data.download(["2526"], dest=tmp_path, league="E1")
    assert calls == [("https://www.football-data.co.uk/mmz4281/2526/E0.csv", "E0_2526.csv"),
                     ("https://www.football-data.co.uk/mmz4281/2526/E1.csv", "E1_2526.csv")]


def test_load_matches_reads_only_the_requested_league(tmp_path):
    (tmp_path / "E0_2526.csv").write_text(
        HEADER + "E0,16/08/2025,Arsenal,Leeds,2,0,1.5,4.2,6.5,1.45,4.4,7.0\n")
    (tmp_path / "E1_2526.csv").write_text(
        HEADER + "E1,09/08/2025,Hull,Derby,1,1,2.5,3.3,2.9,2.4,3.3,3.0\n"
                 "E1,09/08/2025,Stoke,Leicester,0,2,3.1,3.3,2.3,3.2,3.4,2.2\n")
    e0, e1 = data.load_matches(tmp_path), data.load_matches(tmp_path, league="E1")
    assert list(e0.home) == ["Arsenal"] and e0.equals(data.load_matches(tmp_path, "E0"))
    assert list(e1.home) == ["Hull", "Stoke"] and list(e1.result) == [1, 2]
    assert list(e1.season.unique()) == ["2025/26"]
    with pytest.raises(FileNotFoundError, match="--league E2"):
        data.load_matches(tmp_path, league="E2")


@needs["E0"]
def test_e0_load_is_unaffected_by_e1_files_in_the_raw_dir(tmp_path):
    for p in config.RAW_DIR.glob("E0_*.csv"):
        shutil.copy(p, tmp_path / p.name)
    assert data.load_matches().equals(data.load_matches(tmp_path))


@needs["E1"]
def test_e1_coverage_has_full_seasons():
    t = ch.coverage_table(data.load_matches(league="E1"), "E1")
    full = t[t.expected.notna()]
    assert (full.matches == 552).all() and (t.teams == 24).all()
    assert (full.close_missing == 0).all()


@pytest.mark.slow
@needs["E0"]
def test_e0_backtest_reproduces_published_outputs_byte_for_byte(tmp_path, monkeypatch):
    """Opt-in (`pytest -m slow`, ~15 s): re-run the whole E0 backtest into a temp dir."""
    m = data.load_matches()
    if m.date.max() > ch.TEST_END:
        pytest.skip("E0 data now extends past the published test set; outputs legitimately "
                    "differ")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    backtest.run()
    published = config.ROOT / "reports" / "summary.json"
    assert (tmp_path / "reports" / "summary.json").read_bytes() == published.read_bytes()
    saved = config.ROOT / "data" / "processed" / "test_predictions.csv"
    if saved.exists():
        assert (tmp_path / "processed" / "test_predictions.csv").read_bytes() == \
            saved.read_bytes()
    assert json.loads(published.read_text())["n_test_matches"] == 3090
    pd.testing.assert_frame_equal(m, data.load_matches(league="E0"))
