"""Tempo sınıflandırma düzeltmeleri, etkinlik dökümü ve ek tempo verisinin sızıntısız seçimi."""

import copy
import json
from collections import Counter

import numpy as np
import pytest

from rakip.config import classify_otb_event
from rakip.data import CollectConfig, collect, event_family, event_summary
from rakip.features import build_samples
from rakip.games import PlayerMatcher, parse_pgn_text
from rakip.split import LeakageError, Split, assert_no_leakage, select_extra_training, temporal_split

from conftest import make_pgn, make_record
from test_http_sources import FakeResponse, make_fetcher


# ---- sınıflandırma -------------------------------------------------------

@pytest.mark.parametrize(
    "event,site,expected",
    [
        ("Titled Tue 2nd Jan Early", "Chess.com INT", "blitz"),     # TWIC'in kısaltması
        ("Titled Tuesday Blitz", "Chess.com INT", "blitz"),
        ("Speed Chess Championship 2024", "Chess.com INT", "blitz"),  # eskiden yanlışlıkla rapid
        ("Champions Chess Tour Finals", "Chess.com INT", "rapid"),
        ("Global Chess League 2024", "London ENG", "rapid"),
        ("Chess.com Weekly Arena", "Chess.com INT", None),          # çevrim içi, tempo bilinmiyor
        ("Norway Chess 2025", "Stavanger NOR", "classical"),        # masa başı -> klasik
        ("Tata Steel Masters", None, "classical"),
    ],
)
def test_classification_with_site(event, site, expected):
    assert classify_otb_event(event, site) == expected


def test_unknown_online_events_are_dropped_and_counted():
    pgn = make_pgn(white="Carlsen,M", black="X,Y", tc=None, clocks=False, utc=False,
                   extra_headers={"Event": "Chess.com Weekly Arena", "Site": "Chess.com INT"})
    stats = Counter()
    assert parse_pgn_text(pgn, PlayerMatcher(["Carlsen, Magnus"]), "twic", stats=stats) == []
    assert stats["dropped_unknown_tempo"] == 1


def test_record_keeps_event_and_how_tempo_was_decided():
    otb = make_pgn(white="Carlsen,M", black="X,Y", tc=None, clocks=False, utc=False,
                   extra_headers={"Event": "Titled Tue 9th Jan Late", "Site": "Chess.com INT"})
    r = parse_pgn_text(otb, PlayerMatcher(["Carlsen, Magnus"]), "twic")[0]
    assert (r.tempo, r.tempo_source, r.event) == ("blitz", "event_name", "Titled Tue 9th Jan Late")
    online = parse_pgn_text(make_pgn(seed=3), PlayerMatcher(["hedef"], exact=True), "lichess", prefilter=False)[0]
    assert online.tempo_source == "time_control"
    hinted = parse_pgn_text(make_pgn(seed=3), PlayerMatcher(["hedef"], exact=True), "lichess", tempo_hint="blitz", prefilter=False)[0]
    assert hinted.tempo_source == "site"


def test_event_families_group_weekly_events():
    assert event_family("Titled Tue 2nd Jan Early") == event_family("Titled Tue 30th Dec Late") == "titled tue"
    assert event_family("World Rapid 2023") == "world rapid"
    records = []
    for i, name in enumerate(["Titled Tue 2nd Jan Early", "Titled Tue 9th Jan Late", "World Rapid 2023"]):
        r = make_record(i, "2024-01-01", tempo="blitz" if "Titled" in name else "rapid")
        r.event, r.tempo_source, r.source = name, "event_name", "twic"
        records.append(r)
    summary = event_summary(records)
    assert summary[0] == {"family": "titled tue", "tempo": "blitz", "how": "event_name", "source": "twic",
                          "games": 2, "example": "Titled Tue 2nd Jan Early"}
    assert summary[1]["games"] == 1 and summary[1]["tempo"] == "rapid"
    json.dumps(summary)


# ---- ek tempolar: seçim ve sızıntı ---------------------------------------

def dated(i, date, tempo="rapid"):
    return make_record(i, date, tempo=tempo)


@pytest.fixture
def rapid_split():
    games = [dated(i, f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}") for i in range(100)]
    return temporal_split(games, test_fraction=0.2, val_fraction=0.1)


def test_extra_games_only_from_before_validation_period(rapid_split):
    cutoff = min(g.date for g in rapid_split.val)
    extra = [dated(1000 + i, f"2023-0{m}-15", tempo="classical") for i, m in enumerate(range(1, 4))]
    extra.append(dated(2000, "2023-12-30", tempo="classical"))   # test dönemi: girmemeli
    extra.append(dated(2001, cutoff[:10], tempo="blitz"))        # doğrulama başlangıcı: girmemeli
    undated = dated(2002, None, tempo="blitz")
    extra.append(undated)
    chosen, info = select_extra_training(extra, rapid_split)
    assert [g.game_id for g in chosen] == [f"g0{1000 + i}" for i in range(3)]
    assert info["dropped"] == {"after_cutoff": 2, "undated": 1, "duplicate": 0}
    assert info["cutoff"] == cutoff[:10]
    assert_no_leakage(Split(train=rapid_split.train + chosen, val=rapid_split.val, test=rapid_split.test))


def test_extra_copy_of_a_test_game_is_rejected(rapid_split):
    clone = copy.deepcopy(rapid_split.test[0])
    clone.game_id, clone.tempo, clone.date = "kopya", "classical", "2023-01-02"
    # Ek veride seçim temkinlidir: hamleleri bir doğrulama/test oyunuyla aynı olan
    # oyun, tarihi uzak olsa bile alınmaz.
    chosen, info = select_extra_training([clone], rapid_split)
    assert chosen == [] and info["dropped"]["duplicate"] == 1
    # Tarihi yakın bir kopya kontrol olmadan eklenseydi sızıntı denetimi yakalardı.
    near = copy.deepcopy(clone)
    near.date = rapid_split.val[-1].date
    with pytest.raises(LeakageError):
        assert_no_leakage(Split(train=rapid_split.train + [near], val=[], test=rapid_split.test))


def test_sample_weights_follow_game_weight():
    games = [make_record(1, "2024-01-01", plies=20), make_record(2, "2024-01-02", plies=20)]
    plain = build_samples(games)
    assert plain.weight is None and (plain.weights() == 1).all()
    weighted = build_samples(games, game_weight=lambda g: 1.0 if g.game_id == "g00001" else 0.25)
    w = weighted.weights()
    assert set(np.round(w[weighted.game_index == 0], 3)) == {1.0}
    assert set(np.round(w[weighted.game_index == 1], 3)) == {0.25}
    sub = weighted.subset(np.flatnonzero(weighted.game_index == 1))
    assert (sub.weights() == np.float32(0.25)).all()


# ---- veri toplama: ek tempolar ------------------------------------------

def test_collect_downloads_neighbour_tempos_and_returns_them_separately(tmp_path):
    games = [make_pgn(seed=i, date=f"2024.03.{i + 1:02d}", tc="600+5") for i in range(4)]      # rapid
    games += [make_pgn(seed=10 + i, date=f"2024.02.{i + 1:02d}", tc="1800+0") for i in range(3)]  # klasik
    games += [make_pgn(seed=20 + i, date=f"2024.01.{i + 1:02d}", tc="60+0") for i in range(2)]    # bullet
    routes = {
        "https://lichess.org/api/games/user/hedef": FakeResponse(body="\n\n".join(games)),
        "https://lichess.org/api/user/hedef": FakeResponse(body=json.dumps({"perfs": {"bullet": {"games": 900}}})),
    }
    fetcher, session, _ = make_fetcher(tmp_path, routes, min_interval=0)
    messages = []
    cfg = CollectConfig(display_name="Hedef", tempo="rapid", lichess_user="hedef")
    assert cfg.extra_tempos == ["classical", "blitz"]
    result = collect(cfg, fetcher, log=messages.append)
    assert session.calls[0]["params"]["perfType"] == "rapid,classical,blitz"
    assert len(result.games) == 4 and {g.tempo for g in result.games} == {"rapid"}
    assert len(result.extra_games) == 3 and {g.tempo for g in result.extra_games} == {"classical"}
    assert result.counts_by_tempo["bullet"] == 900     # indirilmeyen tempo profil özetinden
    assert any("ek veri" in w for w in result.warnings)
    assert result.events and result.events[0]["how"] == "time_control"

    only = CollectConfig(display_name="Hedef", tempo="rapid", lichess_user="hedef", extra_tempos=[])
    assert only.extra_tempos == []
    with pytest.raises(ValueError):
        CollectConfig(display_name="Hedef", tempo="rapid", lichess_user="hedef", extra_tempos=["hiper"])
