"""Hız sınırı, geri çekilme, önbellek ve kaynak ayrıştırma testleri (sahte ağ)."""

import datetime as dt
import io
import json
import zipfile

import pytest

from rakip.data import CollectConfig, collect, data_warnings
from rakip.games import PlayerMatcher
from rakip.http import FetchError, NotFound, RateLimitedFetcher
from rakip.sources import chesscom, lichess, pgnfile, twic

from conftest import make_pgn


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.headers = headers or {}

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


class FakeSession:
    """URL -> yanıt listesi. Her çağrı sıradaki yanıtı tüketir; son yanıt tekrarlanır."""

    def __init__(self, routes):
        self.routes = {k: list(v) if isinstance(v, list) else [v] for k, v in routes.items()}
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "params": params, "headers": headers})
        queue = self.routes.get(url)
        if queue is None:
            return FakeResponse(404)
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        return response


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self):
        return self.now


def make_fetcher(tmp_path, routes, **kwargs):
    t = FakeTime()
    session = FakeSession(routes)
    fetcher = RateLimitedFetcher(
        tmp_path / "cache", session=session, sleep=t.sleep, clock=t.clock, log=lambda *_: None, **kwargs
    )
    return fetcher, session, t


def test_second_fetch_comes_from_cache(tmp_path):
    fetcher, session, _ = make_fetcher(tmp_path, {"https://x.test/a": FakeResponse(body="merhaba")})
    p1 = fetcher.fetch("https://x.test/a")
    p2 = fetcher.fetch("https://x.test/a")
    assert p1 == p2 and p1.read_text() == "merhaba"
    assert len(session.calls) == 1
    assert fetcher.cache_hits == 1


def test_requests_to_same_host_are_spaced(tmp_path):
    routes = {f"https://x.test/{i}": FakeResponse(body=str(i)) for i in range(3)}
    fetcher, _, t = make_fetcher(tmp_path, routes, min_interval=2.0)
    for i in range(3):
        fetcher.fetch(f"https://x.test/{i}")
    assert t.sleeps == [2.0, 2.0]


def test_429_backs_off_at_least_a_minute_then_succeeds(tmp_path):
    routes = {"https://x.test/a": [FakeResponse(429), FakeResponse(429), FakeResponse(body="ok")]}
    fetcher, session, t = make_fetcher(tmp_path, routes, min_interval=0)
    assert fetcher.fetch("https://x.test/a").read_text() == "ok"
    assert len(session.calls) == 3
    assert t.sleeps == [60.0, 120.0]


def test_429_honours_retry_after(tmp_path):
    routes = {"https://x.test/a": [FakeResponse(429, headers={"Retry-After": "300"}), FakeResponse(body="ok")]}
    fetcher, _, t = make_fetcher(tmp_path, routes, min_interval=0)
    fetcher.fetch("https://x.test/a")
    assert t.sleeps == [300.0]


def test_server_errors_and_connection_errors_retry(tmp_path):
    routes = {"https://x.test/a": [FakeResponse(503), ConnectionError("koptu"), FakeResponse(body="ok")]}
    fetcher, session, t = make_fetcher(tmp_path, routes, min_interval=0)
    assert fetcher.fetch("https://x.test/a").read_text() == "ok"
    assert t.sleeps == [5.0, 10.0]


def test_gives_up_after_max_retries_and_leaves_no_cache(tmp_path):
    fetcher, session, _ = make_fetcher(tmp_path, {"https://x.test/a": FakeResponse(429)}, min_interval=0, max_retries=2)
    with pytest.raises(FetchError):
        fetcher.fetch("https://x.test/a")
    assert len(session.calls) == 3
    assert not fetcher.cache_path("https://x.test/a").exists()


def test_404_raises_not_found_without_retry(tmp_path):
    fetcher, session, _ = make_fetcher(tmp_path, {})
    with pytest.raises(NotFound):
        fetcher.fetch("https://x.test/yok")
    assert len(session.calls) == 1


def test_auth_header_not_in_cache_key_and_user_agent_sent(tmp_path):
    fetcher, session, _ = make_fetcher(tmp_path, {"https://x.test/a": FakeResponse(body="ok")})
    fetcher.fetch("https://x.test/a", headers={"Authorization": "Bearer gizli"})
    fetcher.fetch("https://x.test/a")
    assert len(session.calls) == 1
    assert session.calls[0]["headers"]["User-Agent"].startswith("rakip")
    assert "gizli" not in str(fetcher.cache_path("https://x.test/a"))


# ---- kaynaklar -----------------------------------------------------------

def test_lichess_fetch_requests_tempo_and_clocks(tmp_path):
    pgn = "\n\n".join(make_pgn(seed=i, date=f"2024.02.{i + 1:02d}") for i in range(4))
    url = "https://lichess.org/api/games/user/hedef"
    fetcher, session, _ = make_fetcher(tmp_path, {url: FakeResponse(body=pgn)})
    records = lichess.fetch_games("hedef", "blitz", fetcher, max_games=100, token="tok")
    assert len(records) == 4
    call = session.calls[0]
    assert call["params"]["perfType"] == "blitz"
    assert call["params"]["clocks"] == "true"
    assert call["params"]["rated"] == "true"
    assert call["headers"]["Authorization"] == "Bearer tok"
    assert all(r.source == "lichess" and r.tempo == "blitz" for r in records)


def chesscom_game(seed, time_class, tc, rules="chess", rated=True):
    return {
        "url": f"https://www.chess.com/game/live/{seed}",
        "pgn": make_pgn(seed=seed, tc=tc),
        "time_class": time_class,
        "rules": rules,
        "rated": rated,
    }


def test_chesscom_archives_are_classified_and_cached(tmp_path):
    base = "https://api.chess.com/pub/player/hedef/games"
    old, cur = f"{base}/2024/01", f"{base}/2024/03"
    routes = {
        f"{base}/archives": FakeResponse(body=json.dumps({"archives": [old, cur]})),
        old: FakeResponse(body=json.dumps({"games": [
            chesscom_game(1, "blitz", "180+2"),
            chesscom_game(2, "rapid", "600"),
            chesscom_game(3, "daily", "1/86400"),
            chesscom_game(4, "blitz", "180", rules="chess960"),
            chesscom_game(5, "rapid", "1800"),
        ]})),
        cur: FakeResponse(body=json.dumps({"games": [chesscom_game(6, "bullet", "60"), chesscom_game(7, "blitz", "300", rated=False)]})),
    }
    fetcher, session, _ = make_fetcher(tmp_path, routes, min_interval=0)
    records = chesscom.fetch_games("Hedef", fetcher, today=dt.date(2024, 3, 10))
    tempos = sorted(r.tempo for r in records)
    assert tempos == ["blitz", "bullet", "classical", "rapid"]
    assert all(r.site.startswith("https://www.chess.com/game/") for r in records)
    n_calls = len(session.calls)
    chesscom.fetch_games("Hedef", fetcher, today=dt.date(2024, 3, 10))
    assert len(session.calls) == n_calls  # ikinci tur tamamen önbellekten


def make_zip(pgn_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("twic1500.pgn", pgn_text.encode("latin-1"))
    return buffer.getvalue()


def test_twic_scans_issues_and_filters_player(tmp_path):
    def otb(seed, white, black):
        return make_pgn(seed=seed, white=white, black=black, tc=None, clocks=False, utc=False,
                        extra_headers={"Event": "Tata Steel Masters"})

    issue_1500 = "\n\n".join([otb(1, "Carlsen,M", "Giri,A"), otb(2, "So,W", "Giri,A"), otb(3, "Giri,A", "Carlsen,M")])
    issue_1499 = "\n\n".join([otb(4, "So,W", "Carlsen,M")])
    routes = {
        "https://theweekinchess.com/twic": FakeResponse(body='<a href="zips/twic1500g.zip">'),
        "https://theweekinchess.com/zips/twic1500g.zip": FakeResponse(body=make_zip(issue_1500)),
        "https://theweekinchess.com/zips/twic1499g.zip": FakeResponse(body=make_zip(issue_1499)),
    }
    fetcher, session, _ = make_fetcher(tmp_path, routes, min_interval=0)
    records = twic.fetch_games(PlayerMatcher(["Carlsen, Magnus"]), fetcher, issues=3, log=lambda *_: None)
    assert len(records) == 3
    assert sorted(r.player_color for r in records) == ["black", "black", "white"]
    assert all(r.tempo == "classical" and r.clocks == [] for r in records)


def test_uploaded_pgn_files_and_zips(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "a.pgn").write_text(make_pgn(seed=1, white="Hedef, Ali", black="X, Y", utc=False), encoding="utf-8")
    (folder / "b.zip").write_bytes(make_zip(make_pgn(seed=2, white="Z, Q", black="Hedef, Ali", utc=False)))
    records = pgnfile.load_paths([folder], PlayerMatcher(["Hedef, Ali"]))
    assert sorted(r.player_color for r in records) == ["black", "white"]


# ---- orkestrasyon ve uyarılar --------------------------------------------

def test_low_data_warning_mentions_count_and_richer_tempo():
    warnings = data_warnings("classical", {"classical": 42, "blitz": 3000, "rapid": 10, "bullet": 0})
    assert "42" in warnings[0] and "klasik" in warnings[0]
    assert any("3000" in w and "Blitz" in w for w in warnings)
    assert data_warnings("blitz", {"blitz": 5000}) == []
    assert "sınırlı veri" in data_warnings("rapid", {"rapid": 400})[0]


def test_collect_merges_sources_selects_tempo_and_warns(tmp_path):
    pgn = "\n\n".join(make_pgn(seed=i, date=f"2024.02.{i + 1:02d}") for i in range(5))
    folder = tmp_path / "input"
    folder.mkdir()
    # yüklenen dosyada aynı oyunlardan ikisi tekrar var -> tekilleştirilmeli
    (folder / "dup.pgn").write_text("\n\n".join(make_pgn(seed=i, date=f"2024.02.{i + 1:02d}") for i in range(2)), encoding="utf-8")
    routes = {
        "https://lichess.org/api/games/user/hedef": FakeResponse(body=pgn),
        "https://lichess.org/api/user/hedef": FakeResponse(body=json.dumps(
            {"perfs": {"blitz": {"games": 5}, "rapid": {"games": 1200}, "classical": {"games": 3}}})),
    }
    fetcher, _, _ = make_fetcher(tmp_path, routes, min_interval=0)
    cfg = CollectConfig(display_name="Hedef", tempo="blitz", lichess_user="hedef",
                        otb_names=["hedef"], pgn_paths=[str(folder)])
    result = collect(cfg, fetcher, log=lambda *_: None)
    assert len(result.games) == 5
    assert result.stats["all"]["dropped_duplicate"] == 2
    assert result.counts_by_tempo["blitz"] == 5 and result.counts_by_tempo["rapid"] == 1200
    assert [g.date for g in result.games] == sorted(g.date for g in result.games)
    assert any("yalnızca 5" in w for w in result.warnings)
    assert any("1200" in w for w in result.warnings)


def test_collect_config_validation():
    with pytest.raises(ValueError):
        CollectConfig(display_name="x", tempo="blitz")
    with pytest.raises(ValueError):
        CollectConfig(display_name="x", tempo="hiperbullet", lichess_user="a")
    with pytest.raises(ValueError):
        CollectConfig(display_name="x", use_twic=True)
