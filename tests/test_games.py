from collections import Counter

from rakip.config import classify_otb_event, classify_time_control
from rakip.games import GameRecord, PlayerMatcher, dedupe, parse_pgn_text

from conftest import make_pgn


def test_tempo_classification():
    assert classify_time_control("60+0") == "bullet"
    assert classify_time_control("180+0") == "blitz"
    assert classify_time_control("180+2") == "blitz"
    assert classify_time_control("300+3") == "blitz"
    assert classify_time_control("600+0") == "rapid"
    assert classify_time_control("900+10") == "rapid"
    assert classify_time_control("1800+0") == "classical"
    assert classify_time_control("5400+30") == "classical"
    assert classify_time_control("-") == "correspondence"
    assert classify_time_control("1/86400") == "correspondence"
    assert classify_time_control(None) is None
    assert classify_time_control("40/7200:3600") is None
    assert classify_otb_event("World Blitz Championship 2023") == "blitz"
    assert classify_otb_event("FIDE World Rapid") == "rapid"
    assert classify_otb_event("Tata Steel Masters") == "classical"


def test_parse_keeps_moves_clocks_color_rating_date():
    pgn = make_pgn(white="Hedef", black="Biri", date="2024.03.02", seed=1)
    stats = Counter()
    records = parse_pgn_text(pgn, PlayerMatcher(["hedef"], exact=True), "lichess", stats=stats, prefilter=False)
    assert len(records) == 1
    r = records[0]
    assert r.player_color == "white"
    assert r.tempo == "blitz"
    assert r.date == "2024-03-02T12:00:00"
    assert r.result == 1.0
    assert r.player_rating == 2100 and r.opponent_rating == 2050
    assert r.opponent_name == "Biri"
    assert len(r.moves) == 30 and len(r.clocks) == 30
    assert all(c is not None for c in r.clocks)
    assert list(r.player_ply_indices())[:3] == [0, 2, 4]
    assert stats["kept"] == 1


def test_black_perspective_result_and_ratings():
    pgn = make_pgn(white="Biri", black="hedef", result="1-0", seed=2)
    r = parse_pgn_text(pgn, PlayerMatcher(["hedef"], exact=True), "lichess", prefilter=False)[0]
    assert r.player_color == "black"
    assert r.result == 0.0
    assert r.player_rating == 2050 and r.opponent_rating == 2100
    assert list(r.player_ply_indices())[:2] == [1, 3]


def test_cleaning_drops_variants_short_and_foreign_games():
    text = "\n\n".join(
        [
            make_pgn(seed=1),
            make_pgn(seed=2, extra_headers={"Variant": "Chess960"}),
            make_pgn(seed=3, plies=6),
            make_pgn(seed=4, white="baskasi", black="digeri"),
            make_pgn(seed=5, tc="-"),
            make_pgn(seed=6, extra_headers={"FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "SetUp": "1"}),
        ]
    )
    stats = Counter()
    records = parse_pgn_text(text, PlayerMatcher(["hedef"], exact=True), "lichess", stats=stats, prefilter=False)
    assert len(records) == 1
    assert stats["dropped_variant_or_custom_start"] == 2
    assert stats["dropped_too_short"] == 1
    assert stats["dropped_player_not_found"] == 1
    assert stats["dropped_correspondence"] == 1


def test_missing_clocks_are_empty_and_otb_tempo_from_event():
    pgn = make_pgn(
        white="Carlsen,M", black="Caruana,F", tc=None, clocks=False, seed=7, utc=False,
        extra_headers={"Event": "World Rapid 2023", "WhiteFideId": "1503014"},
    )
    r = parse_pgn_text(pgn, PlayerMatcher(["Carlsen, Magnus"]), "twic")[0]
    assert r.clocks == [] and not r.has_clocks
    assert r.tempo == "rapid"
    assert r.date == "2024-01-15"


def test_otb_name_matching():
    m = PlayerMatcher(["Magnus Carlsen"])
    assert m.color({"White": "Carlsen,M", "Black": "Nakamura,Hi"}) == "white"
    assert m.color({"White": "Nakamura, Hikaru", "Black": "Carlsen, Magnus"}) == "black"
    assert m.color({"White": "Carlsen,J", "Black": "Nakamura,Hi"}) is None
    assert m.color({"White": "Carlsson,M", "Black": "Nakamura,Hi"}) is None
    # aksanlar ve büyük/küçük harf
    assert PlayerMatcher(["Sarıcı, Ali"]).color({"White": "Sarici,A", "Black": "x"}) == "white"
    assert PlayerMatcher(["Sarici, Ali"]).color({"White": "SARICI,Ali", "Black": "x"}) == "white"
    assert PlayerMatcher(["Mamedyarov, Shakhriyar"]).color({"White": "x", "Black": "MAMEDYAROV,S"}) == "black"


def test_fide_id_overrides_ambiguous_name():
    m = PlayerMatcher(["Carlsen, M"], fide_id="1503014")
    headers_right = {"White": "Carlsen,M", "Black": "x", "WhiteFideId": "1503014"}
    headers_wrong = {"White": "Carlsen,M", "Black": "x", "WhiteFideId": "9999999"}
    headers_no_id = {"White": "Carlsen,M", "Black": "x"}
    assert m.color(headers_right) == "white"
    assert m.color(headers_wrong) is None
    assert m.color(headers_no_id) == "white"


def test_prefilter_skips_unrelated_games_without_parsing():
    text = "\n\n".join([make_pgn(seed=i, white="baskasi", black="digeri") for i in range(5)] + [make_pgn(seed=9)])
    stats = Counter()
    records = parse_pgn_text(text, PlayerMatcher(["hedef"], exact=True), "pgn", stats=stats)
    assert len(records) == 1
    assert stats["skipped_prefilter"] == 5


def test_dedupe_same_game_from_two_sources():
    pgn = make_pgn(seed=11)
    a = parse_pgn_text(pgn, PlayerMatcher(["hedef"], exact=True), "lichess", prefilter=False)[0]
    b = parse_pgn_text(pgn, PlayerMatcher(["hedef"], exact=True), "pgn", prefilter=False)[0]
    assert a.game_id == b.game_id
    stats = Counter()
    assert len(dedupe([a, b], stats)) == 1
    assert stats["dropped_duplicate"] == 1


def test_jsonl_roundtrip(tmp_path):
    from rakip.games import load_jsonl, save_jsonl

    r = parse_pgn_text(make_pgn(seed=12), PlayerMatcher(["hedef"], exact=True), "lichess", prefilter=False)[0]
    save_jsonl([r], tmp_path / "g.jsonl")
    loaded = load_jsonl(tmp_path / "g.jsonl")
    assert loaded == [r]
    assert isinstance(loaded[0], GameRecord)


def test_otb_series_without_tempo_word_are_rapid_and_armageddon_is_blitz():
    assert classify_otb_event("Global Chess League 2024") == "rapid"
    assert classify_otb_event("Champions Chess Tour Finals 2024") == "rapid"
    assert classify_otb_event("Meltwater Champions Chess Tour  Airthings") == "rapid"
    assert classify_otb_event("Norway Chess Armageddon 2024") == "blitz"
    assert classify_otb_event("Superbet Rapid 2025") == "rapid"
    assert classify_otb_event("Superbet Blitz 2025") == "blitz"
    assert classify_otb_event("Norway Chess 2025") == "classical"
    assert classify_otb_event(None) == "classical"
