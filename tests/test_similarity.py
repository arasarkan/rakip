"""Benzerlik skorunun dürüstlük kurallarını doğrulayan testler."""

import math

import numpy as np
import pytest

from rakip import similarity as sim
from rakip.similarity import Predictions, compute_scores, error_profile, main_score


def make_case(n_games=40, per_game=50, m=0.58, b=0.50, sf=0.40, seed=0, forced_share=0.1):
    """Hedef eşleşme oranlarıyla sentetik tahminler üretir."""
    rng = np.random.default_rng(seed)
    n = n_games * per_game
    target = rng.integers(0, 4000, size=n)
    groups = np.repeat(np.arange(n_games), per_game)
    measured = rng.random(n) >= forced_share

    def preds(rate, offset):
        top1 = target.copy()
        miss = rng.random(n) >= rate
        top1[miss] = target[miss] + offset
        return Predictions(top1=top1.astype(np.int32), p_target=np.full(n, 0.3, np.float32), p_top1=np.full(n, 0.4, np.float32))

    bot, generic = preds(m, 1), preds(b, 2)
    stockfish = target.copy()
    miss = rng.random(n) >= sf
    stockfish[miss] = target[miss] + 3
    return dict(target=target, groups=groups, measured=measured, bot=bot, generic=generic, stockfish_best=stockfish)


def base_kwargs(case, n_games=1500, coverage=0.9, error=None, error_generic=None):
    return dict(
        player="Test Oyuncu", tempo="blitz", n_games=n_games,
        opening={"coverage": coverage}, ceiling={"value": 0.75, "self_consistency": None},
        error_bot=error, error_generic=error_generic, n_test_games=40, **case,
    )


def good_error():
    return {"available": True, "score": 70.0, "rate_similarity": 0.8, "co_occurrence": 0.6, "few_errors": False}


# ---- formül ---------------------------------------------------------------

def test_main_score_formula_matches_documentation():
    # 0.6 * (0.58 / 0.75) + 0.2 * 0.91 + 0.2 * 0.70
    assert main_score(0.58, 0.75, 0.91, 70.0) == pytest.approx(100 * (0.6 * 0.58 / 0.75 + 0.2 * 0.91 + 0.2 * 0.70))
    assert main_score(0.58, 0.75, 0.91, 70.0) == pytest.approx(78.6, abs=0.01)


def test_move_part_is_capped_at_ceiling():
    assert main_score(0.90, 0.75, 1.0, 100.0) == pytest.approx(100.0)
    assert main_score(0.75, 0.75, 0.0, 0.0) == pytest.approx(60.0)


def test_without_error_profile_weights_renormalize():
    assert main_score(0.75, 0.75, 1.0, None) == pytest.approx(100.0)
    assert main_score(0.375, 0.75, 0.0, None) == pytest.approx(37.5)


def test_scores_round_down_never_up():
    assert sim.floor_score(71.99) == 71
    assert sim.floor_score(100.4) == 100
    assert sim.floor_score(-3) == 0
    assert sim.floor_score(float("nan")) == 0


def test_higher_ceiling_can_only_lower_the_score():
    low = main_score(0.58, 0.75, 0.9, 70.0)
    high = main_score(0.58, 0.82, 0.9, 70.0)
    assert high < low


# ---- ölçüm kuralları -----------------------------------------------------

def test_only_measured_positions_count():
    """Kitap/zorunlu hamle pozisyonlarında bot kusursuz olsa bile skor şişmez."""
    case = make_case(m=0.5, b=0.4)
    excluded = ~case["measured"]
    case["bot"].top1[excluded] = case["target"][excluded]   # dışlananlarda %100 isabet
    scores = compute_scores(**base_kwargs(case, error=good_error()))
    assert scores["references"]["bot"] == pytest.approx(0.5, abs=0.03)
    assert scores["counts"]["measured_positions"] == int(case["measured"].sum())


def test_clear_improvement_is_reported_as_significant():
    scores = compute_scores(**base_kwargs(make_case(m=0.58, b=0.50), error=good_error()))
    refs = scores["references"]
    assert refs["significant_vs_generic"] is True and refs["significant_vs_stockfish"] is True
    assert refs["bot_minus_generic_ci"][0] > 0
    assert scores["main"]["state"] == "ok"
    assert scores["main"]["score"] > scores["main"]["generic_score"]
    assert "anlamlı şekilde daha iyi" in scores["main"]["verdict"]


def test_no_improvement_over_generic_is_shown_openly():
    scores = compute_scores(**base_kwargs(make_case(m=0.50, b=0.50, seed=3), error=good_error()))
    assert scores["references"]["significant_vs_generic"] is False
    assert scores["main"]["state"] == "uncertain"
    assert "daha iyi değil" in scores["main"]["verdict"]


def test_not_better_than_stockfish_is_shown_openly():
    scores = compute_scores(**base_kwargs(make_case(m=0.55, b=0.45, sf=0.56, seed=4), error=good_error()))
    assert scores["references"]["significant_vs_stockfish"] is False
    assert scores["main"]["state"] == "uncertain"
    assert "Stockfish" in scores["main"]["verdict"]


def test_tiny_sample_cannot_claim_significance():
    """3 puanlık fark 5 oyunda anlamlı sayılmamalı."""
    case = make_case(n_games=5, per_game=30, m=0.55, b=0.52, seed=5)
    scores = compute_scores(**base_kwargs(case, n_games=120, error=good_error()))
    assert scores["references"]["significant_vs_generic"] is False
    assert scores["sub"]["confidence"]["level"] == "low"
    assert scores["main"]["state"] == "uncertain"


def test_sentence_states_raw_rate_not_normalised_score():
    scores = compute_scores(**base_kwargs(make_case(m=0.58, b=0.50), error=good_error()))
    assert "her 10 hamlenin yaklaşık 6'sında" in scores["main"]["sentence"]
    assert scores["sub"]["move"]["score"] in (56, 57, 58, 59)
    assert "yaklaşık 6'sı" in scores["sub"]["move"]["text"]
    assert scores["expectation"]


def test_missing_stockfish_and_error_profile_are_noted():
    case = make_case()
    case["stockfish_best"] = None
    scores = compute_scores(**base_kwargs(case))
    assert scores["references"]["stockfish"] is None
    assert scores["sub"]["error"]["available"] is False and scores["sub"]["error"]["score"] is None
    assert any("Hata profili hesaplanamadı" in n for n in scores["notes"])


def test_zero_measured_positions_does_not_crash_or_score():
    case = make_case()
    case["measured"][:] = False
    scores = compute_scores(**base_kwargs(case, error=good_error()))
    assert scores["main"]["state"] == "uncertain"
    assert scores["sub"]["move"]["score"] == 0


# ---- güven ---------------------------------------------------------------

@pytest.mark.parametrize(
    "n_games,n_measured,half,expected",
    [
        (50, 5000, 0.01, "low"),
        (312, 5000, 0.01, "medium"),
        (312, 400, 0.01, "low"),        # az ölçüm bir kademe düşürür
        (5000, 5000, 0.01, "high"),
        (5000, 5000, 0.05, "medium"),   # geniş güven aralığı bir kademe düşürür
    ],
)
def test_confidence_levels(n_games, n_measured, half, expected):
    assert sim.confidence_level(n_games, n_measured, half)[0] == expected


def test_low_confidence_greys_out_even_a_good_score():
    scores = compute_scores(**base_kwargs(make_case(m=0.65, b=0.50), n_games=80, error=good_error()))
    assert scores["sub"]["confidence"]["label"] == "Düşük"
    assert scores["main"]["state"] == "uncertain"
    assert "80 blitz oyun" in scores["sub"]["confidence"]["text"]


# ---- hata profili --------------------------------------------------------

def test_error_profile_identical_behaviour_scores_100():
    loss = np.array([0, 25, 0, 30, 0, 0, 12, 0] * 20, dtype=float)
    phases = np.array(["opening", "middlegame", "endgame", "middlegame"] * 40, dtype=object)
    profile = error_profile(loss, loss.copy(), phases)
    assert profile["score"] == pytest.approx(100.0)
    assert profile["co_occurrence"] == 1.0 and profile["rate_similarity"] == 1.0


def test_error_profile_perfect_engine_scores_low():
    """Hiç hata yapmayan (Stockfish gibi) bir bot bu ürün için başarısızdır."""
    loss_player = np.array([0, 25, 0, 30, 0, 0, 12, 0] * 20, dtype=float)
    phases = np.array(["middlegame"] * 160, dtype=object)
    profile = error_profile(loss_player, np.zeros(160), phases)
    assert profile["co_occurrence"] == 0.0
    assert profile["rate_similarity"] == 0.0
    assert profile["score"] == 0.0


def test_error_profile_same_rate_but_different_places_is_penalised():
    loss_player = np.array([20, 0] * 100, dtype=float)
    loss_bot = np.array([0, 20] * 100, dtype=float)
    profile = error_profile(loss_player, loss_bot, np.array(["middlegame"] * 200, dtype=object))
    assert profile["rate_similarity"] == 1.0
    assert profile["co_occurrence"] == 0.0
    assert profile["score"] == pytest.approx(50.0)


def test_error_profile_ignores_unevaluated_positions_and_flags_few_errors():
    loss_player = np.array([20.0, np.nan, 0.0, 0.0])
    loss_bot = np.array([20.0, 50.0, np.nan, 0.0])
    profile = error_profile(loss_player, loss_bot, np.array(["middlegame"] * 4, dtype=object))
    assert profile["n"] == 2 and profile["few_errors"] is True
    assert error_profile(np.array([np.nan]), np.array([np.nan]), np.array(["opening"], dtype=object))["available"] is False


def test_win_percent_is_symmetric_and_bounded():
    assert sim.win_percent(0) == pytest.approx(50.0)
    assert sim.win_percent(100) + sim.win_percent(-100) == pytest.approx(100.0)
    assert 99 < sim.win_percent(100000) <= 100
    assert sim.win_percent(300) - sim.win_percent(0) > 20


# ---- güven aralığı, tavan, oyun aşaması ----------------------------------

def test_cluster_bootstrap_is_wider_than_naive_when_games_differ():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 50)
    game_level = rng.random(20)                       # oyundan oyuna büyük fark
    values = (rng.random(1000) < game_level[groups]).astype(float)
    lo, hi = sim.cluster_bootstrap(values, groups)
    naive = 1.96 * values.std() / math.sqrt(values.size)
    assert lo < values.mean() < hi
    assert (hi - lo) / 2 > naive * 1.5


def test_ceiling_defaults_to_assumption_and_only_rises(repertoire_like):
    few = sim.estimate_ceiling(repertoire_like[:5])
    assert few["value"] == 0.75 and few["self_consistency"] is None
    consistent = sim.estimate_ceiling(repertoire_like, min_total=10)
    assert consistent["self_consistency"] is not None
    assert consistent["value"] >= 0.75


def test_phase_detection():
    import chess
    from rakip.encoding import board_codes

    start = board_codes(chess.Board())
    rook_ending = board_codes(chess.Board("8/5pk1/6p1/8/8/6P1/5PK1/3R3r w - - 0 40"))
    codes = np.stack([start, start, rook_ending])
    phases = sim.phases_from_codes(codes, np.array([4, 40, 80]))
    assert list(phases) == ["opening", "middlegame", "endgame"]


@pytest.fixture
def repertoire_like():
    from test_encoding_book import game_from_san

    return [game_from_san(i, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], filler_seed=i % 3) for i in range(40)]
