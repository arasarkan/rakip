import numpy as np
import pytest

from rakip.calibrate import (
    MIN_POSITIONS,
    TEMPERATURES,
    calibrate,
    choose_temperature,
    expected_error_rates,
    softmax_temperatures,
)


def synthetic(n=1000, seed=0):
    """Her pozisyonda 5 yasal hamle: en olası 2'si iyi, kalan 3'ü ciddi hata.

    Sıcaklık arttıkça olasılık kötü hamlelere kayar, beklenen hata oranı artar.
    """
    rng = np.random.default_rng(seed)
    logits = np.column_stack([
        rng.normal(3.0, 0.3, n), rng.normal(2.0, 0.3, n),
        rng.normal(0.0, 0.3, n), rng.normal(-0.5, 0.3, n), rng.normal(-1.0, 0.3, n),
    ])
    probs_all = softmax_temperatures(logits, TEMPERATURES)          # (n, nT, 5)
    k = 4                                                           # 5. hamle kuyrukta kalır
    probs = probs_all[:, :, :k]
    tail = probs_all[:, :, k:].sum(axis=-1)
    err = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), (n, 1))
    return probs, tail, err, probs_all


def test_expected_error_rate_rises_with_temperature_and_matches_exact_value():
    probs, tail, err, probs_all = synthetic()
    rates = expected_error_rates(probs, tail, err)
    assert np.all(np.diff(rates) > 0)
    # Kuyruk (5. hamle) gerçekten hata olduğu için tahmin burada birebir doğru.
    exact = probs_all[:, :, 2:].sum(axis=-1).mean(axis=0)
    assert np.allclose(rates, exact)


def test_unevaluated_candidates_count_as_errors():
    probs = np.array([[[0.6, 0.4]]])          # 1 pozisyon, 1 sıcaklık, 2 aday
    tail = np.array([[0.0]])
    err = np.array([[0.0, np.nan]])
    assert expected_error_rates(probs, tail, err)[0] == pytest.approx(0.4)
    assert expected_error_rates(probs, tail, err, tail_err=0.0)[0] == pytest.approx(0.0)


def test_choose_temperature_hits_target_and_prefers_one_on_ties():
    temps = (0.5, 0.8, 1.0, 1.2)
    assert choose_temperature(temps, [0.02, 0.05, 0.08, 0.11], 0.05) == 0.8
    assert choose_temperature(temps, [0.02, 0.05, 0.08, 0.11], 0.06) == 0.8
    assert choose_temperature(temps, [0.02, 0.05, 0.08, 0.11], 0.065) == 1.0   # 0.8 ile 1.0 eşit uzak -> 1.0
    assert choose_temperature(temps, [0.10, 0.10, 0.10, 0.10], 0.0) == 1.0     # eşitlik -> 1.0
    assert choose_temperature(temps, [0.20, 0.30, 0.40, 0.50], 0.01) == 0.5    # hedefe ulaşılamıyorsa en yakını


def test_calibrate_lowers_temperature_for_a_careful_player():
    probs, tail, err, _ = synthetic()
    rates = expected_error_rates(probs, tail, err)
    at_one = rates[TEMPERATURES.index(1.0)]
    result = calibrate(TEMPERATURES, probs, tail, err, target=at_one * 0.4)
    assert result["calibrated"] and result["temperature"] < 1.0
    assert result["expected_error_rate"] < at_one
    assert abs(result["expected_error_rate"] - at_one * 0.4) <= min(abs(r - at_one * 0.4) for r in rates) + 1e-12
    assert len(result["table"]) == len(TEMPERATURES)
    reckless = calibrate(TEMPERATURES, probs, tail, err, target=at_one * 1.5)
    assert reckless["temperature"] > 1.0


def test_calibrate_falls_back_with_too_few_positions():
    probs, tail, err, _ = synthetic(n=MIN_POSITIONS - 1)
    result = calibrate(TEMPERATURES, probs, tail, err, target=0.05)
    assert result == {"temperature": 1.0, "calibrated": False, "positions": MIN_POSITIONS - 1,
                      "reason": f"yetersiz veri ({MIN_POSITIONS - 1} pozisyon)"}
    assert calibrate(TEMPERATURES, *synthetic()[:3], target=float("nan"))["calibrated"] is False


def test_temperature_scaling_matches_browser_formula():
    """Tarayıcı p^(1/T) ile örnekliyor; bu, logitleri T'ye bölmekle aynı dağılımdır."""
    logits = np.array([[2.0, 1.0, -1.0]])
    for t in (0.5, 1.0, 1.4):
        python = softmax_temperatures(logits, [t])[0, 0]
        p1 = softmax_temperatures(logits, [1.0])[0, 0]
        browser = p1 ** (1 / t) / (p1 ** (1 / t)).sum()
        assert np.allclose(python, browser)


def test_pipeline_calibration_with_fake_engine(monkeypatch):
    """Hattın kalibrasyon adımı: adaylar değerlendirilir, hedef oyuncunun hata oranıdır."""
    from rakip import pipeline
    from rakip.encoding import legal_indices, index_move
    from rakip.engine import choose_rows
    from test_engine_report import fake_evaluator, setup

    games, samples = setup(n_games=40, plies=60)
    rows = choose_rows(samples.measured, None)
    k = 3
    idx = np.zeros((len(rows), k), dtype=np.int32)
    legal = samples.legal_mask(rows)
    for i in range(len(rows)):
        options = np.flatnonzero(legal[i])
        idx[i, : min(k, len(options))] = options[:k]
        idx[i, len(options):] = options[0]
    logits = np.tile(np.array([2.0, 1.0, 0.0]), (len(rows), 1))
    probs = softmax_temperatures(logits, TEMPERATURES)
    tail = np.zeros((len(rows), len(TEMPERATURES)))
    cand = (idx, probs, tail)

    monkeypatch.setattr(pipeline, "stockfish_evaluator", lambda *a, **kw: fake_evaluator)
    info = pipeline.calibrate_temperatures(games, samples, rows, cand, cand, "sahte", 1000, 60, log=lambda *_: None)
    assert set(info) == {"bot", "generic"}
    bot = info["bot"]
    assert bot["calibrated"] and bot["temperature"] in TEMPERATURES and bot["positions"] == len(rows)
    assert info["generic"]["temperature"] == bot["temperature"]   # aynı adaylar -> aynı sonuç
    # Hedef, oyuncunun bu pozisyonlardaki gerçek ciddi hata oranı.
    full = pipeline.engine_pass(games, samples, rows, fake_evaluator)
    expected = float(np.mean(full.loss_player[rows] >= 10))
    assert bot["target_error_rate"] == pytest.approx(expected)
