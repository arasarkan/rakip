"""Test setinin eğitime sızmadığını doğrulayan testler."""

import copy

import pytest

from rakip.split import (
    LeakageError,
    Split,
    assert_ids_clean,
    assert_no_leakage,
    manifest,
    temporal_split,
)

from conftest import make_record


def test_test_set_is_the_newest_games(dated_records):
    split = temporal_split(reversed(dated_records), test_fraction=0.2, val_fraction=0.1)
    assert len(split.test) == 20
    assert len(split.val) == 8
    assert len(split.train) == 72
    newest_fit = max(g.date for g in split.fit)
    oldest_test = min(g.date for g in split.test)
    assert newest_fit < oldest_test
    # doğrulama da eğitimden yenidir (erken durdurma geleceğe bakar)
    assert max(g.date for g in split.train) < min(g.date for g in split.val)


def test_splits_are_disjoint_and_cover_everything(dated_records):
    split = temporal_split(dated_records)
    ids = [g.game_id for g in split.train + split.val + split.test]
    assert len(ids) == len(set(ids)) == len(dated_records)


@pytest.mark.parametrize("fraction", [0.15, 0.18, 0.20])
def test_fraction_stays_in_allowed_band(dated_records, fraction):
    split = temporal_split(dated_records, test_fraction=fraction)
    assert 0.15 <= len(split.test) / len(dated_records) <= 0.20


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.25, 0.5])
def test_fraction_outside_band_is_rejected(dated_records, fraction):
    with pytest.raises(ValueError):
        temporal_split(dated_records, test_fraction=fraction)


def test_undated_games_never_enter_test(dated_records):
    undated = [make_record(500 + i, None) for i in range(30)]
    split = temporal_split(dated_records + undated)
    assert all(g.date for g in split.test)
    assert {g.game_id for g in undated} <= {g.game_id for g in split.fit}


def test_all_undated_cannot_be_split():
    with pytest.raises(ValueError):
        temporal_split([make_record(i, None) for i in range(20)])


def test_leak_same_id_is_detected(dated_records):
    split = temporal_split(dated_records)
    split.train.append(split.test[0])
    with pytest.raises(LeakageError):
        assert_no_leakage(split)


def test_leak_same_moves_different_id_is_detected(dated_records):
    """Aynı oyunun başka kaynaktan gelen kopyası (farklı kimlik) da sızıntıdır."""
    split = temporal_split(dated_records)
    clone = copy.deepcopy(split.test[3])
    clone.game_id = "baska-kimlik"
    clone.date = split.train[0].date
    split.train.append(clone)
    with pytest.raises(LeakageError):
        assert_no_leakage(split)


def test_leak_future_game_in_train_is_detected(dated_records):
    split = temporal_split(dated_records)
    future = make_record(999, "2030-01-01")
    split.train.append(future)
    with pytest.raises(LeakageError):
        assert_no_leakage(split)


def test_random_split_would_fail_the_temporal_check(dated_records):
    """Rastgele bölme (yaygın hata) kontrolden geçemez."""
    import random

    shuffled = dated_records[:]
    random.Random(0).shuffle(shuffled)
    bad = Split(train=shuffled[:70], val=shuffled[70:80], test=shuffled[80:])
    with pytest.raises(LeakageError):
        assert_no_leakage(bad)


def test_assert_ids_clean_guards_downstream_stages(dated_records):
    split = temporal_split(dated_records)
    assert_ids_clean([g.game_id for g in split.fit], split.test, "eğitim")
    with pytest.raises(LeakageError):
        assert_ids_clean([split.fit[0].game_id, split.test[-1].game_id], split.test, "açılış kitabı")


def test_manifest_records_test_ids(dated_records):
    split = temporal_split(dated_records)
    m = manifest(split)
    assert m["counts"] == {"train": 72, "val": 8, "test": 20}
    assert set(m["test_game_ids"]) == {g.game_id for g in split.test}
    assert m["date_span"]["train"][1] < m["date_span"]["test"][0]
    assert m["test_fraction"] == 0.2
