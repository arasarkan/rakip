from rakip.budget import choose_regime, format_duration, plan_budget
from rakip.hub import repo_name, slugify


def test_regime_follows_data_amount():
    assert choose_regime(50) == "embedding"
    assert choose_regime(199) == "embedding"
    assert choose_regime(200) == "light"
    assert choose_regime(999) == "light"
    assert choose_regime(1000) == "full"


def test_small_job_is_not_reduced():
    plan = plan_budget(n_train=40_000, n_val=4_000, batch_size=256, sec_per_train_step=0.05,
                       sec_per_eval_step=0.02, max_epochs=10, fixed_overhead_seconds=3600)
    assert plan.epochs == 10 and plan.n_train == 40_000 and not plan.reduced
    assert plan.steps_per_epoch == 157
    assert 3600 < plan.estimated_seconds < 3600 + 200
    assert "Tahmini eğitim süresi" in plan.describe()


def test_over_budget_reduces_epochs_first():
    # 1 sn/adım, 10.000 adım/epoch -> epoch başı ~2.8 saat
    plan = plan_budget(n_train=2_560_000, n_val=0, batch_size=256, sec_per_train_step=1.0,
                       sec_per_eval_step=0.0, max_epochs=6, budget_seconds=10 * 3600)
    assert plan.epochs == 3 and plan.n_train == 2_560_000
    assert plan.estimated_seconds <= 10 * 3600
    assert plan.reduced and "epoch sayısı 6 -> 3" in plan.notes[0]


def test_still_over_budget_reduces_data_and_fits():
    plan = plan_budget(n_train=25_600_000, n_val=0, batch_size=256, sec_per_train_step=1.0,
                       sec_per_eval_step=0.0, max_epochs=6, budget_seconds=10 * 3600, fixed_overhead_seconds=3600)
    assert plan.epochs == 2
    assert plan.n_train < 25_600_000
    assert plan.estimated_seconds <= 10 * 3600 + 5
    assert len(plan.notes) == 2 and "pozisyona düşürüldü" in plan.notes[1]


def test_duration_formatting():
    assert format_duration(42) == "42 sn"
    assert format_duration(600) == "10 dk"
    assert format_duration(3 * 3600) == "3.0 saat"


def test_repo_names_are_url_safe():
    assert slugify("Şükrü Çağlar Ağaoğlu") == "sukru-caglar-agaoglu"
    assert repo_name("Magnus Carlsen", "blitz") == "rakip-magnus-carlsen-blitz"
    assert slugify("???") == "oyuncu"
