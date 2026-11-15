import pytest

from rakip import env


def clear(monkeypatch):
    for name in ("KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG", "COLAB_GPU"):
        monkeypatch.delenv(name, raising=False)


def test_detects_kaggle_and_colab_from_environment(monkeypatch):
    clear(monkeypatch)
    assert env.detect() == "local"
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert env.detect() == "kaggle"
    assert not env.is_interactive_kaggle()
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert env.is_interactive_kaggle()
    clear(monkeypatch)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab")
    assert env.detect() == "colab"


def test_paths_are_created_under_root(tmp_path):
    p = env.paths("kaggle", root=tmp_path)
    assert p.env == "kaggle"
    assert p.cache.is_dir() and p.out.is_dir()
    assert p.cache.parent == tmp_path


def test_secret_falls_back_to_env_var_and_missing_secret_explains(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("RAKIP_TEST_SECRET", "deger")
    assert env.get_secret("RAKIP_TEST_SECRET", env="local") == "deger"
    monkeypatch.delenv("RAKIP_TEST_SECRET")
    assert env.get_secret("RAKIP_TEST_SECRET", env="local", required=False) is None
    with pytest.raises(RuntimeError, match="Secrets"):
        env.get_secret("RAKIP_TEST_SECRET", env="kaggle")


def test_shutdown_batch_kaggle_and_local_do_not_kill_process(monkeypatch):
    clear(monkeypatch)
    messages = []
    env.shutdown("local", log=messages.append)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    env.shutdown("kaggle", log=messages.append)
    assert len(messages) == 2
