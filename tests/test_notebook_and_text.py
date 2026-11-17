"""Not defteri ve torch gerektiren dosyalar için yerelde yapılabilen kontroller."""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def load_notebook():
    return json.loads((ROOT / "notebooks" / "rakip_train.ipynb").read_text(encoding="utf-8"))


def test_notebook_is_up_to_date_with_generator():
    import build_notebook

    assert load_notebook() == build_notebook.build(), "python tools/build_notebook.py çalıştırılmalı"


def test_notebook_cells_are_valid_python_and_parameterised():
    nb = load_notebook()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 3
    for source in code:
        ast.parse(source)
    params = {}
    exec(code[0], params)
    for name in ("OYUNCU_ADI", "TEMPO", "LICHESS_KULLANICI", "CHESSCOM_KULLANICI", "TWIC_KULLAN", "PGN_YOLLARI",
                 "FIDE_ID", "HUB_DEPO_OZEL", "RAKIP_REPO", "BITINCE_KAPAT", "EK_TEMPOLAR", "EK_VERI_AGIRLIGI"):
        assert name in params
    assert "env.shutdown()" in code[2] and "finally" in code[2]


def test_no_secrets_in_repository_files():
    """Token asla koda yazılmaz: hf_ ile başlayan anahtar ya da doğrudan atama olmamalı."""
    pattern = re.compile(r"hf_[A-Za-z0-9]{20,}|HF_TOKEN\s*=\s*[\"'][^\"']+[\"']")
    files = (
        list((ROOT / "rakip").rglob("*.py"))
        + list((ROOT / "notebooks").glob("*.ipynb"))
        + list((ROOT / "tools").glob("*.py"))
    )
    for path in files:
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_torch_modules_parse_and_light_modules_import_without_torch():
    for name in ("model", "train", "export", "pipeline", "hub", "engine"):
        ast.parse((ROOT / "rakip" / f"{name}.py").read_text(encoding="utf-8"))
    # Bu modüller torch olmadan içe aktarılabilmeli (yerel testler ve hafif ortamlar için).
    import rakip.engine  # noqa: F401
    import rakip.export  # noqa: F401
    import rakip.hub  # noqa: F401
    import rakip.pipeline  # noqa: F401


def test_scores_text_summary_shows_bars_and_uncertainty():
    from rakip.similarity import Predictions, compute_scores, format_scores_text

    rng = np.random.default_rng(0)
    n = 2000
    target = rng.integers(0, 4000, n)
    bot = target.copy()
    bot[rng.random(n) > 0.58] += 1
    gen = target.copy()
    gen[rng.random(n) > 0.50] += 2

    def mk(t):
        return Predictions(top1=t.astype(np.int32), p_target=np.ones(n, np.float32), p_top1=np.ones(n, np.float32))

    scores = compute_scores(
        player="Test Oyuncu", tempo="classical", n_games=312, target=target, measured=np.ones(n, bool),
        groups=np.repeat(np.arange(40), 50), bot=mk(bot), generic=mk(gen),
        opening={"coverage": 0.91}, ceiling={"value": 0.75},
        error_bot={"available": True, "score": 70.0, "rate_similarity": 0.8, "co_occurrence": 0.6, "few_errors": False},
    )
    text = format_scores_text(scores)
    assert "RAKİP BENZERLİK SKORU:" in text and "/100" in text
    assert "├─ Açılış uyumu    █████████░   91" in text
    assert "Orta  (312 klasik oyun ile eğitildi)" in text
    assert "genel model" in text and "%100" in scores["expectation"]
    json.dumps(scores)
