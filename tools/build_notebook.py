"""notebooks/rakip_train.ipynb dosyasını üretir.

Not defteri elle değil bu betikle düzenlenir:  python tools/build_notebook.py
"""

import json
from pathlib import Path

MD_INTRO = """# Rakip: bir rakibin botunu eğit

Bu not defteri seçtiğin oyuncunun oyunlarını indirir, onun gibi oynayan bir bot eğitir,
botun ona ne kadar benzediğini ölçer ve sonucu Hugging Face Hub'daki kendi depona yükler.
Kaggle'da ve Google Colab'da aynı şekilde çalışır.

**Çalıştırmadan önce**

1. Sağ panelde **Accelerator: GPU T4** ve **Internet: On** seçili olsun.
2. Hugging Face *write* token'ını gizli anahtar olarak ekle: adı tam olarak `HF_TOKEN` olmalı.
   - Kaggle: *Add-ons > Secrets*, ekledikten sonra bu not defterine erişimi aç.
   - Colab: soldaki anahtar simgesi, *Notebook access* açık olsun.
3. Aşağıdaki **Parametreler** hücresini doldur.
4. Kaggle'da **Save Version > Save & Run All (Commit)** ile çalıştır. Böylece iş bitince
   oturum kendiliğinden kapanır ve GPU kotan boşa harcanmaz. Etkileşimli oturumda
   çalıştırırsan iş bitince oturumu sağ üstten kendin durdurman gerekir.

Token'ı asla bir hücreye yazma.
"""

PARAMS = """# ============================ PARAMETRELER ============================
OYUNCU_ADI = "Örnek Oyuncu"      # raporda ve sitede görünecek ad
TEMPO = "blitz"                   # "classical" | "rapid" | "blitz" | "bullet"

# Kaynaklar: en az birini doldur.
LICHESS_KULLANICI = ""            # ör. "DrNykterstein"
CHESSCOM_KULLANICI = ""           # ör. "hikaru"

# Masa başı (OTB) oyunlar: TWIC arşivi ve/veya kendi yüklediğin PGN dosyaları.
TWIC_KULLAN = False
TWIC_SAYI_ADEDI = 150             # en yeni kaç haftalık sayı taransın (150 ~ 3 yıl)
PGN_YOLLARI = []                  # ör. ["/kaggle/input/oyunlarim"] (.pgn ya da .zip)
OTB_ISIMLER = []                  # ör. ["Carlsen, Magnus"]  ("Soyad, Ad" biçimi)
FIDE_ID = ""                      # ör. "1503014" (isim karışıklığını önler, önerilir)

# Ek eğitim verisi: diğer tempolardaki oyunlar da (düşük ağırlıkla) eğitime katılır.
# Benzerlik yine yalnızca TEMPO'daki oyunlarla ölçülür.
EK_TEMPOLAR = None                # None: komşu tempolar (rapid -> klasik + blitz), []: kapalı
EK_VERI_AGIRLIGI = 0.5            # ek verinin toplam etkisi, hedef tempo verisinin en çok bu oranı

# Depo ve kod
HUB_DEPO_OZEL = False             # True: özel depo (site tarayıcıda token ister)
RAKIP_REPO = "https://github.com/arasarkan/rakip"   # kendi GitHub deponun adresi
RAKIP_REF = "main"

# İleri ayarlar (genellikle dokunmaya gerek yok)
EN_FAZLA_OYUN = 20000
EGITIM = {}                       # ör. {"regime": "light", "max_epochs": 6}
BITINCE_KAPAT = True              # iş bitince oturumu kapat (GPU kotası için)
"""

INSTALL = """import subprocess, sys, time

SESSION_START = time.time()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", f"rakip[train] @ git+{RAKIP_REPO}@{RAKIP_REF}"],
    check=True,
)
import rakip
print("rakip", rakip.__version__, "kuruldu")
"""

RUN = """import traceback

from rakip import env
from rakip.data import CollectConfig
from rakip.pipeline import RunConfig, run

sonuc = None
try:
    cfg = RunConfig(
        collect=CollectConfig(
            display_name=OYUNCU_ADI,
            tempo=TEMPO,
            lichess_user=LICHESS_KULLANICI or None,
            chesscom_user=CHESSCOM_KULLANICI or None,
            otb_names=list(OTB_ISIMLER),
            fide_id=FIDE_ID or None,
            use_twic=TWIC_KULLAN,
            twic_issues=TWIC_SAYI_ADEDI,
            pgn_paths=list(PGN_YOLLARI),
            max_games=EN_FAZLA_OYUN,
            extra_tempos=None if EK_TEMPOLAR is None else list(EK_TEMPOLAR),
        ),
        extra_weight=EK_VERI_AGIRLIGI,
        hub_private=HUB_DEPO_OZEL,
        train=dict(EGITIM),
    )
    sonuc = run(cfg, session_start=SESSION_START)
except Exception:
    traceback.print_exc()
    print()
    print("HATA: iş tamamlanamadı. Yukarıdaki mesajı okuyup parametreleri düzelt.")
    print("Eğitim başladıysa checkpoint Hub'da duruyor; aynı parametrelerle yeniden çalıştırınca kaldığı yerden devam eder.")
finally:
    if BITINCE_KAPAT:
        env.shutdown()
"""

MD_AFTER = """## Sonra ne olacak?

Çıktının sonunda `siteye eklemek için depo adı: kullanıcı/rakip-...` satırını göreceksin.
Siteyi aç, **Rakip ekle** alanına bu depo adını yaz. Rapor, benzerlik skoru ve bot oradan yüklenir.

Seans yarıda kesilirse aynı parametrelerle yeniden çalıştır: eğitim Hub'daki son
checkpoint'ten devam eder.
"""


def cell(kind, source):
    lines = source.split("\n")
    body = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    base = {"cell_type": kind, "metadata": {}, "source": body}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def build():
    return {
        "cells": [
            cell("markdown", MD_INTRO),
            cell("code", PARAMS),
            cell("code", INSTALL),
            cell("code", RUN),
            cell("markdown", MD_AFTER),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "rakip_train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("notebook:", out)
