"""Çalışma ortamı: Kaggle / Colab algılama, dosya yolları, gizli anahtarlar, kapanış.

Gizli anahtarlar asla koda yazılmaz. Sırasıyla Kaggle Secrets, Colab Secrets
ve (yalnızca test/CI için) ortam değişkeni denenir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def detect() -> str:
    """"kaggle", "colab" ya da "local" döner."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return "colab"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        return "local"


@dataclass
class Paths:
    env: str
    work: Path      # yazılabilir çalışma klasörü
    cache: Path     # indirme önbelleği
    out: Path       # Hub'a yüklenecek çıktılar
    inputs: Path    # kullanıcının yüklediği PGN'ler


def paths(env: str | None = None, root: str | Path | None = None) -> Paths:
    env = env or detect()
    if root is not None:
        work = Path(root)
        inputs = work / "input"
    elif env == "kaggle":
        work, inputs = Path("/kaggle/working"), Path("/kaggle/input")
    elif env == "colab":
        work, inputs = Path("/content/rakip_work"), Path("/content/input")
    else:
        work, inputs = Path.cwd() / "rakip_work", Path.cwd() / "input"
    result = Paths(env=env, work=work, cache=work / "cache", out=work / "out", inputs=inputs)
    for p in (result.work, result.cache, result.out):
        p.mkdir(parents=True, exist_ok=True)
    return result


def get_secret(name: str, env: str | None = None, required: bool = True) -> str | None:
    """Gizli anahtarı ortamın kendi kasasından okur."""
    env = env or detect()
    value = None
    if env == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret(name)
        except Exception:
            value = None
    elif env == "colab":
        try:
            from google.colab import userdata

            value = userdata.get(name)
        except Exception:
            value = None
    value = value or os.environ.get(name)
    if not value and required:
        where = {
            "kaggle": "Kaggle: Add-ons > Secrets menüsünden ekleyip not defterine erişim verin",
            "colab": "Colab: soldaki anahtar simgesinden (Secrets) ekleyip not defteri erişimini açın",
        }.get(env, "ortam değişkeni olarak tanımlayın")
        raise RuntimeError(f"'{name}' gizli anahtarı bulunamadı. {where}.")
    return value or None


def is_interactive_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "interactive"


def shutdown(env: str | None = None, log=print) -> None:
    """İş bitince oturumu kapatır ki GPU kotası boşa harcanmasın.

    - Colab: çalışma zamanı serbest bırakılır (kesin kapanır).
    - Kaggle "Save & Run All" (arka plan): not defteri bitince oturum kendiliğinden
      kapanır; ek bir şey gerekmez.
    - Kaggle etkileşimli oturum: Kaggle, oturumu koddan kapatmaya izin vermez.
      Çekirdek durdurulur ve GPU belleği bırakılır, ama kota oturum zaman aşımına
      kadar işlemeye devam edebilir. Bu yüzden rehber "Save & Run All" önerir.
    """
    env = env or detect()
    if env == "colab":
        log("[kapanış] Colab çalışma zamanı serbest bırakılıyor.")
        try:
            from google.colab import runtime

            runtime.unassign()
        except Exception as exc:
            log(f"[kapanış] Otomatik kapatılamadı ({exc}). Runtime > Disconnect and delete runtime ile kapatın.")
        return
    if env == "kaggle":
        if is_interactive_kaggle():
            log(
                "[kapanış] DİKKAT: etkileşimli Kaggle oturumu koddan kapatılamıyor.\n"
                "          Sağ üstteki güç simgesinden oturumu ŞİMDİ durdurun,\n"
                "          yoksa GPU kotanız işlemeye devam eder. Çekirdek durduruluyor."
            )
            import sys

            sys.stdout.flush()
            os._exit(0)
        log("[kapanış] Arka plan çalıştırması bitti; Kaggle oturumu kendiliğinden kapanacak.")
        return
    log("[kapanış] Yerel ortam: yapılacak bir şey yok.")
