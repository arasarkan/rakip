"""Hugging Face Hub: her rakip için ayrı depo, checkpoint ve çıktı yükleme.

Token hiçbir zaman koda yazılmaz; `env.get_secret("HF_TOKEN")` ile okunur.
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

from .config import HUB_TAG


def slugify(text: str) -> str:
    text = text.translate(str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
                                         "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c"}))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "oyuncu"


def repo_name(display_name: str, tempo: str) -> str:
    return f"rakip-{slugify(display_name)}-{tempo}"


class HubStore:
    """Tek bir rakip deposu üzerinde yükleme/indirme. Ağ hataları eğitimi durdurmaz."""

    def __init__(self, repo_id: str, token: str, private: bool = False, log=print):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        self.log = log
        self.api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    @classmethod
    def for_player(cls, display_name: str, tempo: str, token: str, private: bool = False, log=print) -> "HubStore":
        from huggingface_hub import HfApi

        user = HfApi(token=token).whoami()["name"]
        return cls(f"{user}/{repo_name(display_name, tempo)}", token, private=private, log=log)

    def upload(self, local: str | Path, remote: str, message: str | None = None, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=remote, repo_id=self.repo_id,
                    repo_type="model", commit_message=message or f"rakip: {remote}",
                )
                return True
            except Exception as exc:
                self.log(f"[hub] {remote} yüklenemedi (deneme {attempt + 1}/{retries}): {exc}")
                time.sleep(10 * (attempt + 1))
        return False

    def upload_folder(self, folder: str | Path, message: str, delete_patterns: list[str] | None = None) -> bool:
        try:
            self.api.upload_folder(
                folder_path=str(folder), repo_id=self.repo_id, repo_type="model",
                commit_message=message, delete_patterns=delete_patterns,
            )
            return True
        except Exception as exc:
            self.log(f"[hub] klasör yüklenemedi: {exc}")
            return False

    def download(self, remote: str, local_dir: str | Path) -> Path | None:
        """Dosya depoda varsa indirir, yoksa None döner."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        try:
            return Path(hf_hub_download(
                self.repo_id, remote, repo_type="model", token=self.token,
                local_dir=str(local_dir), force_download=True,
            ))
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        except Exception as exc:
            self.log(f"[hub] {remote} indirilemedi: {exc}")
            return None

    def delete(self, remote: str) -> None:
        try:
            self.api.delete_file(remote, repo_id=self.repo_id, repo_type="model")
        except Exception:
            pass


def model_card(display_name: str, tempo: str, summary: dict) -> str:
    """Depo README'si. `tags` içindeki etiket, sitenin depoyu bulmasını sağlar."""
    main = summary.get("main_score")
    return f"""---
license: agpl-3.0
library_name: onnx
tags:
- {HUB_TAG}
- chess
- maia3
base_model: UofTCSSLab/Maia3-5M
---

# Rakip botu: {display_name} ({tempo})

Bu depo, [Rakip](https://github.com/arasarkan/rakip) ile üretilmiş bir satranç botudur. Bot,
{display_name} adlı oyuncunun geçmiş oyunlarından öğrenip onun gibi oynamaya
çalışır. Güçlü oynamak için değil, benzer oynamak için eğitilmiştir.

- Rakip benzerlik skoru: **{main if main is not None else '?'} / 100**
- Güven seviyesi: {summary.get('confidence_label', '?')}
- Eğitim oyunu: {summary.get('n_games', '?')} ({tempo})

Dosyalar: `model.onnx` (tarayıcıda çalışan model), `meta.json`, `scores.json`
(benzerlik skorları ve nasıl hesaplandıkları), `report.json` (rakip raporu),
`book.json` (açılış kitabı).

Temel model: [Maia3-5M](https://huggingface.co/UofTCSSLab/Maia3-5M) (CSSLab, AGPL-3.0).
"""
