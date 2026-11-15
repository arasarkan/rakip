"""Hız sınırına uyan, diske önbellekleyen HTTP indirici.

Kurallar:
- Aynı sunucuya art arda istekler arasında en az `min_interval` saniye beklenir
  ve istekler hep sırayla yapılır (Lichess aynı anda tek istek ister).
- 429 yanıtında `Retry-After` kadar, yoksa en az 60 saniye beklenir ve her
  denemede süre katlanır.
- 5xx ve bağlantı hatalarında üstel geri çekilme uygulanır.
- İndirilen her yanıt diske yazılır; aynı istek tekrar ağa gitmez.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

from .config import USER_AGENT


class FetchError(RuntimeError):
    pass


class NotFound(FetchError):
    pass


class RateLimitedFetcher:
    def __init__(
        self,
        cache_dir: str | Path,
        min_interval: float = 1.0,
        max_retries: int = 6,
        timeout: float = 120.0,
        user_agent: str = USER_AGENT,
        session=None,
        sleep=time.sleep,
        clock=time.monotonic,
        log=print,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.user_agent = user_agent
        self._sleep = sleep
        self._clock = clock
        self._log = log
        self._last_request: dict[str, float] = {}
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.network_requests = 0
        self.cache_hits = 0

    # -- önbellek ---------------------------------------------------------

    def cache_path(self, url: str, params: dict | None = None, suffix: str = "") -> Path:
        # Kimlik başlıkları anahtara girmez; aynı içerik aynı dosyada durur.
        key = url + ("?" + urlencode(sorted(params.items())) if params else "")
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        host = urlparse(url).netloc.replace(":", "_")
        return self.cache_dir / host / f"{digest}{suffix}"

    def _fresh(self, path: Path, ttl: float | None) -> bool:
        if not path.exists():
            return False
        if ttl is None:
            return True
        return (time.time() - path.stat().st_mtime) < ttl

    # -- ağ ---------------------------------------------------------------

    def _wait_turn(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            remaining = self.min_interval - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)

    def fetch(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        ttl: float | None = None,
        suffix: str = "",
    ) -> Path:
        """URL'yi indirir ve önbellekteki dosyanın yolunu döner.

        `ttl=None` kalıcı önbellek demektir (değişmeyen arşivler için);
        saniye verilirse o süreden eski kopya yeniden indirilir.
        """
        path = self.cache_path(url, params, suffix)
        if self._fresh(path, ttl):
            self.cache_hits += 1
            return path

        host = urlparse(url).netloc
        all_headers = {"User-Agent": self.user_agent, **(headers or {})}
        last_error = "bilinmeyen hata"
        for attempt in range(self.max_retries + 1):
            self._wait_turn(host)
            response = None
            try:
                self.network_requests += 1
                response = self.session.get(
                    url, params=params, headers=all_headers, timeout=self.timeout, stream=True
                )
                self._last_request[host] = self._clock()
                status = response.status_code
                if status == 200:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(path.name + ".part")
                    with tmp.open("wb") as fh:
                        for chunk in response.iter_content(chunk_size=1 << 16):
                            if chunk:
                                fh.write(chunk)
                    os.replace(tmp, path)
                    return path
                if status == 404:
                    raise NotFound(f"Bulunamadı (404): {url}")
                if status == 429:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    wait = max(retry_after, 60.0) * (2 ** attempt)
                    last_error = "429 hız sınırı"
                    self._log(f"[http] 429 alındı, {wait:.0f} sn bekleniyor: {host}")
                elif 500 <= status < 600:
                    wait = min(300.0, 5.0 * (2 ** attempt))
                    last_error = f"sunucu hatası {status}"
                    self._log(f"[http] {status} alındı, {wait:.0f} sn sonra yeniden: {host}")
                else:
                    raise FetchError(f"Beklenmeyen yanıt {status}: {url}")
            except FetchError:
                raise
            except Exception as exc:  # bağlantı kopması, zaman aşımı vb.
                self._last_request[host] = self._clock()
                wait = min(300.0, 5.0 * (2 ** attempt))
                last_error = f"{type(exc).__name__}: {exc}"
                self._log(f"[http] bağlantı hatası, {wait:.0f} sn sonra yeniden: {last_error}")
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
            if attempt < self.max_retries:
                self._sleep(wait)
        raise FetchError(f"{self.max_retries + 1} denemeden sonra indirilemedi ({last_error}): {url}")


def _parse_retry_after(value: str | None) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
