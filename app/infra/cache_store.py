# app/infra/cache_store.py

"""cache_store.py
Redis 캐시 어댑터.

서비스 레이어는 이 클래스를 통해서만 캐시 계층에 접근한다. 덕분에 Redis 외의
다른 캐시 구현(Memcached 등)으로 교체하기 쉽다.
"""

from typing import Optional
from app.domain.interfaces import CacheIF
from app.cache.cache_db import get_cache_db  # RedisCacheDB 싱글턴 반환


class CacheStore(CacheIF):
    """CacheIF 구현체.

    Attributes
    ----------
    cache : RedisCacheDB
        내부에서 재사용하는 Redis 싱글턴.
    """

    def __init__(self):
        # Redis 연결은 싱글턴으로 관리한다.
        self.cache = get_cache_db() 

    # ───────────────────── CacheIF 구현 ─────────────────────
    def get_summary(self, key: str) -> Optional[str]:
        return self.cache.get_pdf(key)

    def set_summary(self, key: str, summary: str) -> None:
        self.cache.set_pdf(key, summary)

    def exists_summary(self, key: str) -> bool:
        return self.cache.exists_pdf(key)

    def set_log(self, file_id: str, url: str, query: str, lang: str, msg: str):
        """PDF 처리 단계별 로그를 Redis(HSET) 에 기록한다."""
        self.cache.set_log(file_id, url, query, lang, msg)

