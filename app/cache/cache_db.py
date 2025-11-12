"""cache_db.py
PDF 요약·실행 로그·사용자 피드백을 Redis에 임시 보관하는 캐시 모듈입니다.

설계 포인트
===========
1. 날짜별 HSET 구분 – 요약은 `pdf:summaries:YYYY-MM-DD` 형식 HSET에 저장해
   하루 단위 만료(TTL) 관리가 쉽습니다.
2. 메타데이터 키 – `pdf:metadata:<file_id>` 키에 날짜 버킷 정보를 따로 기록해
   존재 여부를 빠르게 확인합니다.
3. **TTL 관리** – 기본 보존 기간은 7일(`REDIS_TTL_DAYS`)이며, 모두 초(second)
   단위 TTL 로 설정해 계산을 단순화했습니다.

환경 변수
---------
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` : Redis 접속 정보
- `REDIS_TTL_DAYS`                     : 기본 보존 기간(일)
"""

import os
import redis
from functools import lru_cache
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

class RedisCacheDB:
    """Redis 캐시 어댑터.

    Parameters
    ----------
    host : str, optional
        Redis 호스트명.
    port : int, optional
        Redis 포트 번호.
    db : int, optional
        Redis 논리 DB.
    ttl_days : int, optional
        요약·피드백 기본 보존 기간(일).
    """
    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", "6379")),
        db: int = int(os.getenv("REDIS_DB", "0")),
        ttl_days: int = int(os.getenv("REDIS_TTL_DAYS", "7"))
    ):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl_days = ttl_days
        
    # ----- 내부 키 생성 -------------------------------------------------    
    def _get_date_key(self, date: datetime = None) -> str:
        """`pdf:summaries:YYYY-MM-DD` 형태 날짜 버킷 키 반환."""
        if date is None:
            date = datetime.now(ZoneInfo("Asia/Seoul"))
        return f"pdf:summaries:{date.strftime('%Y-%m-%d')}"
    
    def _get_metadata_key(self, file_id: str) -> str:
        """`pdf:metadata:<file_id>` 키 반환."""
        return f"pdf:metadata:{file_id}"

    # ----- 요약본 조회 -------------------------------------------------
    def get_pdf(self, fid: str) -> Optional[str]:
        """요약본을 찾으면 문자열을, 없으면 None."""
        metadata_key = self._get_metadata_key(fid)
        metadata = self.r.get(metadata_key)

        if metadata:
            self.r.expire(metadata_key, self.ttl_days * 86400)
            meta = json.loads(metadata)
            date_key = f"pdf:summaries:{meta['date']}"
            summary = self.r.hget(date_key, fid)
            if summary:
                return summary
        
        for i in range(self.ttl_days):
            date = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=i)
            date_key = self._get_date_key(date)
            summary = self.r.hget(date_key, fid)
            if summary:
                return summary
        
        return None

    def exists_pdf(self, fid: str) -> bool:
        """요약 존재 여부만 확인 (내용은 가져오지 않음)."""
        metadata_key = self._get_metadata_key(fid)
        if self.r.exists(metadata_key):
            self.r.expire(metadata_key, self.ttl_days * 86400)
            return True

        now = datetime.now(ZoneInfo("Asia/Seoul"))
        for i in range(self.ttl_days):
            date_key = self._get_date_key(now - timedelta(days=i))
            if self.r.hexists(date_key, fid):
                return True

        return False

    # ----- 요약본 저장 / 삭제 -----------------------------------------
    def set_pdf(self, fid: str, s: str):
        """요약 저장 & 메타데이터 갱신."""
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        date_key = self._get_date_key(now)
        
        self.r.hset(date_key, fid, s)
        
        metadata = {
            'date': now.strftime('%Y-%m-%d'),
            'timestamp': now.isoformat(),
            'ttl_days': self.ttl_days
        }
        self.r.setex(
            self._get_metadata_key(fid), 
            self.ttl_days * 86400,  # TTL in seconds
            json.dumps(metadata)
        )
        
        self.r.expire(date_key, (self.ttl_days + 1) * 86400)

    def delete_pdf(self, fid: str) -> bool:
        """요약·메타데이터 삭제 후 삭제 로그 남김."""
        metadata_key = self._get_metadata_key(fid)
        metadata = self.r.get(metadata_key)
        deleted = False

        if metadata:
            meta = json.loads(metadata)
            date_key = f"pdf:summaries:{meta['date']}"
            deleted = bool(self.r.hdel(date_key, fid))
            self.r.delete(metadata_key)
        else:
            for i in range(self.ttl_days):
                date = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=i)
                date_key = self._get_date_key(date)
                if self.r.hexists(date_key, fid):
                    deleted = bool(self.r.hdel(date_key, fid))
                    break

        if deleted:
            self._log_cache_deletion(fid)

        return deleted

    # ----- 로그 --------------------------------------------------------
    def set_log(self, file_id: str, url: str, query: str, lang: str, msg: str):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H:%M:%S")

        log_key = f"log:{date_str}"
        log_value = {
            "file_id": file_id,
            "url": url,
            "query": query,
            "lang": lang,
            "time": timestamp,
            "msg": msg
        }
        self.r.hset(log_key, now.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(log_value))

    def _log_cache_deletion(self, file_id: str):
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        date_str = now.strftime('%Y-%m-%d')
        date_key = f"cache:deleted:{date_str}"
        entry = f"{file_id}|{now.isoformat()}"
        self.r.rpush(date_key, entry)
        print(f"[LOG] Deleted cache entry for {file_id} → {date_key} / {entry}")

    # ----- 피드백 ------------------------------------------------------
    def add_feedback(self, file_id: str, fb_id: str, payload: dict):
        """
        Key   : feedback:<YYYY-MM-DD>
        Field : <file_id>|<fb_id>|<HH:MM:SS>
        Value : JSON 직렬화된 payload
        TTL   : summaries 정책과 동일 (ttl_days + 1 일)
        """
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        date_key = f"feedback:{now:%Y-%m-%d}"
        field    = f"{file_id}|{fb_id}|{now:%H:%M:%S}"

        self.r.hset(date_key, field, json.dumps(payload))
        self.r.expire(date_key, (self.ttl_days + 1) * 86_400)

    def get_feedbacks(self, file_id: str) -> List[dict]:
        """해당 PDF(file_id)에 달린 모든 피드백 반환."""
        results: List[dict] = []
        for i in range(self.ttl_days + 1):
            date = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=i)
            date_key = f"feedback:{date:%Y-%m-%d}"
            for field, val in self.r.hgetall(date_key).items():
                if field.startswith(f"{file_id}|"):
                    data = json.loads(val)
                    data["id"] = field.split("|")[1]
                    results.append(data)
        return results

# 싱글턴 ---------------------------------------------------------------
@lru_cache(maxsize=1)
def get_cache_db() -> "RedisCacheDB":
    return RedisCacheDB()
