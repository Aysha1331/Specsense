"""
Persistent SQLite & In-Memory Hybrid Result Cache.

Keyed by (part_number, brand).
Guarantees lookups survive server restarts, preventing repeat lookups from
burning daily AI API allowances or causing rate-limit exhaustion.

Features:
- Fast in-memory cache synchronized with persistent SQLite (specsense_cache.db).
- Hits are instant (~0.001s) and require 0 external API calls.
- Cache statistics tracking (hits, misses, total items).
"""
import os
import json
import sqlite3
from typing import Optional
from models import StructuredProduct

DB_PATH = os.getenv("CACHE_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "specsense_cache.db"))

_memory_cache: dict[str, StructuredProduct] = {}
_stats = {"hits": 0, "misses": 0}


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    global _memory_cache
    try:
        with _get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS product_cache (
                    cache_key TEXT PRIMARY KEY,
                    part_number TEXT,
                    brand TEXT,
                    data_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

            cursor = conn.execute("SELECT cache_key, data_json FROM product_cache")
            for row in cursor.fetchall():
                try:
                    product_dict = json.loads(row["data_json"])
                    product_dict["extraction_engine"] = "cache"
                    _memory_cache[row["cache_key"]] = StructuredProduct(**product_dict)
                except Exception:
                    pass
        print(f"[cache] SQLite persistent store initialized: {len(_memory_cache)} products loaded from {os.path.basename(DB_PATH)}")
    except Exception as e:
        print(f"[cache] Failed to initialize SQLite cache db: {e}")


# Initialize on import
_init_db()


def _key(part_number: str, brand: str) -> str:
    pn_clean = "".join(c for c in (part_number or "").strip().lower() if c.isalnum())
    b_clean = "".join(c for c in (brand or "").strip().lower() if c.isalnum())
    return f"{b_clean}::{pn_clean}"


def get(part_number: str, brand: str) -> Optional[StructuredProduct]:
    global _stats
    k = _key(part_number, brand)
    item = _memory_cache.get(k)
    if item:
        _stats["hits"] += 1
        item_copy = item.model_copy(deep=True)
        item_copy.extraction_engine = "cache"
        return item_copy

    _stats["misses"] += 1
    return None


def set(part_number: str, brand: str, result: StructuredProduct) -> None:
    k = _key(part_number, brand)
    _memory_cache[k] = result
    
    # Save to SQLite asynchronously/safely
    try:
        data_json = result.model_dump_json()
        with _get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO product_cache (cache_key, part_number, brand, data_json) VALUES (?, ?, ?, ?)",
                (k, part_number, brand, data_json)
            )
            conn.commit()
    except Exception as e:
        print(f"[cache] failed to persist to SQLite: {e}")


def clear() -> None:
    global _memory_cache, _stats
    _memory_cache.clear()
    _stats = {"hits": 0, "misses": 0}
    try:
        with _get_db() as conn:
            conn.execute("DELETE FROM product_cache")
            conn.commit()
    except Exception as e:
        print(f"[cache] failed to clear SQLite cache: {e}")


def get_stats() -> dict:
    total = len(_memory_cache)
    hits = _stats["hits"]
    misses = _stats["misses"]
    total_requests = hits + misses
    hit_rate = round((hits / total_requests) * 100, 1) if total_requests > 0 else 0.0
    return {
        "total_cached_products": total,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": hit_rate,
        "db_path": os.path.basename(DB_PATH),
    }
