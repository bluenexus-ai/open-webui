"""
BlueNexus In-Memory Cache

Simple TTL-based cache for reducing redundant BlueNexus API calls.
"""

import time
import logging
from typing import Any, Optional
from collections import OrderedDict
from threading import Lock

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BLUENEXUS", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


class TTLCache:
    """
    Simple TTL-based in-memory cache with LRU eviction.

    Thread-safe for concurrent access.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 60):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of items to store
            default_ttl: Default time-to-live in seconds
        """
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Returns None if key doesn't exist or is expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expires_at = self._cache[key]

            if time.time() > expires_at:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        if ttl is None:
            ttl = self._default_ttl

        expires_at = time.time() + ttl

        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Add new entry
            self._cache[key] = (value, expires_at)

            # Evict oldest if over max size
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern (prefix match).

        Returns count of deleted keys.
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
            }


# Global cache instances for different data types
# Short TTL for data that changes frequently
chat_cache = TTLCache(max_size=500, default_ttl=30)  # Chat metadata
messages_cache = TTLCache(max_size=200, default_ttl=15)  # Message maps

# Longer TTL for relatively static data
tools_cache = TTLCache(max_size=100, default_ttl=300)  # Tool definitions

# Record ID cache - maps owui_id to BlueNexus record ID (longer TTL since IDs don't change)
record_id_cache = TTLCache(max_size=1000, default_ttl=600)  # 10 minutes


def make_cache_key(user_id: str, collection: str, record_id: str = "") -> str:
    """Create a consistent cache key."""
    return f"{user_id}:{collection}:{record_id}"


def make_record_id_key(user_id: str, collection: str, owui_id: str) -> str:
    """Create a cache key for BlueNexus record ID lookup."""
    return f"rid:{user_id}:{collection}:{owui_id}"


def get_cached_record_id(user_id: str, collection: str, owui_id: str) -> Optional[str]:
    """Get cached BlueNexus record ID for an owui_id."""
    key = make_record_id_key(user_id, collection, owui_id)
    return record_id_cache.get(key)


def set_cached_record_id(user_id: str, collection: str, owui_id: str, record_id: str) -> None:
    """Cache BlueNexus record ID for an owui_id."""
    key = make_record_id_key(user_id, collection, owui_id)
    record_id_cache.set(key, record_id)


def invalidate_user_cache(user_id: str) -> None:
    """Invalidate all cache entries for a user."""
    pattern = f"{user_id}:"
    rid_pattern = f"rid:{user_id}:"
    chat_cache.delete_pattern(pattern)
    messages_cache.delete_pattern(pattern)
    tools_cache.delete_pattern(pattern)
    record_id_cache.delete_pattern(rid_pattern)
    log.debug(f"[BlueNexus Cache] Invalidated cache for user {user_id}")


def invalidate_chat_cache(user_id: str, chat_id: str) -> None:
    """Invalidate cache entries for a specific chat."""
    chat_cache.delete(make_cache_key(user_id, "chats", chat_id))
    messages_cache.delete(make_cache_key(user_id, "messages", chat_id))
    log.debug(f"[BlueNexus Cache] Invalidated cache for chat {chat_id}")


def get_cache_stats() -> dict:
    """Get statistics for all caches."""
    return {
        "chat_cache": chat_cache.stats(),
        "messages_cache": messages_cache.stats(),
        "tools_cache": tools_cache.stats(),
        "record_id_cache": record_id_cache.stats(),
    }
