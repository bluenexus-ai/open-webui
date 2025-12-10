import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

from open_webui.utils.mcp.client import MCPClient
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MCP", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


@dataclass
class PooledConnection:
    """Represents a pooled MCP connection"""

    client: MCPClient
    server_id: str
    url: str
    headers: Optional[Dict[str, str]]
    created_at: float
    last_used: float
    in_use: bool = False
    use_count: int = 0


@dataclass
class PoolStats:
    """Statistics for the connection pool"""

    servers: int = 0
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connections_per_server: Dict[str, int] = field(default_factory=dict)


class MCPClientPool:
    """
    Connection pool for MCP clients with TTL and max connections per server.
    Thread-safe using asyncio locks.
    """

    def __init__(
        self,
        max_connections_per_server: int = 5,
        connection_ttl_seconds: float = 300.0,
        cleanup_interval_seconds: float = 60.0,
    ):
        self._pools: Dict[str, list[PooledConnection]] = {}
        self._lock = asyncio.Lock()
        self._max_per_server = max_connections_per_server
        self._ttl = connection_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False

    async def start(self):
        """Start the background cleanup task"""
        if not self._started:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._started = True
            log.info("[MCPPool] Connection pool started")

    async def stop(self):
        """Stop the pool and cleanup all connections"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        await self.close_all()
        self._started = False
        log.info("[MCPPool] Connection pool stopped")

    async def _cleanup_loop(self):
        """Background task to cleanup expired connections"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[MCPPool] Cleanup error: {e}")

    async def _cleanup_expired(self):
        """Remove expired connections from the pool"""
        now = time.time()
        cleaned = 0

        async with self._lock:
            for server_id, connections in list(self._pools.items()):
                expired = []
                for conn in connections:
                    if not conn.in_use and (now - conn.last_used) > self._ttl:
                        expired.append(conn)

                for conn in expired:
                    try:
                        await conn.client.disconnect()
                        cleaned += 1
                    except Exception as e:
                        log.warning(f"[MCPPool] Error disconnecting expired connection: {e}")
                    connections.remove(conn)

                if not connections:
                    del self._pools[server_id]

        if cleaned > 0:
            log.info(f"[MCPPool] Cleaned up {cleaned} expired connections")

    async def acquire(
        self,
        server_id: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        max_wait_seconds: float = 30.0,
    ) -> Tuple[MCPClient, bool]:
        """
        Acquire a connection from the pool or create a new one.
        Returns (client, is_new) tuple.

        If max connections reached, waits up to max_wait_seconds for an available connection.
        """
        wait_start = time.time()

        while True:
            async with self._lock:
                # Try to get existing idle connection
                if server_id in self._pools:
                    for conn in self._pools[server_id]:
                        if not conn.in_use:
                            conn.in_use = True
                            conn.last_used = time.time()
                            conn.use_count += 1
                            log.debug(f"[MCPPool] Reusing connection for {server_id} (use_count={conn.use_count})")
                            return conn.client, False

                # Check if we can create a new connection
                current_count = len(self._pools.get(server_id, []))
                if current_count < self._max_per_server:
                    # Can create new connection - break out of loop to create it
                    break

            # Max connections reached - wait and retry
            elapsed = time.time() - wait_start
            if elapsed >= max_wait_seconds:
                log.error(f"[MCPPool] Timeout waiting for connection to {server_id} after {elapsed:.1f}s")
                raise TimeoutError(f"Timeout waiting for MCP connection to {server_id}")

            log.debug(f"[MCPPool] Max connections ({self._max_per_server}) reached for {server_id}, waiting...")
            await asyncio.sleep(0.5)  # Wait before retrying

        # Create new connection outside lock to avoid blocking other operations
        log.info(f"[MCPPool] Creating new connection for {server_id}")
        client = MCPClient()

        try:
            await client.connect(url=url, headers=headers)
        except Exception as e:
            log.error(f"[MCPPool] Failed to create connection for {server_id}: {e}")
            raise

        pooled = PooledConnection(
            client=client,
            server_id=server_id,
            url=url,
            headers=headers,
            created_at=time.time(),
            last_used=time.time(),
            in_use=True,
            use_count=1,
        )

        async with self._lock:
            if server_id not in self._pools:
                self._pools[server_id] = []
            self._pools[server_id].append(pooled)

        log.info(f"[MCPPool] Created new connection for {server_id}")
        return client, True

    async def release(self, server_id: str, client: MCPClient):
        """Release a connection back to the pool"""
        async with self._lock:
            if server_id in self._pools:
                for conn in self._pools[server_id]:
                    if conn.client is client:
                        conn.in_use = False
                        conn.last_used = time.time()
                        log.debug(f"[MCPPool] Released connection for {server_id}")
                        return

        log.warning(f"[MCPPool] Tried to release unknown connection for {server_id}")

    async def remove(self, server_id: str, client: MCPClient):
        """Remove a connection from the pool and disconnect it"""
        async with self._lock:
            if server_id in self._pools:
                for conn in self._pools[server_id]:
                    if conn.client is client:
                        self._pools[server_id].remove(conn)
                        try:
                            await conn.client.disconnect()
                        except Exception as e:
                            log.warning(f"[MCPPool] Error disconnecting removed connection: {e}")
                        log.debug(f"[MCPPool] Removed connection for {server_id}")
                        return

    async def close_all(self):
        """Close all connections in the pool"""
        async with self._lock:
            total = 0
            for server_id, connections in self._pools.items():
                for conn in connections:
                    try:
                        await conn.client.disconnect()
                        total += 1
                    except Exception as e:
                        log.warning(f"[MCPPool] Error closing connection for {server_id}: {e}")
            self._pools.clear()
            log.info(f"[MCPPool] Closed {total} connections")

    async def get_stats(self) -> PoolStats:
        """Get pool statistics (thread-safe)"""
        stats = PoolStats()

        async with self._lock:
            stats.servers = len(self._pools)

            for server_id, connections in self._pools.items():
                stats.connections_per_server[server_id] = len(connections)
                stats.total_connections += len(connections)
                for conn in connections:
                    if conn.in_use:
                        stats.active_connections += 1
                    else:
                        stats.idle_connections += 1

        return stats


# Global pool instance
_mcp_pool: Optional[MCPClientPool] = None


def get_mcp_pool() -> MCPClientPool:
    """Get or create the global MCP pool"""
    global _mcp_pool
    if _mcp_pool is None:
        _mcp_pool = MCPClientPool()
    return _mcp_pool


async def init_mcp_pool(
    max_connections_per_server: int = 5,
    connection_ttl_seconds: float = 300.0,
    cleanup_interval_seconds: float = 60.0,
) -> MCPClientPool:
    """Initialize and start the MCP pool with custom settings"""
    global _mcp_pool
    if _mcp_pool is None:
        _mcp_pool = MCPClientPool(
            max_connections_per_server=max_connections_per_server,
            connection_ttl_seconds=connection_ttl_seconds,
            cleanup_interval_seconds=cleanup_interval_seconds,
        )
    await _mcp_pool.start()
    return _mcp_pool


async def shutdown_mcp_pool():
    """Shutdown the MCP pool"""
    global _mcp_pool
    if _mcp_pool:
        await _mcp_pool.stop()
        _mcp_pool = None
