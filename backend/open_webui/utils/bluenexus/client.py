"""
BlueNexus User-Data API Client

This client provides methods for interacting with the BlueNexus User-Data API
to store and retrieve user data for Open WebUI.
"""

import asyncio
import json
import logging
import ssl
from datetime import datetime
from typing import Any, Optional, TypeVar
from urllib.parse import urlparse

import aiohttp


def _serialize_for_json(obj: Any) -> Any:
    """
    Recursively convert datetime objects to ISO format strings for JSON serialization.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    return obj

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.bluenexus.types import (
    BlueNexusRecord,
    BlueNexusError,
    BlueNexusAuthError,
    BlueNexusNotFoundError,
    BlueNexusValidationError,
    BlueNexusConnectionError,
    PaginatedResponse,
    QueryOptions,
    SortBy,
    SortOrder,
    ValidationError,
    VerifyResponse,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BLUENEXUS", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))

T = TypeVar("T", bound=BlueNexusRecord)


# Global session pool for connection reuse
_session_pool: dict[str, aiohttp.ClientSession] = {}


async def cleanup_session_pool() -> int:
    """
    Close all sessions in the pool and clear it.
    Returns the number of sessions closed.
    Should be called during application shutdown.
    """
    global _session_pool
    closed_count = 0
    for session_key, session in list(_session_pool.items()):
        try:
            if not session.closed:
                await session.close()
                closed_count += 1
                log.debug(f"[BlueNexus Client] Closed session {session_key}")
        except Exception as e:
            log.warning(f"[BlueNexus Client] Error closing session {session_key}: {e}")
    _session_pool.clear()
    log.info(f"[BlueNexus Client] Session pool cleanup complete, closed {closed_count} sessions")
    return closed_count


def cleanup_session_pool_sync() -> int:
    """
    Synchronous version - closes sessions without awaiting.
    Use only when async cleanup is not possible (e.g., atexit).
    """
    global _session_pool
    closed_count = 0
    for session_key, session in list(_session_pool.items()):
        try:
            if not session.closed:
                # Schedule close without waiting
                asyncio.get_event_loop().create_task(session.close())
                closed_count += 1
        except Exception:
            pass
    _session_pool.clear()
    return closed_count


class BlueNexusDataClient:
    """
    Client for BlueNexus User-Data API.

    This client handles authentication, SSL configuration, and provides
    methods for CRUD operations on user data collections.

    Uses connection pooling for improved performance.

    Usage:
        client = BlueNexusDataClient(
            base_url="https://api.bluenexus.ai",
            access_token="your-oauth-token"
        )

        # Create a record
        record = await client.create("chats", {"title": "My Chat", "messages": []})

        # Query records
        response = await client.query("chats", QueryOptions(limit=10))

        # Get a single record
        chat = await client.get("chats", "record-id")

        # Update a record
        updated = await client.update("chats", "record-id", {"title": "Updated"})

        # Delete a record
        await client.delete("chats", "record-id")
    """

    def __init__(
        self,
        base_url: str,
        access_token: str,
        timeout: int = 30,
        verify_ssl: Optional[bool] = None,
    ):
        """
        Initialize the BlueNexus Data Client.

        Args:
            base_url: BlueNexus API base URL (e.g., "https://api.bluenexus.ai")
            access_token: OAuth access token with 'user-data' scope
            timeout: Request timeout in seconds (default: 30)
            verify_ssl: Whether to verify SSL certificates. If None, auto-detects
                       based on whether the host is localhost.
        """
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Auto-detect SSL verification for localhost or Docker host
        if verify_ssl is None:
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname or ""
            self.verify_ssl = hostname not in ["localhost", "127.0.0.1", "::1", "host.docker.internal"]
        else:
            self.verify_ssl = verify_ssl

        self._data_api_path = "/api/v1/data"
        self._session_key = f"{self.base_url}:{self.verify_ssl}"

        log.info(f"[BlueNexus Client] Initialized with base_url={self.base_url}, verify_ssl={self.verify_ssl}")

    def _get_ssl_context(self) -> ssl.SSLContext | bool:
        """Get SSL context based on configuration."""
        if not self.verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            return ssl_context
        return True

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authorization."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_url(self, collection: str, record_id: Optional[str] = None) -> str:
        """Build the API URL for a collection and optional record ID."""
        # Normalize to avoid accidental double slashes if callers pass a leading slash
        # Support Enum values (Collections) by using .value when present
        collection_value = getattr(collection, "value", collection)
        collection_path = str(collection_value).strip("/")
        record_path = str(record_id).lstrip("/") if record_id is not None else None

        if record_path:
            return f"{self.base_url}{self._data_api_path}/{collection_path}/{record_path}"
        return f"{self.base_url}{self._data_api_path}/{collection_path}"

    async def _handle_response(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        """
        Handle API response and raise appropriate errors.

        Args:
            response: The aiohttp response object

        Returns:
            Parsed JSON response data

        Raises:
            BlueNexusAuthError: For 401/403 responses
            BlueNexusNotFoundError: For 404 responses
            BlueNexusValidationError: For 400 responses
            BlueNexusError: For other error responses
        """
        status = response.status

        # Success responses
        if status == 204:  # No content (delete)
            return {}

        try:
            data = await response.json()
        except Exception:
            data = {"message": await response.text()}

        if 200 <= status < 300:
            return data

        # Error responses
        message = data.get("message", data.get("error", f"HTTP {status}"))
        details = data.get("details", {})

        log.error(f"[BlueNexus Client] API error: status={status}, message={message}")

        if status == 401 or status == 403:
            raise BlueNexusAuthError(message, status_code=status, details=details)
        elif status == 404:
            raise BlueNexusNotFoundError(message, status_code=status, details=details)
        elif status == 400:
            errors = None
            if "errors" in data:
                errors = [ValidationError(**e) for e in data["errors"]]
            raise BlueNexusValidationError(message, errors=errors, status_code=status, details=details)
        else:
            raise BlueNexusError(message, status_code=status, details=details)

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create a shared session for connection pooling.

        Sessions are pooled by base_url, SSL settings, and event loop for reuse.
        Each event loop gets its own session to avoid "Future attached to different loop" errors.
        """
        global _session_pool

        # Include event loop id in session key to avoid cross-loop issues
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            loop_id = 0

        session_key = f"{self._session_key}:{loop_id}"

        # Check if we have a valid session for this loop
        if session_key in _session_pool:
            session = _session_pool[session_key]
            if not session.closed:
                return session

        # Create new session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Max connections
            limit_per_host=30,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            keepalive_timeout=30,  # Keep connections alive
        )

        session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
            trust_env=True,
        )

        _session_pool[session_key] = session
        log.debug(f"[BlueNexus Client] Created new session pool for {session_key}")

        return session

    async def _request(
        self,
        method: str,
        url: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request to the BlueNexus API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Full URL to request
            data: JSON body data (for POST/PUT)
            params: Query parameters (for GET)

        Returns:
            Parsed JSON response

        Raises:
            BlueNexusConnectionError: For network/connection errors
            BlueNexusError: For API errors
        """
        log.debug(f"[BlueNexus Client] {method} {url}")

        try:
            session = await self._get_session()

            kwargs = {
                "headers": self._get_headers(),
                "ssl": self._get_ssl_context(),
            }

            if data is not None:
                kwargs["json"] = _serialize_for_json(data)

            if params is not None:
                kwargs["params"] = params

            async with session.request(method, url, **kwargs) as response:
                log.debug(f"[BlueNexus Client] Response status: {response.status}")
                return await self._handle_response(response)

        except aiohttp.ClientError as e:
            log.error(f"[BlueNexus Client] Connection error: {e}")
            raise BlueNexusConnectionError(f"Connection error: {e}")
        except BlueNexusError:
            raise
        except Exception as e:
            log.error(f"[BlueNexus Client] Unexpected error: {e}")
            raise BlueNexusError(f"Unexpected error: {e}")

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create(
        self,
        collection: str,
        data: dict[str, Any],
        schema_uri: Optional[str] = None,
    ) -> BlueNexusRecord:
        """
        Create a new record in a collection.

        Args:
            collection: Collection name (e.g., "chats", "memories")
            data: Record data to store
            schema_uri: Optional URL to JSON schema for validation

        Returns:
            Created BlueNexusRecord with id, timestamps, and data

        Raises:
            BlueNexusValidationError: If data fails schema validation
            BlueNexusAuthError: If not authenticated
        """
        url = self._build_url(collection)
        log.info(f"[BlueNexus Client] POST {url} - Creating record in collection '{collection}'")

        payload = {**data}
        if schema_uri:
            payload["schemaUri"] = schema_uri

        response = await self._request("POST", url, data=payload)

        record = BlueNexusRecord.from_api_response(response)
        log.info(f"[BlueNexus Client] Created record id={record.id} in '{collection}'")

        return record

    async def get(
        self,
        collection: str,
        record_id: str,
    ) -> BlueNexusRecord:
        """
        Get a single record by ID.

        Args:
            collection: Collection name
            record_id: Record ID to retrieve

        Returns:
            BlueNexusRecord

        Raises:
            BlueNexusNotFoundError: If record not found
            BlueNexusAuthError: If not authenticated
        """
        url = self._build_url(collection, record_id)
        log.info(f"[BlueNexus Client] GET {url} - Getting record '{record_id}' from '{collection}'")

        response = await self._request("GET", url)

        return BlueNexusRecord.from_api_response(response)

    async def update(
        self,
        collection: str,
        record_id: str,
        data: dict[str, Any],
        schema_uri: Optional[str] = None,
    ) -> BlueNexusRecord:
        """
        Update an existing record.

        Args:
            collection: Collection name
            record_id: Record ID to update
            data: Updated record data (replaces existing data)
            schema_uri: Optional URL to JSON schema for validation

        Returns:
            Updated BlueNexusRecord

        Raises:
            BlueNexusNotFoundError: If record not found
            BlueNexusValidationError: If data fails schema validation
            BlueNexusAuthError: If not authenticated
        """
        url = self._build_url(collection, record_id)
        log.info(f"[BlueNexus Client] PUT {url} - Updating record '{record_id}' in '{collection}'")

        payload = {**data}
        if schema_uri:
            payload["schemaUri"] = schema_uri

        response = await self._request("PUT", url, data=payload)

        record = BlueNexusRecord.from_api_response(response)
        log.info(f"[BlueNexus Client] Updated record id={record.id}")

        return record

    async def delete(
        self,
        collection: str,
        record_id: str,
    ) -> None:
        """
        Delete a record.

        Args:
            collection: Collection name
            record_id: Record ID to delete

        Raises:
            BlueNexusNotFoundError: If record not found
            BlueNexusAuthError: If not authenticated
        """
        url = self._build_url(collection, record_id)
        log.info(f"[BlueNexus Client] DELETE {url} - Deleting record '{record_id}' from '{collection}'")

        await self._request("DELETE", url)

        log.info(f"[BlueNexus Client] Deleted record id={record_id}")

    async def query(
        self,
        collection: str,
        options: Optional[QueryOptions] = None,
    ) -> PaginatedResponse:
        """
        Query records in a collection with filtering and pagination.

        Args:
            collection: Collection name
            options: Query options (filter, sort, pagination)

        Returns:
            PaginatedResponse with data and pagination info

        Example:
            # Get first 10 chats sorted by creation date
            response = await client.query("chats", QueryOptions(
                limit=10,
                sort_by=SortBy.CREATED_AT,
                sort_order=SortOrder.DESC
            ))

            # Filter by status
            response = await client.query("chats", QueryOptions(
                filter={"archived": False, "pinned": True}
            ))

        Raises:
            BlueNexusAuthError: If not authenticated
        """
        url = self._build_url(collection)
        log.info(f"[BlueNexus Client] GET {url} - Querying collection '{collection}'")

        if options is None:
            options = QueryOptions()

        # Extract enum values - use .value for enums
        sort_by = options.sort_by.value if hasattr(options.sort_by, 'value') else options.sort_by
        sort_order = options.sort_order.value if hasattr(options.sort_order, 'value') else options.sort_order

        params = {
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "limit": str(options.limit),
            "page": str(options.page),
        }

        if options.filter:
            # URL-encode the filter JSON
            params["filter"] = json.dumps(options.filter)

        log.debug(f"[BlueNexus Client] Query params: {params}")
        response = await self._request("GET", url, params=params)

        paginated = PaginatedResponse(**response)
        log.info(f"[BlueNexus Client] Query returned {len(paginated.data)} records from '{collection}', total={paginated.pagination.total}")

        return paginated

    async def query_all(
        self,
        collection: str,
        filter: Optional[dict[str, Any]] = None,
        sort_by: SortBy = SortBy.CREATED_AT,
        sort_order: SortOrder = SortOrder.DESC,
        batch_size: int = 100,
    ) -> list[BlueNexusRecord]:
        """
        Query all records matching criteria (handles pagination automatically).

        Warning: Use with caution for large datasets.

        Args:
            collection: Collection name
            filter: Optional filter criteria
            sort_by: Field to sort by
            sort_order: Sort direction
            batch_size: Records per page (max 100)

        Returns:
            List of all matching BlueNexusRecord objects
        """
        log.info(f"[BlueNexus Client] Querying all records from '{collection}'")

        all_records = []
        page = 1

        while True:
            options = QueryOptions(
                filter=filter,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=batch_size,
                page=page,
            )

            response = await self.query(collection, options)
            all_records.extend(response.get_records())

            if not response.pagination.hasNext:
                break

            page += 1

        log.info(f"[BlueNexus Client] Retrieved {len(all_records)} total records from '{collection}'")
        return all_records

    async def verify(
        self,
        data: dict[str, Any],
        schema_uri: str,
    ) -> VerifyResponse:
        """
        Verify data against a schema without saving.

        Args:
            data: Data to validate
            schema_uri: URL to JSON schema

        Returns:
            VerifyResponse with valid status and any errors
        """
        log.debug(f"[BlueNexus Client] Verifying data against schema '{schema_uri}'")

        url = f"{self.base_url}{self._data_api_path}/verify"
        payload = {"schemaUri": schema_uri, **data}

        response = await self._request("POST", url, data=payload)

        return VerifyResponse(**response)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def exists(self, collection: str, record_id: str) -> bool:
        """Check if a record exists."""
        try:
            await self.get(collection, record_id)
            return True
        except BlueNexusNotFoundError:
            return False

    async def count(
        self,
        collection: str,
        filter: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Count records in a collection.

        Args:
            collection: Collection name
            filter: Optional filter criteria

        Returns:
            Total count of matching records
        """
        response = await self.query(collection, QueryOptions(filter=filter, limit=1))
        return response.pagination.total

    async def create_or_update(
        self,
        collection: str,
        record_id: str,
        data: dict[str, Any],
        schema_uri: Optional[str] = None,
    ) -> BlueNexusRecord:
        """
        Create a record if it doesn't exist, or update if it does.

        Note: This uses the provided record_id for checking existence,
        but BlueNexus generates its own IDs. For true upsert semantics,
        consider using a filter-based approach.

        Args:
            collection: Collection name
            record_id: Record ID to check
            data: Record data
            schema_uri: Optional schema URL

        Returns:
            Created or updated BlueNexusRecord
        """
        if await self.exists(collection, record_id):
            return await self.update(collection, record_id, data, schema_uri)
        else:
            return await self.create(collection, data, schema_uri)

    def with_token(self, access_token: str) -> "BlueNexusDataClient":
        """
        Create a new client instance with a different access token.

        Useful for per-request token handling.

        Args:
            access_token: New OAuth access token

        Returns:
            New BlueNexusDataClient instance
        """
        return BlueNexusDataClient(
            base_url=self.base_url,
            access_token=access_token,
            timeout=self.timeout.total,
            verify_ssl=self.verify_ssl,
        )
