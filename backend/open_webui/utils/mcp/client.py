import asyncio
import logging
import ssl
from typing import Optional
from contextlib import AsyncExitStack
from urllib.parse import urlparse

import anyio
import httpx

from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MCP", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


def create_mcp_http_client_factory(url: str):
    """
    Create a factory function that returns an httpx client with appropriate SSL settings.
    Disables SSL verification for localhost URLs.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    is_localhost = hostname in ["localhost", "127.0.0.1", "::1"]

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        kwargs = {
            "follow_redirects": True,
        }

        if timeout is None:
            kwargs["timeout"] = httpx.Timeout(30.0)
        else:
            kwargs["timeout"] = timeout

        if headers is not None:
            kwargs["headers"] = headers

        if auth is not None:
            kwargs["auth"] = auth

        # Disable SSL verification for localhost
        if is_localhost:
            log.info(f"[MCP Client] Disabling SSL verification for localhost URL: {url}")
            kwargs["verify"] = False

        return httpx.AsyncClient(**kwargs)

    return factory


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = None

    async def connect(self, url: str, headers: Optional[dict] = None):
        log.info(f"[MCP Client] Connecting to {url}")
        log.debug(f"[MCP Client] Headers provided: {list(headers.keys()) if headers else 'None'}")

        async with AsyncExitStack() as exit_stack:
            try:
                log.debug(f"[MCP Client] Creating streamable HTTP client...")
                # Use custom httpx client factory to handle SSL for localhost
                httpx_client_factory = create_mcp_http_client_factory(url)
                self._streams_context = streamablehttp_client(
                    url,
                    headers=headers,
                    httpx_client_factory=httpx_client_factory,
                )

                log.debug(f"[MCP Client] Entering stream context...")
                transport = await exit_stack.enter_async_context(self._streams_context)
                read_stream, write_stream, _ = transport
                log.debug(f"[MCP Client] Transport established")

                self._session_context = ClientSession(
                    read_stream, write_stream
                )  # pylint: disable=W0201

                log.debug(f"[MCP Client] Creating client session...")
                self.session = await exit_stack.enter_async_context(
                    self._session_context
                )

                log.debug(f"[MCP Client] Initializing session (timeout: 10s)...")
                with anyio.fail_after(10):
                    await self.session.initialize()

                self.exit_stack = exit_stack.pop_all()
                log.info(f"[MCP Client] Successfully connected to {url}")
            except Exception as e:
                log.error(f"[MCP Client] Connection FAILED to {url}: {type(e).__name__}: {e}")
                # Log nested exceptions for TaskGroup errors
                if hasattr(e, 'exceptions'):
                    for i, sub_exc in enumerate(e.exceptions):
                        log.error(f"[MCP Client] Sub-exception {i}: {type(sub_exc).__name__}: {sub_exc}")
                        if hasattr(sub_exc, '__cause__') and sub_exc.__cause__:
                            log.error(f"[MCP Client] Sub-exception {i} cause: {type(sub_exc.__cause__).__name__}: {sub_exc.__cause__}")
                await asyncio.shield(self.disconnect())
                raise e

    async def list_tool_specs(self, timeout: float = 60.0) -> Optional[dict]:
        """
        List available tools from the MCP server.

        Uses asyncio.shield() to protect against external cancellation during
        the HTTP request, ensuring the operation completes even if the parent
        task is cancelled.

        Args:
            timeout: Maximum time to wait for the operation (default: 60s)
        """
        if not self.session:
            raise RuntimeError("MCP client is not connected.")

        log.debug("[MCP Client] Listing tools...")
        try:
            # Shield the operation from external cancellation
            # This ensures the MCP request completes even if parent task is cancelled
            async def _list_tools():
                return await self.session.list_tools()

            result = await asyncio.wait_for(
                asyncio.shield(_list_tools()),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            log.error(f"[MCP Client] list_tools() timed out after {timeout}s")
            raise
        except asyncio.CancelledError:
            log.warning("[MCP Client] list_tools() was cancelled but operation may have completed")
            raise

        tools = result.tools
        log.info(f"[MCP Client] Retrieved {len(tools)} tools from server")

        tool_specs = []
        for tool in tools:
            name = tool.name
            description = tool.description

            inputSchema = tool.inputSchema

            # TODO: handle outputSchema if needed
            outputSchema = getattr(tool, "outputSchema", None)

            tool_specs.append(
                {"name": name, "description": description, "parameters": inputSchema}
            )
            log.debug(f"[MCP Client] Tool: {name} - {description[:50] if description else 'No description'}...")

        return tool_specs

    async def call_tool(
        self, function_name: str, function_args: dict, timeout: float = 120.0
    ) -> Optional[dict]:
        """
        Call a tool on the MCP server.

        Uses asyncio.shield() to protect against external cancellation during
        the HTTP request, ensuring the tool call completes even if the parent
        task is cancelled.

        Args:
            function_name: Name of the tool to call
            function_args: Arguments to pass to the tool
            timeout: Maximum time to wait for the operation (default: 120s)
        """
        if not self.session:
            raise RuntimeError("MCP client is not connected.")

        log.info(f"[MCP Client] Calling tool '{function_name}' with args: {list(function_args.keys())}")
        log.debug(f"[MCP Client] Full args: {function_args}")

        try:
            # Shield the operation from external cancellation
            # This ensures the MCP tool call completes even if parent task is cancelled
            async def _call_tool():
                return await self.session.call_tool(function_name, function_args)

            result = await asyncio.wait_for(
                asyncio.shield(_call_tool()),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            log.error(f"[MCP Client] Tool '{function_name}' timed out after {timeout}s")
            raise
        except asyncio.CancelledError:
            log.warning(f"[MCP Client] Tool '{function_name}' was cancelled but operation may have completed")
            raise

        if not result:
            log.error(f"[MCP Client] Tool '{function_name}' returned no result")
            raise Exception("No result returned from MCP tool call.")

        result_dict = result.model_dump(mode="json")
        result_content = result_dict.get("content", {})

        if result.isError:
            log.error(f"[MCP Client] Tool '{function_name}' returned ERROR: {result_content}")
            raise Exception(result_content)
        else:
            content_type = type(result_content).__name__
            content_length = len(result_content) if isinstance(result_content, (list, dict, str)) else "N/A"
            log.info(f"[MCP Client] Tool '{function_name}' SUCCESS - result_type={content_type}, length={content_length}")
            log.debug(f"[MCP Client] Tool '{function_name}' result: {str(result_content)[:500]}")
            return result_content

    async def list_resources(self, cursor: Optional[str] = None) -> Optional[dict]:
        if not self.session:
            raise RuntimeError("MCP client is not connected.")

        result = await self.session.list_resources(cursor=cursor)
        if not result:
            raise Exception("No result returned from MCP list_resources call.")

        result_dict = result.model_dump()
        resources = result_dict.get("resources", [])

        return resources

    async def read_resource(self, uri: str) -> Optional[dict]:
        if not self.session:
            raise RuntimeError("MCP client is not connected.")

        result = await self.session.read_resource(uri)
        if not result:
            raise Exception("No result returned from MCP read_resource call.")
        result_dict = result.model_dump()

        return result_dict

    async def disconnect(self):
        # Clean up and close the session
        await self.exit_stack.aclose()

    async def __aenter__(self):
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.exit_stack.__aexit__(exc_type, exc_value, traceback)
        await self.disconnect()
