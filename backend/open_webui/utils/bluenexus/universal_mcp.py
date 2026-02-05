"""
BlueNexus Universal MCP Agent Handler

This module provides the handler for the Universal MCP agent that replaces
Open WebUI's ReACT agent. All tool execution is delegated to BlueNexus.

The Universal MCP server exposes:
- use-agent: AI agent with dynamic MCP tool discovery and execution
- list-connections: List available MCP provider connections

When enabled, this completely replaces Open WebUI's local ReACT agent.
All tool discovery and execution happens on the BlueNexus side.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.utils.mcp.client import MCPClient
from open_webui.utils.bluenexus.config import (
    BLUENEXUS_API_BASE_URL,
    is_bluenexus_enabled,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MCP", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))

# Universal MCP Constants
UNIVERSAL_MCP_ENDPOINT = "/mcp"
UNIVERSAL_MCP_TOOL_NAME = "use-agent"


@dataclass
class UniversalMcpResult:
    """Result from Universal MCP agent execution."""
    success: bool
    response: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def get_universal_mcp_url() -> Optional[str]:
    """Get the Universal MCP endpoint URL."""
    if not BLUENEXUS_API_BASE_URL.value:
        return None
    base = BLUENEXUS_API_BASE_URL.value.rstrip("/")
    return f"{base}{UNIVERSAL_MCP_ENDPOINT}"


def get_user_access_token(user_id: str) -> Optional[str]:
    """Get the BlueNexus access token for a user."""
    try:
        oauth_session = OAuthSessions.get_session_by_provider_and_user_id(
            provider="bluenexus", user_id=user_id
        )
        if oauth_session and oauth_session.token:
            return oauth_session.token.get("access_token")
    except Exception as e:
        log.warning(f"Failed to get BlueNexus access token: {e}")
    return None


def is_universal_mcp_available(user_id: str) -> bool:
    """
    Check if Universal MCP is available for the user.

    Returns True if:
    - BlueNexus is enabled
    - Universal MCP URL is configured
    - User has a valid BlueNexus access token
    """
    if not is_bluenexus_enabled():
        return False
    if not get_universal_mcp_url():
        return False
    if not get_user_access_token(user_id):
        return False
    return True


async def call_universal_mcp_agent(
    user_id: str,
    prompt: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    connection: Optional[str] = None,
    event_emitter: Optional[Callable] = None,
    image_generation_enabled: bool = False,
    web_search_context: Optional[str] = None,
) -> UniversalMcpResult:
    """
    Call the BlueNexus Universal MCP agent with a prompt.

    This replaces Open WebUI's ReACT agent - all tool discovery and execution
    is handled by BlueNexus.

    Args:
        user_id: User ID for authentication
        prompt: The user's query/prompt
        conversation_history: Optional previous messages for context
        connection: Optional filter to specific MCP provider (e.g., "github", "notion")
        event_emitter: Optional callback for status updates
        image_generation_enabled: If True, instruct agent to only retrieve data (not generate images)
        web_search_context: Optional web search results to include in the prompt

    Returns:
        UniversalMcpResult with the agent's response
    """
    url = get_universal_mcp_url()
    if not url:
        return UniversalMcpResult(
            success=False,
            response="",
            error="Universal MCP URL not configured"
        )

    access_token = get_user_access_token(user_id)
    if not access_token:
        return UniversalMcpResult(
            success=False,
            response="",
            error="Not authenticated with BlueNexus. Please connect your BlueNexus account."
        )

    log.info(f"[Universal MCP] Calling agent for user {user_id}")

    # Emit connecting status
    if event_emitter:
        await event_emitter({
            "type": "status",
            "data": {
                "action": "universal_mcp_connecting",
                "description": "Connecting to BlueNexus AI Agent...",
                "done": False,
            },
        })

    mcp_client = MCPClient()

    try:
        # Connect to Universal MCP
        headers = {"Authorization": f"Bearer {access_token}"}

        log.info(f"[Universal MCP] Connecting to {url}")
        await mcp_client.connect(url=url, headers=headers)

        # Emit executing status
        if event_emitter:
            await event_emitter({
                "type": "status",
                "data": {
                    "action": "universal_mcp_executing",
                    "description": "BlueNexus AI Agent is working...",
                    "done": False,
                },
            })

        # Build the prompt with conversation context
        full_prompt = _build_prompt_with_context(prompt, conversation_history)

        # Include web search context if available
        if web_search_context:
            full_prompt = (
                f"[Web Search Results]\n"
                f"The following information was retrieved from web search. "
                f"Use this context to answer the user's question:\n\n"
                f"{web_search_context}\n\n"
                f"[User Query]\n{full_prompt}"
            )
            log.info(f"[Universal MCP] Including web search context ({len(web_search_context)} chars)")

        # If image generation is enabled, instruct agent to only retrieve data
        if image_generation_enabled:
            full_prompt = (
                "<<SYSTEM OVERRIDE - SILENT DATA RETRIEVAL>>\n"
                "IMMEDIATELY fetch and return the user's data. NO questions. NO clarifications.\n\n"
                "ABSOLUTE RULES:\n"
                "1. FETCH DATA NOW - Do not ask what metric/date/style they want\n"
                "2. Return ALL available health data (steps, calories, sleep, heart rate, etc.)\n"
                "3. Output ONLY raw data values, nothing else\n"
                "4. NO questions like 'What metric would you like?'\n"
                "5. NO explanations or descriptions\n\n"
                "CORRECT OUTPUT:\n"
                "Steps: 80\n"
                "Sedentary: 909 minutes\n"
                "Active Calories: 1,092\n"
                "Sleep: 7h 12m\n"
                "Heart Rate: 65 bpm\n\n"
                "WRONG OUTPUT (NEVER do this):\n"
                "- 'What metric would you like to visualize?'\n"
                "- 'Which date range?'\n"
                "- 'I can help with that...'\n\n"
                "FETCH NOW: " + prompt
            )
            log.info("[Universal MCP] Image generation mode: instructing agent to retrieve data only")

        # Call use-agent tool
        log.info(f"[Universal MCP] Calling use-agent with prompt: {prompt[:100]}...")

        args = {"prompt": full_prompt}
        if connection:
            args["connection"] = connection

        result = await mcp_client.call_tool(
            UNIVERSAL_MCP_TOOL_NAME,
            args,
            timeout=600.0,  # 10 minute timeout for complex tasks
        )

        # Emit complete status
        if event_emitter:
            await event_emitter({
                "type": "status",
                "data": {
                    "action": "universal_mcp_complete",
                    "description": "BlueNexus AI Agent completed",
                    "done": True,
                },
            })

        # Extract response text from MCP result
        response_text = _extract_response_text(result)

        log.info(f"[Universal MCP] Response received ({len(response_text)} chars)")

        return UniversalMcpResult(
            success=True,
            response=response_text,
            metadata={"raw_result": result} if isinstance(result, dict) else None
        )

    except Exception as e:
        log.error(f"[Universal MCP] Agent call failed: {e}")
        import traceback
        log.error(f"[Universal MCP] Traceback:\n{traceback.format_exc()}")

        if event_emitter:
            await event_emitter({
                "type": "status",
                "data": {
                    "action": "universal_mcp_error",
                    "description": f"Agent error: {str(e)[:100]}",
                    "done": True,
                },
            })

        return UniversalMcpResult(
            success=False,
            response="",
            error=str(e)
        )

    finally:
        try:
            await mcp_client.disconnect()
        except Exception:
            pass


def _build_prompt_with_context(
    prompt: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build the full prompt with conversation context."""
    if not conversation_history:
        return prompt

    context_parts = []
    # Include last few messages for context (max 6)
    recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

    for msg in recent:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")

        # Handle multimodal content (list of content blocks)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts)

        if content:
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"{role}: {content}")

    if context_parts:
        context = "\n".join(context_parts)
        return f"Conversation context:\n{context}\n\nCurrent request: {prompt}"

    return prompt


def _extract_response_text(result: Any) -> str:
    """Extract text response from MCP result."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, list):
        # MCP content blocks format
        texts = []
        for block in result:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif "text" in block:
                    texts.append(block["text"])
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)

    if isinstance(result, dict):
        # Try common keys
        if "text" in result:
            return str(result["text"])
        if "content" in result:
            return _extract_response_text(result["content"])
        if "response" in result:
            return str(result["response"])
        if "message" in result:
            return str(result["message"])
        # Fallback to string representation
        return str(result)

    return str(result)
