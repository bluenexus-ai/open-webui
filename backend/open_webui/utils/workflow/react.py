"""
ReAct Agent - Reasoning and Acting in an iterative loop.

ReAct (Reasoning + Acting) is an agent paradigm that interleaves:
1. Thought - Reasoning about what to do next
2. Action - Executing a tool
3. Observation - Processing the result
4. Loop - Repeat until task is complete

Unlike "plan all then execute", ReAct decides one action at a time based on
observations from previous actions, allowing for dynamic adaptation.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MAIN", logging.INFO))


class ReActStepType(Enum):
    """Type of step in the ReAct loop."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    VERIFICATION = "verification"  # LLM verifies answer against actual tool results
    FINAL_ANSWER = "final_answer"


class ReActStatus(Enum):
    """Status of the ReAct execution."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class ReActConfig:
    """Configuration for ReAct agent behavior."""

    # Maximum number of iterations (thought-action-observation cycles)
    max_iterations: int = 20

    # Maximum consecutive failures before giving up
    max_consecutive_failures: int = 3

    # Whether to continue on tool failures
    continue_on_failure: bool = True

    # Timeout for each tool execution in seconds
    tool_timeout: float = 60.0

    # Whether to emit detailed step events
    emit_step_events: bool = True

    # Tools that should not be used (blocklist)
    blocked_tools: Set[str] = field(default_factory=set)

    # If set, only these tools can be used (allowlist)
    allowed_tools: Optional[Set[str]] = None

    # Maximum recent messages to include from conversation history
    max_history_messages: int = 6


# Default configuration
DEFAULT_REACT_CONFIG = ReActConfig()


@dataclass
class ReActStep:
    """Represents a single step in the ReAct execution."""

    step_number: int
    step_type: ReActStepType
    content: str
    tool_name: Optional[str] = None
    tool_parameters: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_parameters": self.tool_parameters,
            "tool_result": self.tool_result,
            "error": self.error,
        }


@dataclass
class ReActResult:
    """Result of a ReAct execution."""

    status: ReActStatus
    final_answer: Optional[str] = None
    steps: List[ReActStep] = field(default_factory=list)
    total_iterations: int = 0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "status": self.status.value,
            "final_answer": self.final_answer,
            "steps": [s.to_dict() for s in self.steps],
            "total_iterations": self.total_iterations,
            "errors": self.errors,
        }


# System prompt for the ReAct agent
REACT_SYSTEM_PROMPT = """You are a ReAct (Reasoning and Acting) agent that solves tasks by thinking step-by-step and using tools.

For each step, you must respond in one of these formats:

1. THOUGHT: When you need to reason about what to do next
   Format: Thought: <your reasoning>

2. ACTION: When you decide to use a tool
   Format:
   Action: <tool_name>
   Action Input: <json parameters>

3. FINAL ANSWER: When you have completed ALL parts of the user's request
   Format: Final Answer: <your complete answer to the user>

IMPORTANT RULES:
- Always start with a Thought to analyze the task and identify ALL subtasks
- After each Observation (tool result), think about what it means before the next action
- Only use tools that are available to you
- Do NOT make up tool results - always use tools to get real data

CONVERSATION CONTEXT (IMPORTANT):
- Use the conversation history below to understand what the user is referring to
- If the user says "retry", "again", "planA", "yes", etc., refer back to the history
- Previous messages may contain context needed to understand the current request
- Example: If history shows "What's the weather in Tokyo?" and user says "And New York?", this is a weather query for New York

{conversation_history}

ERROR HANDLING (CRITICAL):
- If a tool fails due to parameter validation (e.g., "String must contain at least 1 character"), you MUST retry with corrected parameters
- Read the error message carefully to understand what parameter was wrong and fix it
- For search tools: always provide a non-empty query string (e.g., use "*" or a specific search term)
- Do NOT give up after a single failure - retry with different parameters or try alternative tools
- Only consider a subtask failed after 2-3 retry attempts with different approaches
- MCP Server errors often include specific field names - fix those exact fields

TASK COMPLETION (CRITICAL):
- Break down complex requests into subtasks (e.g., "search Notion AND Google Drive AND send email" = 3 subtasks)
- You MUST complete ALL subtasks before providing a Final Answer
- Do NOT provide Final Answer until you have successfully executed tools for EACH subtask
- If the user asks to send an email, you MUST actually call the email tool - do not skip it
- Track your progress: "Completed: X, Y | Remaining: Z"

ANTI-HALLUCINATION (CRITICAL):
- NEVER claim you did something you didn't do
- NEVER say "email sent" unless you actually executed an email tool and got success
- NEVER say "found X in Google Drive" unless you actually executed a drive search tool
- Your Final Answer will be VALIDATED against actually executed tools
- If you claim an action without executing the tool, your answer will be REJECTED and you must continue
- Only report what you ACTUALLY did and observed

Available Tools:
{tools}

Remember: Think -> Act -> Observe -> Think -> ... -> Final Answer (only after ALL subtasks done)"""


REACT_USER_PROMPT = """Task: {query}

{history}

{instruction}"""


class ReActAgent:
    """
    ReAct agent that executes tasks by interleaving reasoning and acting.

    The agent follows a loop:
    1. Think about what to do next
    2. Execute a tool action
    3. Observe the result
    4. Repeat until task is complete
    """

    def __init__(
        self,
        tools_dict: Dict[str, Any],
        generate_completion_fn: Callable,
        event_emitter: Optional[Callable] = None,
        event_caller: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        process_tool_result_fn: Optional[Callable] = None,
        request: Any = None,
        user: Any = None,
        config: Optional[ReActConfig] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        tools_refresh_fn: Optional[Callable] = None,
    ):
        self.tools = tools_dict
        self.generate_completion_fn = generate_completion_fn
        self.event_emitter = event_emitter
        self.event_caller = event_caller
        self.metadata = metadata or {}
        self.process_tool_result = process_tool_result_fn
        self.request = request
        self.user = user
        self.config = config or DEFAULT_REACT_CONFIG
        self.conversation_history = conversation_history or []
        self.tools_refresh_fn = tools_refresh_fn

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with their descriptions."""
        available = []

        for tool_name, tool_info in self.tools.items():
            # Check blocklist
            if tool_name in self.config.blocked_tools:
                continue

            # Check allowlist
            if self.config.allowed_tools is not None:
                if tool_name not in self.config.allowed_tools:
                    continue

            spec = tool_info.get("spec", {})
            description = spec.get("description", "No description available")
            parameters = spec.get("parameters", {}).get("properties", {})
            required = spec.get("parameters", {}).get("required", [])

            available.append({
                "name": tool_name,
                "description": description,
                "parameters": parameters,
                "required": required,
            })

        return available

    async def _verify_answer_with_llm(
        self,
        query: str,
        final_answer: str,
        steps: List[ReActStep],
    ) -> Optional[str]:
        """
        Use LLM to verify if the final answer is consistent with actual tool executions.

        Returns:
            Correction instruction if hallucination detected, None if answer is valid.
        """
        # Build a summary of what tools were actually called and their results
        tool_executions = []
        for step in steps:
            if step.step_type == ReActStepType.ACTION and step.tool_name:
                tool_executions.append(f"- Called: {step.tool_name}")
            elif step.step_type == ReActStepType.OBSERVATION:
                result_preview = str(step.content)[:500] if step.content else "No result"
                if step.error:
                    tool_executions.append(f"  Result: ERROR - {step.error}")
                else:
                    tool_executions.append(f"  Result: {result_preview}")

        # If no tools were called but answer claims action was done, that's suspicious
        has_tool_calls = any(s.step_type == ReActStepType.ACTION for s in steps)

        if not has_tool_calls:
            # Quick check: does the answer claim success?
            success_indicators = ["successfully", "done", "completed", "added", "sent", "posted", "i've", "has been"]
            if any(ind in final_answer.lower() for ind in success_indicators):
                log.warning("[ReAct] Verification: Answer claims success but no tools were called")
                return (
                    "STOP! You claimed to have completed an action but you have NOT executed ANY tools. "
                    "You MUST call the appropriate tool to actually perform the requested action. "
                    f"Available tools: {list(self.tools.keys())[:10]}..."
                )

        tool_summary = "\n".join(tool_executions) if tool_executions else "No tools were executed."

        # Use LLM to verify
        verification_prompt = f"""You are a verification system. Your job is to check if the Final Answer is consistent with what tools were ACTUALLY executed.

USER REQUEST:
{query}

TOOLS ACTUALLY EXECUTED:
{tool_summary}

FINAL ANSWER BEING VERIFIED:
{final_answer}

VERIFICATION RULES:
1. The Final Answer can ONLY claim actions that were actually performed by executed tools
2. If the user asked to "send email" but no email tool was called, the answer cannot claim email was sent
3. If the user asked to "add reaction" but only a search tool was called, the answer cannot claim reaction was added
4. Search/list/get tools are READ-ONLY - they cannot send, create, add, or modify anything
5. The answer must accurately reflect what the tool results show

RESPOND WITH EXACTLY ONE OF:
- "VALID" - if the Final Answer accurately reflects what tools did
- "INVALID: <reason>" - if the Final Answer claims something that wasn't actually done

Your response:"""

        try:
            response = await self.generate_completion_fn(
                system_prompt="You are a strict verification system that detects hallucinations. Be very strict - if there's any doubt, mark as INVALID.",
                user_prompt=verification_prompt,
            )

            response_clean = response.strip().upper()

            if response_clean.startswith("VALID"):
                log.info("[ReAct] Verification passed: Answer is consistent with tool executions")
                return None
            elif response_clean.startswith("INVALID"):
                # Extract reason after "INVALID:" or "INVALID"
                response_stripped = response.strip()
                if ":" in response_stripped:
                    reason = response_stripped.split(":", 1)[1].strip()
                else:
                    reason = response_stripped[7:].strip()  # Skip "INVALID"

                if not reason:
                    reason = "Answer claims actions that were not performed"

                log.warning(f"[ReAct] Verification FAILED: {reason}")
                return (
                    f"STOP! Your answer was verified and found to be INCORRECT. "
                    f"Reason: {reason}\n\n"
                    f"You must actually execute the required tools before claiming success. "
                    f"Continue by calling the appropriate tool NOW."
                )
            else:
                # Ambiguous response - be safe and reject
                log.warning(f"[ReAct] Verification unclear response: {response[:100]}")
                return None  # Let it pass if unclear

        except Exception as e:
            log.error(f"[ReAct] Verification LLM call failed: {e}")
            return None  # Don't block on verification errors

    def _quick_hallucination_check(
        self,
        query: str,
        final_answer: str,
        steps: List[ReActStep],
    ) -> bool:
        """
        Quick pattern-based check to see if LLM verification is needed.
        Returns True if verification should be run.
        """
        # Always verify if no tools were called but answer claims success
        has_tool_calls = any(s.step_type == ReActStepType.ACTION for s in steps)
        success_indicators = ["successfully", "done", "completed", "added", "sent", "posted", "i've", "has been"]
        claims_success = any(ind in final_answer.lower() for ind in success_indicators)

        if not has_tool_calls and claims_success:
            return True

        # Check for action words in query that might need verification
        action_words = ["send", "add", "create", "post", "schedule", "delete", "update", "react", "thumbsup"]
        query_has_action = any(w in query.lower() for w in action_words)

        if query_has_action and claims_success:
            return True

        return False

    async def _detect_hallucination(
        self,
        query: str,
        final_answer: str,
        steps: List[ReActStep],
    ) -> Optional[str]:
        """
        Detect if the final answer claims things that weren't actually done.
        Uses LLM-based verification for accuracy.

        Returns:
            Correction instruction if hallucination detected, None otherwise.
        """
        # Quick check first - only run expensive LLM verification if needed
        if not self._quick_hallucination_check(query, final_answer, steps):
            return None

        # Run LLM-based verification
        return await self._verify_answer_with_llm(query, final_answer, steps)

    def _format_conversation_history(self) -> str:
        """Format recent conversation history for context."""
        if not self.conversation_history:
            return ""

        # Get recent messages (configured limit)
        recent = self.conversation_history[-self.config.max_history_messages:]

        lines = []
        for msg in recent:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content")

            # Handle None or empty content
            if content is None:
                continue

            # Handle content that may be a list (multimodal)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                # Convert other types to string
                content = str(content)

            # Skip empty content
            if not content.strip():
                continue

            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the system prompt."""
        tools = self._get_available_tools()

        if not tools:
            return "No tools available."

        lines = []
        for tool in tools:
            params_desc = []
            for param_name, param_info in tool["parameters"].items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                required = "(required)" if param_name in tool["required"] else "(optional)"
                params_desc.append(f"    - {param_name} ({param_type}) {required}: {param_desc}")

            params_text = "\n".join(params_desc) if params_desc else "    No parameters"
            lines.append(f"- {tool['name']}: {tool['description']}\n  Parameters:\n{params_text}")

        return "\n\n".join(lines)

    def _format_history(self, steps: List[ReActStep]) -> str:
        """Format previous steps for the prompt."""
        if not steps:
            return ""

        lines = []
        for step in steps:
            if step.step_type == ReActStepType.THOUGHT:
                lines.append(f"Thought: {step.content or ''}")
            elif step.step_type == ReActStepType.ACTION:
                lines.append(f"Action: {step.tool_name or 'unknown'}")
                params = step.tool_parameters if step.tool_parameters else {}
                lines.append(f"Action Input: {json.dumps(params, ensure_ascii=False)}")
            elif step.step_type == ReActStepType.OBSERVATION:
                # Truncate long observations
                content = step.content or ""
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                lines.append(f"Observation: {content}")
            elif step.step_type == ReActStepType.VERIFICATION:
                # Show verification failures to guide the agent
                lines.append(f"Verification: {step.content or ''}")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract thought, action, or final answer.

        Returns:
            {
                "type": "thought" | "action" | "final_answer",
                "content": str,
                "tool_name": str (if action),
                "tool_parameters": dict (if action),
            }
        """
        response = response.strip()

        # Check for Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
        if final_match:
            return {
                "type": "final_answer",
                "content": final_match.group(1).strip(),
            }

        # Check for Action
        action_match = re.search(
            r"Action:\s*([^\n]+)\s*\nAction Input:\s*(.+?)(?=\n(?:Thought|Action|Final Answer|Observation):|$)",
            response,
            re.DOTALL | re.IGNORECASE
        )
        if action_match:
            tool_name = action_match.group(1).strip()
            params_str = action_match.group(2).strip()

            # Parse parameters as JSON
            try:
                # Try to parse the entire params_str as JSON first
                tool_parameters = json.loads(params_str)
            except json.JSONDecodeError:
                # If that fails, try to find a JSON object (non-greedy match for balanced braces)
                try:
                    # Find first { and try to parse from there
                    brace_start = params_str.find('{')
                    if brace_start != -1:
                        # Find matching closing brace by counting
                        depth = 0
                        for i, char in enumerate(params_str[brace_start:], brace_start):
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 0:
                                    json_str = params_str[brace_start:i+1]
                                    tool_parameters = json.loads(json_str)
                                    break
                        else:
                            # No matching brace found
                            tool_parameters = {"input": params_str}
                    else:
                        # No JSON found, treat as simple string parameter
                        tool_parameters = {"input": params_str}
                except (json.JSONDecodeError, ValueError):
                    tool_parameters = {"input": params_str}

            # Extract thought if present before action
            thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:)", response, re.DOTALL | re.IGNORECASE)
            thought = thought_match.group(1).strip() if thought_match else None

            return {
                "type": "action",
                "content": thought or f"Executing {tool_name}",
                "tool_name": tool_name,
                "tool_parameters": tool_parameters,
            }

        # Check for Thought only
        thought_match = re.search(r"Thought:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            return {
                "type": "thought",
                "content": thought_match.group(1).strip(),
            }

        # If nothing matches, treat as thought
        return {
            "type": "thought",
            "content": response,
        }

    def _parse_mcp_error(self, result: Any) -> Optional[str]:
        """
        Parse MCP-style error responses.
        MCP errors are indicated by isError=True in the response.
        Returns error message if found, None otherwise.
        """
        if not result:
            return None

        # Check for MCP content with isError
        if isinstance(result, dict):
            # Direct isError in result
            if result.get("isError"):
                content = result.get("content", [])
                if isinstance(content, list):
                    error_texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            error_texts.append(item.get("text", ""))
                    return " ".join(error_texts) if error_texts else "MCP tool returned error"
                return str(content) if content else "MCP tool returned error"

            # Check for error in results array
            if "results" in result:
                for item in result.get("results", []):
                    if isinstance(item, dict) and item.get("isError"):
                        return self._parse_mcp_error(item)

            # Check for "error" key in result (common pattern)
            if "error" in result and result.get("error"):
                error_val = result["error"]
                if isinstance(error_val, str):
                    return error_val
                elif isinstance(error_val, dict):
                    return error_val.get("message", str(error_val))

        return None

    def _format_error_for_retry(self, error: str) -> str:
        """Format error message with helpful retry guidance."""
        guidance = []

        error_lower = error.lower()

        # Detect validation errors and provide specific guidance
        if "must contain at least" in error_lower:
            guidance.append("The parameter needs a non-empty value. Try using '*' for wildcard search or provide a specific term.")
        if "required" in error_lower and ("missing" in error_lower or "field" in error_lower):
            guidance.append("A required parameter is missing. Check the tool's parameter specification.")
        if "invalid" in error_lower or "validation" in error_lower:
            guidance.append("Parameter format may be incorrect. Check the expected type and format.")
        if "timeout" in error_lower:
            guidance.append("The operation took too long. Try with simpler parameters or break into smaller steps.")
        if "not found" in error_lower or "does not exist" in error_lower:
            guidance.append("The resource may not exist. Verify the ID or name is correct.")

        if guidance:
            return f"{error}\n\nRetry Guidance:\n- " + "\n- ".join(guidance)
        return error

    async def _execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool and return the result."""
        # Check if tool exists, try refresh if not found
        if tool_name not in self.tools:
            # Try to refresh tools if callback provided
            if self.tools_refresh_fn:
                log.info(f"[ReAct] Tool '{tool_name}' not found, attempting refresh...")
                try:
                    refreshed_tools = await self.tools_refresh_fn()
                    if refreshed_tools:
                        self.tools = refreshed_tools
                        log.info(f"[ReAct] Tools refreshed, now have {len(self.tools)} tools")
                except Exception as e:
                    log.warning(f"[ReAct] Tool refresh failed: {e}")

            # Check again after refresh
            if tool_name not in self.tools:
                available = list(self.tools.keys())[:10]
                return {
                    "status": "error",
                    "error": f"Tool '{tool_name}' not found. Available tools: {available}...",
                    "result": None,
                }

        tool = self.tools[tool_name]
        tool_type = tool.get("type", "")
        direct_tool = tool.get("direct", False)

        # Filter to allowed parameters based on spec
        spec = tool.get("spec", {})
        allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
        filtered_params = {k: v for k, v in parameters.items() if k in allowed_params}

        log.info(f"[ReAct] Executing tool '{tool_name}' with params: {list(filtered_params.keys())}")

        try:
            if direct_tool and self.event_caller:
                # Direct tool execution via event_caller
                tool_result = await asyncio.wait_for(
                    self.event_caller(
                        {
                            "type": "execute:tool",
                            "data": {
                                "id": str(uuid4()),
                                "name": tool_name,
                                "params": filtered_params,
                                "server": tool.get("server", {}),
                                "session_id": self.metadata.get("session_id"),
                            },
                        }
                    ),
                    timeout=self.config.tool_timeout,
                )
            else:
                # Regular callable execution
                tool_function = tool.get("callable")
                if tool_function:
                    tool_result = await asyncio.wait_for(
                        tool_function(**filtered_params),
                        timeout=self.config.tool_timeout,
                    )
                else:
                    raise ValueError(f"No callable found for tool '{tool_name}'")

            # Process the result
            files = []
            embeds = []
            if self.process_tool_result:
                tool_result, files, embeds = self.process_tool_result(
                    self.request,
                    tool_name,
                    tool_result,
                    tool_type,
                    direct_tool,
                    self.metadata,
                    self.user,
                )

            # Emit files and embeds to client
            if self.event_emitter:
                if files:
                    await self.event_emitter(
                        {
                            "type": "files",
                            "data": {"files": files},
                        }
                    )
                if embeds:
                    await self.event_emitter(
                        {
                            "type": "embeds",
                            "data": {"embeds": embeds},
                        }
                    )

            # Check for MCP-style errors in successful responses
            mcp_error = self._parse_mcp_error(tool_result)
            if mcp_error:
                log.warning(f"[ReAct] Tool '{tool_name}' returned MCP error: {mcp_error}")
                formatted_error = self._format_error_for_retry(mcp_error)
                return {
                    "status": "error",
                    "error": formatted_error,
                    "result": tool_result,  # Include result for debugging
                }

            log.info(f"[ReAct] Tool '{tool_name}' executed successfully")
            return {
                "status": "success",
                "result": tool_result,
                "error": None,
                "files": files,
                "embeds": embeds,
            }

        except asyncio.TimeoutError:
            log.error(f"[ReAct] Tool '{tool_name}' timed out after {self.config.tool_timeout}s")
            return {
                "status": "error",
                "error": f"Tool execution timed out after {self.config.tool_timeout}s. Try again or use a different approach.",
                "result": None,
            }
        except Exception as e:
            error_str = str(e)
            log.error(f"[ReAct] Tool '{tool_name}' execution failed: {error_str}")
            formatted_error = self._format_error_for_retry(error_str)
            return {
                "status": "error",
                "error": formatted_error,
                "result": None,
            }

    async def _emit_step(self, step: ReActStep) -> None:
        """Emit a step event to the client."""
        if not self.event_emitter or not self.config.emit_step_events:
            return

        step_type_icons = {
            ReActStepType.THOUGHT: "thinking",
            ReActStepType.ACTION: "tool_execution",
            ReActStepType.OBSERVATION: "observation",
            ReActStepType.VERIFICATION: "verification",
            ReActStepType.FINAL_ANSWER: "complete",
        }

        await self.event_emitter(
            {
                "type": "status",
                "data": {
                    "action": step_type_icons.get(step.step_type, "react"),
                    "description": self._get_step_description(step),
                    "done": step.step_type == ReActStepType.FINAL_ANSWER,
                },
            }
        )

    def _get_step_description(self, step: ReActStep) -> str:
        """Get a human-readable description of a step."""
        if step.step_type == ReActStepType.THOUGHT:
            # Truncate long thoughts
            content = step.content[:100] + "..." if len(step.content) > 100 else step.content
            return f"Thinking: {content}"
        elif step.step_type == ReActStepType.ACTION:
            return f"Executing: {step.tool_name}"
        elif step.step_type == ReActStepType.OBSERVATION:
            return f"Observed result from {step.tool_name or 'tool'}"
        elif step.step_type == ReActStepType.VERIFICATION:
            return "Verifying answer against tool executions"
        elif step.step_type == ReActStepType.FINAL_ANSWER:
            return "Task completed"
        return "Processing..."

    async def run(self, query: str) -> ReActResult:
        """
        Run the ReAct agent on a query.

        Args:
            query: The user's query or task to solve

        Returns:
            ReActResult with the final answer and execution history
        """
        result = ReActResult(status=ReActStatus.RUNNING)
        consecutive_failures = 0
        iteration = 0

        log.info(f"[ReAct] Starting execution for query: {query[:100]}...")

        # Build system prompt with available tools and conversation history
        tools_description = self._format_tools_for_prompt()
        conversation_context = self._format_conversation_history()
        if conversation_context:
            conversation_section = f"Recent Conversation History:\n{conversation_context}"
        else:
            conversation_section = "No previous conversation history."

        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools=tools_description,
            conversation_history=conversation_section,
        )

        while iteration < self.config.max_iterations:
            iteration += 1
            result.total_iterations = iteration

            log.info(f"[ReAct] Iteration {iteration}/{self.config.max_iterations}")

            # Build user prompt with history
            history = self._format_history(result.steps)

            if not result.steps:
                instruction = "Begin by thinking about how to approach this task."
            else:
                last_step = result.steps[-1]
                if last_step.step_type == ReActStepType.OBSERVATION:
                    instruction = "Based on the observation above, think about what to do next or provide the final answer."
                elif last_step.step_type == ReActStepType.THOUGHT:
                    instruction = "Decide on your next action or provide the final answer."
                else:
                    instruction = "Continue with your reasoning and actions."

            user_prompt = REACT_USER_PROMPT.format(
                query=query,
                history=history,
                instruction=instruction,
            )

            try:
                # Get LLM response
                response = await self.generate_completion_fn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

                # Parse the response
                parsed = self._parse_response(response)

                if parsed["type"] == "final_answer":
                    # Before accepting, verify with LLM that answer matches actual tool executions
                    log.info("[ReAct] Verifying final answer against tool executions...")

                    # Check for hallucination using LLM-based verification
                    hallucination_correction = await self._detect_hallucination(
                        query=query,
                        final_answer=parsed["content"],
                        steps=result.steps,
                    )

                    if hallucination_correction:
                        # Hallucination detected! Don't accept this answer
                        # Add a VERIFICATION step recording the rejection
                        # The agent will see this in the history via _format_history
                        verification_step = ReActStep(
                            step_number=len(result.steps) + 1,
                            step_type=ReActStepType.VERIFICATION,
                            content=f"[VERIFICATION FAILED] {hallucination_correction}",
                        )
                        result.steps.append(verification_step)
                        await self._emit_step(verification_step)

                        log.warning("[ReAct] Verification rejected final answer, continuing...")
                        continue  # Skip to next iteration - agent will see verification failure in history

                    # Verification passed - Task complete!
                    step = ReActStep(
                        step_number=len(result.steps) + 1,
                        step_type=ReActStepType.FINAL_ANSWER,
                        content=parsed["content"],
                    )
                    result.steps.append(step)
                    await self._emit_step(step)

                    result.status = ReActStatus.COMPLETED
                    result.final_answer = parsed["content"]
                    log.info(f"[ReAct] Completed after {iteration} iterations")
                    return result

                elif parsed["type"] == "action":
                    # Record thought if present
                    if parsed.get("content"):
                        thought_step = ReActStep(
                            step_number=len(result.steps) + 1,
                            step_type=ReActStepType.THOUGHT,
                            content=parsed["content"],
                        )
                        result.steps.append(thought_step)
                        await self._emit_step(thought_step)

                    # Record action
                    action_step = ReActStep(
                        step_number=len(result.steps) + 1,
                        step_type=ReActStepType.ACTION,
                        content=f"Execute {parsed['tool_name']}",
                        tool_name=parsed["tool_name"],
                        tool_parameters=parsed["tool_parameters"],
                    )
                    result.steps.append(action_step)
                    await self._emit_step(action_step)

                    # Execute the tool
                    tool_result = await self._execute_tool(
                        parsed["tool_name"],
                        parsed["tool_parameters"],
                    )

                    # Record observation
                    if tool_result["status"] == "success":
                        # Safely serialize the result
                        try:
                            observation_content = json.dumps(tool_result["result"], ensure_ascii=False, indent=2)
                        except (TypeError, ValueError):
                            # Handle non-serializable results
                            observation_content = str(tool_result["result"])

                        consecutive_failures = 0

                        # Add to sources
                        tool = self.tools.get(parsed["tool_name"], {})
                        tool_id = tool.get("tool_id", "")
                        source_name = f"{tool_id}/{parsed['tool_name']}" if tool_id else parsed["tool_name"]
                        result.sources.append(
                            {
                                "source": {"name": source_name},
                                "document": [str(tool_result["result"])],
                                "metadata": [
                                    {
                                        "source": source_name,
                                        "parameters": parsed["tool_parameters"],
                                    }
                                ],
                                "tool_result": True,
                            }
                        )
                    else:
                        observation_content = f"Error: {tool_result['error']}"
                        consecutive_failures += 1
                        result.errors.append({
                            "iteration": iteration,
                            "tool": parsed["tool_name"],
                            "error": tool_result["error"],
                        })

                        if consecutive_failures >= self.config.max_consecutive_failures:
                            log.warning(f"[ReAct] Max consecutive failures reached ({consecutive_failures})")
                            if not self.config.continue_on_failure:
                                result.status = ReActStatus.FAILED
                                result.final_answer = f"Task failed after {consecutive_failures} consecutive tool failures."
                                return result

                    observation_step = ReActStep(
                        step_number=len(result.steps) + 1,
                        step_type=ReActStepType.OBSERVATION,
                        content=observation_content,
                        tool_name=parsed["tool_name"],
                        tool_result=tool_result.get("result"),
                        error=tool_result.get("error"),
                    )
                    result.steps.append(observation_step)
                    await self._emit_step(observation_step)

                elif parsed["type"] == "thought":
                    # Record thought
                    thought_step = ReActStep(
                        step_number=len(result.steps) + 1,
                        step_type=ReActStepType.THOUGHT,
                        content=parsed["content"],
                    )
                    result.steps.append(thought_step)
                    await self._emit_step(thought_step)

            except Exception as e:
                log.error(f"[ReAct] Error in iteration {iteration}: {e}")
                result.errors.append({
                    "iteration": iteration,
                    "error": str(e),
                })
                consecutive_failures += 1

                if consecutive_failures >= self.config.max_consecutive_failures:
                    if not self.config.continue_on_failure:
                        result.status = ReActStatus.FAILED
                        result.final_answer = f"Task failed due to errors: {e}"
                        return result

        # Max iterations reached
        log.warning(f"[ReAct] Max iterations reached ({self.config.max_iterations})")
        result.status = ReActStatus.MAX_ITERATIONS
        result.final_answer = "I was unable to complete the task within the allowed number of steps. Here's what I found so far based on my observations."

        # Try to generate a summary from what we have
        if result.sources:
            try:
                summary_response = await self.generate_completion_fn(
                    system_prompt="You are a helpful assistant. Summarize the findings from the tool results.",
                    user_prompt=f"Task: {query}\n\nTool results gathered:\n{self._format_history(result.steps)}\n\nProvide a summary of what was found:",
                )
                result.final_answer = summary_response
            except Exception:
                pass

        return result


async def run_react_agent(
    query: str,
    tools_dict: Dict[str, Any],
    generate_completion_fn: Callable,
    event_emitter: Optional[Callable] = None,
    event_caller: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None,
    process_tool_result_fn: Optional[Callable] = None,
    request: Any = None,
    user: Any = None,
    config: Optional[ReActConfig] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    tools_refresh_fn: Optional[Callable] = None,
) -> ReActResult:
    """
    Convenience function to run a ReAct agent on a query.

    Args:
        query: The user's query or task
        tools_dict: Dictionary of available tools
        generate_completion_fn: Async function to generate LLM completions
        event_emitter: Optional event emitter for status updates
        event_caller: Optional event caller for direct tool execution
        metadata: Optional metadata dict
        process_tool_result_fn: Optional function to process tool results
        request: Optional request object
        user: Optional user object
        config: Optional ReActConfig
        conversation_history: Optional list of previous messages for context
        tools_refresh_fn: Optional async function to refresh tools if not found

    Returns:
        ReActResult with final answer and execution history
    """
    agent = ReActAgent(
        tools_dict=tools_dict,
        generate_completion_fn=generate_completion_fn,
        event_emitter=event_emitter,
        event_caller=event_caller,
        metadata=metadata,
        process_tool_result_fn=process_tool_result_fn,
        request=request,
        user=user,
        config=config,
        conversation_history=conversation_history,
        tools_refresh_fn=tools_refresh_fn,
    )

    return await agent.run(query)
