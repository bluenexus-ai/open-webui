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
    max_iterations: int = 10

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

3. FINAL ANSWER: When you have enough information to answer the user's question
   Format: Final Answer: <your complete answer to the user>

IMPORTANT RULES:
- Always start with a Thought to analyze the task
- After each Observation (tool result), think about what it means before the next action
- Only use tools that are available to you
- If a tool fails, think about alternatives or whether to retry
- When you have enough information, provide a Final Answer
- Be concise but thorough in your reasoning
- Do NOT make up tool results - always use tools to get real data

Available Tools:
{tools}

Remember: Think -> Act -> Observe -> Think -> ... -> Final Answer"""


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
                lines.append(f"Thought: {step.content}")
            elif step.step_type == ReActStepType.ACTION:
                lines.append(f"Action: {step.tool_name}")
                lines.append(f"Action Input: {json.dumps(step.tool_parameters, ensure_ascii=False)}")
            elif step.step_type == ReActStepType.OBSERVATION:
                # Truncate long observations
                content = step.content
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                lines.append(f"Observation: {content}")

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
                # Try to find JSON object in the params string
                json_match = re.search(r'\{.*\}', params_str, re.DOTALL)
                if json_match:
                    tool_parameters = json.loads(json_match.group())
                else:
                    # If no JSON found, treat as simple string parameter
                    tool_parameters = {"input": params_str}
            except json.JSONDecodeError:
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

    async def _execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool and return the result."""
        if tool_name not in self.tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not found",
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
                "error": f"Tool execution timed out after {self.config.tool_timeout}s",
                "result": None,
            }
        except Exception as e:
            log.error(f"[ReAct] Tool '{tool_name}' execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
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

        # Build system prompt with available tools
        tools_description = self._format_tools_for_prompt()
        system_prompt = REACT_SYSTEM_PROMPT.format(tools=tools_description)

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
                    # Task complete!
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
                        observation_content = json.dumps(tool_result["result"], ensure_ascii=False, indent=2)
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
    )

    return await agent.run(query)
