import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from open_webui.utils.workflow.types import (
    ExecutionContext,
    ExecutionPlan,
    ExecutionStep,
    ExecutionStatus,
    ToolCall,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MAIN", logging.INFO))


class ParallelToolExecutor:
    """
    Executes tool calls in parallel within steps while respecting dependencies.
    """

    # Special built-in tool for LLM content generation
    LLM_COMPOSE_TOOL = "__llm_compose__"

    def __init__(
        self,
        tools_dict: Dict[str, Any],
        event_emitter: Optional[Callable] = None,
        event_caller: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        process_tool_result_fn: Optional[Callable] = None,
        request: Any = None,
        user: Any = None,
        generate_completion_fn: Optional[Callable] = None,
    ):
        self.tools = tools_dict
        self.event_emitter = event_emitter
        self.event_caller = event_caller
        self.metadata = metadata or {}
        self.process_tool_result = process_tool_result_fn
        self.request = request
        self.user = user
        self.generate_completion_fn = generate_completion_fn

    def _get_nested_value(self, data: Any, path: str) -> Any:
        """
        Extract a nested value from data using dot notation path.
        Returns the value if found, None if path is invalid.

        Examples:
            _get_nested_value({"a": {"b": 1}}, "a.b") -> 1
            _get_nested_value({"items": [{"id": 1}]}, "items.0.id") -> 1
        """
        if not path:
            return data

        parts = path.split(".")
        current = data

        for part in parts:
            if current is None:
                return None

            # Handle dict access
            if isinstance(current, dict):
                current = current.get(part)
            # Handle list access with numeric index
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return None
                except ValueError:
                    # Try to find item by key in list of dicts
                    return None
            # Handle string that might be JSON
            elif isinstance(current, str):
                try:
                    parsed = json.loads(current)
                    if isinstance(parsed, dict):
                        current = parsed.get(part)
                    elif isinstance(parsed, list):
                        try:
                            idx = int(part)
                            current = parsed[idx] if 0 <= idx < len(parsed) else None
                        except ValueError:
                            return None
                    else:
                        return None
                except (json.JSONDecodeError, TypeError):
                    return None
            else:
                return None

        return current

    def _resolve_parameter_references(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Resolve step result references in parameters.

        Supported formats:
        - {{step_N_result_toolName}} - Full result from tool in step N
        - {{step_N_result_toolName.property}} - Specific property from result
        - {{step_N_result_toolName.nested.path}} - Nested property access
        - {{step_N_result_toolName.array.0.id}} - Array index access

        Falls back to full result if property path is invalid.
        """
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                resolved_value = self._resolve_string_references(value, context)
                resolved[key] = resolved_value
            elif isinstance(value, dict):
                resolved[key] = self._resolve_parameter_references(value, context)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_parameter_references({"_": item}, context).get("_", item)
                    if isinstance(item, (dict, str))
                    else item
                    for item in value
                ]
            else:
                resolved[key] = value

        return resolved

    def _resolve_string_references(
        self,
        value: str,
        context: ExecutionContext,
    ) -> str:
        """
        Resolve all {{...}} references in a string value.
        """
        # Pattern to match {{step_N_result_toolName}} with optional property path
        # Groups: (1) step_num, (2) tool_name, (3) optional property path including the dot
        pattern = r"\{\{step_(\d+)_result_(\w+)((?:\.[a-zA-Z0-9_]+)*)\}\}"

        def replace_match(match):
            step_num = int(match.group(1))
            tool_name = match.group(2)
            property_path = match.group(3)  # e.g., ".id" or ".data.items" or ""

            # Remove leading dot from property path
            if property_path:
                property_path = property_path[1:]  # Remove the leading "."

            # Get the step results
            if step_num not in context.step_results:
                log.warning(f"[Executor] Step {step_num} not found in results, keeping reference as-is")
                return match.group(0)

            step_result = context.step_results[step_num]

            if tool_name not in step_result:
                log.warning(f"[Executor] Tool '{tool_name}' not found in step {step_num} results, keeping reference as-is")
                return match.group(0)

            result = step_result[tool_name]

            # If there's a property path, try to extract the nested value
            if property_path:
                nested_value = self._get_nested_value(result, property_path)
                if nested_value is not None:
                    result = nested_value
                    log.debug(f"[Executor] Resolved {{{{step_{step_num}_result_{tool_name}.{property_path}}}}} to nested value")
                else:
                    # Property path invalid - fall back to full result and log warning
                    log.warning(
                        f"[Executor] Property path '{property_path}' not found in step_{step_num}_result_{tool_name}, "
                        f"falling back to full result"
                    )

            # Convert result to string for replacement
            if isinstance(result, dict) or isinstance(result, list):
                return json.dumps(result, ensure_ascii=False)
            elif result is None:
                return ""
            else:
                return str(result)

        resolved_value = re.sub(pattern, replace_match, value)

        # Check for any remaining unresolved references and log warning
        remaining_refs = re.findall(r"\{\{[^}]+\}\}", resolved_value)
        if remaining_refs:
            log.warning(
                f"[Executor] Unresolved template references found: {remaining_refs}. "
                f"Expected format: {{{{step_N_result_toolName}}}} or {{{{step_N_result_toolName.property}}}}"
            )

        return resolved_value

    async def _execute_llm_compose(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Execute the special __llm_compose__ tool that uses LLM to generate content.

        Parameters:
        - data: The data to base the composition on (can be a step reference)
        - prompt: Instructions for the LLM on what to generate
        - output_key: Optional key name for the output (default: "content")

        Returns generated content that can be used in subsequent steps.
        """
        tool_name = tool_call.name

        if not self.generate_completion_fn:
            log.error("[Executor] LLM compose requested but no generate_completion_fn provided")
            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "error",
                "error": "LLM composition not available",
                "result": None,
                "files": [],
                "embeds": [],
            }

        # Resolve parameter references to get actual data
        resolved_params = self._resolve_parameter_references(tool_call.parameters, context)

        data = resolved_params.get("data", "")
        prompt = resolved_params.get("prompt", "Generate content based on the provided data.")
        output_key = resolved_params.get("output_key", "content")

        # Emit status
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "llm_compose",
                        "description": "Generating content with LLM...",
                        "done": False,
                    },
                }
            )

        try:
            # Build the LLM prompt
            system_prompt = """You are a helpful assistant that generates well-formatted content based on provided data.
Your output should be clean, human-readable text suitable for the requested purpose.
Do NOT include any JSON, code blocks, or raw data structures in your response unless specifically asked.
Focus on creating polished, professional content."""

            user_prompt = f"""Based on the following data, {prompt}

Data:
{data}

Generate the requested content directly without any preamble or explanation."""

            # Call the LLM
            result = await self.generate_completion_fn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            log.info(f"[Executor] LLM compose generated content successfully")

            # Return result as a dict with the output_key
            output = {output_key: result}

            if self.event_emitter:
                await self.event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "action": "llm_compose",
                            "description": "Content generated",
                            "done": True,
                        },
                    }
                )

            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "success",
                "result": output,
                "error": None,
                "files": [],
                "embeds": [],
            }

        except Exception as e:
            log.error(f"[Executor] LLM compose failed: {e}")
            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "error",
                "error": str(e),
                "result": None,
                "files": [],
                "embeds": [],
            }

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute a single tool call and return the result"""
        tool_name = tool_call.name

        # Handle special __llm_compose__ tool
        if tool_name == self.LLM_COMPOSE_TOOL:
            return await self._execute_llm_compose(tool_call, context)

        if tool_name not in self.tools:
            log.warning(f"[Executor] Tool '{tool_name}' not found in tools dict")
            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "error",
                "error": f"Tool '{tool_name}' not found",
                "result": None,
                "files": [],
                "embeds": [],
            }

        tool = self.tools[tool_name]
        tool_type = tool.get("type", "")
        direct_tool = tool.get("direct", False)

        # Resolve parameter references from previous steps
        resolved_params = self._resolve_parameter_references(tool_call.parameters, context)

        # Filter to allowed parameters based on spec
        spec = tool.get("spec", {})
        allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
        filtered_params = {k: v for k, v in resolved_params.items() if k in allowed_params}

        log.info(f"[Executor] Executing tool '{tool_name}' with params: {list(filtered_params.keys())}")

        try:
            if direct_tool and self.event_caller:
                # Direct tool execution via event_caller
                tool_result = await self.event_caller(
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
                )
            else:
                # Regular callable execution
                tool_function = tool.get("callable")
                if tool_function:
                    tool_result = await tool_function(**filtered_params)
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

            log.info(f"[Executor] Tool '{tool_name}' executed successfully")
            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "success",
                "result": tool_result,
                "error": None,
                "files": files,
                "embeds": embeds,
            }

        except Exception as e:
            log.error(f"[Executor] Tool '{tool_name}' execution failed: {e}")
            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "error",
                "error": str(e),
                "result": None,
                "files": [],
                "embeds": [],
            }

    async def execute_step(
        self,
        step: ExecutionStep,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Execute all tool calls in a step in parallel.
        Returns results keyed by tool name.
        """
        step.status = ExecutionStatus.IN_PROGRESS
        log.info(f"[Executor] Starting step {step.step_number}: {step.description} ({len(step.tool_calls)} tools)")

        # Emit step start event
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "tool_execution",
                        "description": f"Step {step.step_number}: {step.description}",
                        "done": False,
                    },
                }
            )

        # Create tasks for all tool calls in this step (parallel execution)
        tasks = [self._execute_single_tool(tc, context) for tc in step.tool_calls]

        # Execute in parallel with asyncio.gather
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        # Note: step_results is keyed by tool name. If the same tool is called
        # multiple times in one step, only the last result is stored for parameter
        # reference resolution. However, all results are preserved in context.sources.
        step_results = {}
        all_success = True

        for i, result in enumerate(results):
            tool_call = step.tool_calls[i]

            if isinstance(result, Exception):
                log.error(f"[Executor] Tool execution exception for '{tool_call.name}': {result}")
                step_results[tool_call.name] = None
                tool_call.status = ExecutionStatus.FAILED
                tool_call.error = str(result)
                all_success = False
                context.errors.append(
                    {
                        "step": step.step_number,
                        "tool": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "error": str(result),
                    }
                )
            elif result.get("status") == "error":
                step_results[tool_call.name] = None
                tool_call.status = ExecutionStatus.FAILED
                tool_call.error = result.get("error")
                all_success = False
                context.errors.append(
                    {
                        "step": step.step_number,
                        "tool": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "error": result.get("error"),
                    }
                )
            else:
                step_results[tool_call.name] = result.get("result")
                tool_call.status = ExecutionStatus.COMPLETED
                tool_call.result = result.get("result")

                # Add to sources for citation
                tool = self.tools.get(tool_call.name, {})
                tool_id = tool.get("tool_id", "")
                source_name = f"{tool_id}/{tool_call.name}" if tool_id else tool_call.name

                context.sources.append(
                    {
                        "source": {"name": source_name},
                        "document": [str(result.get("result", ""))],
                        "metadata": [
                            {
                                "source": source_name,
                                "parameters": tool_call.parameters,
                            }
                        ],
                        "tool_result": True,
                    }
                )

                # Check for file_handler flag
                if tool.get("metadata", {}).get("file_handler", False):
                    context.skip_files = True

        step.status = ExecutionStatus.COMPLETED if all_success else ExecutionStatus.FAILED

        # Emit step completion
        if self.event_emitter:
            status_msg = "completed" if all_success else "completed with errors"
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "tool_execution",
                        "description": f"Step {step.step_number} {status_msg}",
                        "done": True,
                    },
                }
            )

        log.info(f"[Executor] Step {step.step_number} finished: {len(step_results)} results, success={all_success}")
        return step_results

    def _validate_and_fix_plan_references(self, plan: ExecutionPlan) -> None:
        """
        Validate and auto-correct step references in the execution plan.
        This helps fix common LLM hallucinations like wrong step numbers or tool names.
        """
        # Build a map of step_number -> list of tool names executed in that step
        step_tools: Dict[int, List[str]] = {}
        all_tool_names: List[str] = []

        for step in plan.steps:
            step_tools[step.step_number] = [tc.name for tc in step.tool_calls]
            all_tool_names.extend(step_tools[step.step_number])

        # Pattern to match references
        ref_pattern = r"\{\{step_(\d+)_result_(\w+)((?:\.[a-zA-Z0-9_]+)*)\}\}"

        for step in plan.steps:
            for tool_call in step.tool_calls:
                self._fix_parameters_references(
                    tool_call.parameters,
                    step.step_number,
                    step_tools,
                    all_tool_names,
                    ref_pattern,
                )

    def _fix_parameters_references(
        self,
        parameters: Dict[str, Any],
        current_step: int,
        step_tools: Dict[int, List[str]],
        all_tool_names: List[str],
        ref_pattern: str,
    ) -> None:
        """
        Recursively fix references in parameters dict.
        Modifies parameters in-place.
        """
        for key, value in list(parameters.items()):
            if isinstance(value, str):
                parameters[key] = self._fix_string_references(
                    value, current_step, step_tools, all_tool_names, ref_pattern
                )
            elif isinstance(value, dict):
                self._fix_parameters_references(
                    value, current_step, step_tools, all_tool_names, ref_pattern
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        value[i] = self._fix_string_references(
                            item, current_step, step_tools, all_tool_names, ref_pattern
                        )
                    elif isinstance(item, dict):
                        self._fix_parameters_references(
                            item, current_step, step_tools, all_tool_names, ref_pattern
                        )

    def _fix_string_references(
        self,
        value: str,
        current_step: int,
        step_tools: Dict[int, List[str]],
        all_tool_names: List[str],
        ref_pattern: str,
    ) -> str:
        """
        Fix references in a string value by correcting step numbers and tool names.
        """
        def fix_match(match):
            ref_step = int(match.group(1))
            ref_tool = match.group(2)
            ref_property = match.group(3) or ""
            original = match.group(0)

            # Check if reference is valid
            if ref_step < current_step and ref_step in step_tools:
                if ref_tool in step_tools[ref_step]:
                    # Reference is valid, keep as-is
                    return original

            # Try to find the correct step and tool
            corrected_step = None
            corrected_tool = None

            # Strategy 1: Look for exact tool name in previous steps
            for step_num in range(current_step - 1, 0, -1):
                if step_num in step_tools and ref_tool in step_tools[step_num]:
                    corrected_step = step_num
                    corrected_tool = ref_tool
                    break

            # Strategy 2: Look for similar tool name (fuzzy match) in previous steps
            if corrected_tool is None:
                best_match = None
                best_score = 0

                for step_num in range(current_step - 1, 0, -1):
                    if step_num not in step_tools:
                        continue
                    for tool_name in step_tools[step_num]:
                        # Check if ref_tool is a substring or similar
                        score = self._similarity_score(ref_tool, tool_name)
                        if score > best_score and score > 0.5:
                            best_score = score
                            best_match = (step_num, tool_name)

                if best_match:
                    corrected_step, corrected_tool = best_match

            # Apply correction if found
            if corrected_step is not None and corrected_tool is not None:
                corrected_ref = f"{{{{step_{corrected_step}_result_{corrected_tool}{ref_property}}}}}"
                log.warning(
                    f"[Executor] Auto-corrected invalid reference: {original} -> {corrected_ref}"
                )
                return corrected_ref

            # No valid correction found, log warning
            log.warning(
                f"[Executor] Invalid step reference '{original}' in step {current_step}. "
                f"Referenced step {ref_step} tool '{ref_tool}' not found in previous steps. "
                f"Available: {step_tools}"
            )
            return original

        return re.sub(ref_pattern, fix_match, value)

    def _similarity_score(self, s1: str, s2: str) -> float:
        """
        Calculate similarity score between two strings.
        Returns a value between 0 and 1.
        """
        s1_lower = s1.lower()
        s2_lower = s2.lower()

        # Exact match
        if s1_lower == s2_lower:
            return 1.0

        # One is substring of the other
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return 0.8

        # Check common parts (split by underscore)
        parts1 = set(s1_lower.split("_"))
        parts2 = set(s2_lower.split("_"))
        common = parts1 & parts2

        if common:
            return len(common) / max(len(parts1), len(parts2))

        return 0.0

    async def execute_plan(
        self,
        plan: ExecutionPlan,
    ) -> ExecutionContext:
        """
        Execute the entire execution plan step by step.
        Steps are executed sequentially, tools within each step execute in parallel.
        """
        context = ExecutionContext()

        log.info(f"[Executor] Starting execution plan: {plan.total_tools} tools in {len(plan.steps)} steps")
        if plan.reasoning:
            log.info(f"[Executor] Plan reasoning: {plan.reasoning}")

        # Validate and fix references before execution
        self._validate_and_fix_plan_references(plan)

        for step in plan.steps:
            step_results = await self.execute_step(step, context)
            context.step_results[step.step_number] = step_results

        log.info(f"[Executor] Execution plan complete: {len(context.sources)} sources, {len(context.errors)} errors")
        return context


def parse_execution_plan(content: str, tools_dict: Dict[str, Any]) -> Optional[ExecutionPlan]:
    """
    Parse LLM response content into an ExecutionPlan.
    Supports both new format (execution_plan) and legacy format (tool_calls).
    """
    # Built-in tools that don't need to be in tools_dict
    BUILTIN_TOOLS = {ParallelToolExecutor.LLM_COMPOSE_TOOL}

    def is_valid_tool(tool_name: str) -> bool:
        return tool_name in tools_dict or tool_name in BUILTIN_TOOLS

    try:
        # Extract JSON from response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            log.debug("[Executor] No JSON object found in response")
            return None

        json_str = content[json_start:json_end]
        data = json.loads(json_str)

        # Handle new execution_plan format
        if "execution_plan" in data:
            plan_data = data["execution_plan"]
            reasoning = data.get("reasoning", "")
        # Handle legacy tool_calls format (backward compatibility)
        elif "tool_calls" in data:
            tool_calls_data = data["tool_calls"]
            if not tool_calls_data:
                return None
            # Convert to single-step plan
            plan_data = {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Execute all tools",
                        "tool_calls": [
                            {
                                "id": f"tc_{i}",
                                "name": tc.get("name"),
                                "parameters": tc.get("parameters", {}),
                                "depends_on": [],
                            }
                            for i, tc in enumerate(tool_calls_data)
                            if is_valid_tool(tc.get("name", ""))
                        ],
                    }
                ]
            }
            reasoning = ""
        else:
            log.debug("[Executor] No execution_plan or tool_calls found in response")
            return None

        # Build ExecutionPlan from parsed data
        steps = []
        total_tools = 0

        for step_data in plan_data.get("steps", []):
            tool_calls = []
            for tc in step_data.get("tool_calls", []):
                tool_name = tc.get("name")
                if tool_name and is_valid_tool(tool_name):
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", str(uuid4())),
                            name=tool_name,
                            parameters=tc.get("parameters", {}),
                            depends_on=tc.get("depends_on", []),
                        )
                    )

            if tool_calls:
                steps.append(
                    ExecutionStep(
                        step_number=step_data.get("step_number", len(steps) + 1),
                        tool_calls=tool_calls,
                        description=step_data.get("description", ""),
                    )
                )
                total_tools += len(tool_calls)

        if not steps:
            log.debug("[Executor] No valid steps with tools found in plan")
            return None

        plan = ExecutionPlan(
            steps=steps,
            reasoning=reasoning,
            total_tools=total_tools,
        )

        log.info(f"[Executor] Parsed execution plan: {total_tools} tools in {len(steps)} steps")
        return plan

    except json.JSONDecodeError as e:
        log.debug(f"[Executor] JSON decode error parsing execution plan: {e}")
        return None
    except Exception as e:
        log.debug(f"[Executor] Error parsing execution plan: {e}")
        return None
