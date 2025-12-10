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

    def __init__(
        self,
        tools_dict: Dict[str, Any],
        event_emitter: Optional[Callable] = None,
        event_caller: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        process_tool_result_fn: Optional[Callable] = None,
        request: Any = None,
        user: Any = None,
    ):
        self.tools = tools_dict
        self.event_emitter = event_emitter
        self.event_caller = event_caller
        self.metadata = metadata or {}
        self.process_tool_result = process_tool_result_fn
        self.request = request
        self.user = user

    def _resolve_parameter_references(
        self,
        parameters: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Resolve {{step_N_result_toolName}} references in parameters.
        """
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                # Find all {{step_N_result_toolName}} patterns
                pattern = r"\{\{step_(\d+)_result_(\w+)\}\}"
                matches = re.findall(pattern, value)

                resolved_value = value
                for step_num_str, tool_name in matches:
                    step_num = int(step_num_str)
                    if step_num in context.step_results:
                        step_result = context.step_results[step_num]
                        if tool_name in step_result:
                            replacement = step_result[tool_name]
                            if isinstance(replacement, dict):
                                replacement = json.dumps(replacement)
                            elif not isinstance(replacement, str):
                                replacement = str(replacement)
                            resolved_value = resolved_value.replace(
                                f"{{{{step_{step_num}_result_{tool_name}}}}}",
                                replacement,
                            )
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

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute a single tool call and return the result"""
        tool_name = tool_call.name

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
                            if tc.get("name") in tools_dict
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
                if tool_name and tool_name in tools_dict:
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
