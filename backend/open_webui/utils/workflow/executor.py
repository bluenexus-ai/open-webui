import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
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


# =============================================================================
# Retry Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for tool call retry behavior."""

    # Maximum number of retry attempts (0 = no retries)
    max_retries: int = 2

    # Base delay between retries in seconds (exponential backoff: delay * 2^attempt)
    base_delay: float = 1.0

    # Maximum delay between retries in seconds
    max_delay: float = 10.0

    # Whether to retry on error (exception/failure)
    retry_on_error: bool = True

    # Whether to retry on empty result (None or empty string/list/dict)
    retry_on_empty: bool = True

    # Specific error messages that should trigger retry
    retryable_errors: Set[str] = field(default_factory=lambda: {
        "timeout",
        "connection",
        "rate limit",
        "temporarily unavailable",
        "server error",
        "503",
        "504",
        "429",
    })

    # Tools that should NOT be retried (e.g., tools with side effects)
    non_retryable_tools: Set[str] = field(default_factory=lambda: {
        "google-workspace_gmail_send_message",
        "slack_slack_send_message",
        "notion_create_page",
        # Add other tools with side effects that shouldn't be retried
    })

    def should_retry(self, tool_name: str, error: Optional[str], result: Any) -> bool:
        """Determine if a tool call should be retried."""
        # Never retry non-retryable tools
        if tool_name in self.non_retryable_tools:
            return False

        # Check if error is retryable
        if error and self.retry_on_error:
            error_lower = error.lower()
            # Check for specific retryable error patterns
            for retryable in self.retryable_errors:
                if retryable in error_lower:
                    return True
            # Retry on generic errors too
            return True

        # Check if result is empty
        if self.retry_on_empty and self._is_empty_result(result):
            return True

        return False

    def _is_empty_result(self, result: Any) -> bool:
        """Check if a result is considered empty."""
        if result is None:
            return True
        if isinstance(result, str) and not result.strip():
            return True
        if isinstance(result, (list, dict)) and len(result) == 0:
            return True
        return False

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt using exponential backoff."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


class ParallelToolExecutor:
    """
    Executes tool calls in parallel within steps while respecting dependencies.
    """

    # Special built-in tools
    LLM_COMPOSE_TOOL = "__llm_compose__"
    LLM_CRITIQUE_TOOL = "__llm_critique__"

    # All built-in tools (for validation)
    BUILTIN_TOOLS = {LLM_COMPOSE_TOOL, LLM_CRITIQUE_TOOL}

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
        retry_config: Optional[RetryConfig] = None,
    ):
        self.tools = tools_dict
        self.event_emitter = event_emitter
        self.event_caller = event_caller
        self.metadata = metadata or {}
        self.process_tool_result = process_tool_result_fn
        self.request = request
        self.user = user
        self.generate_completion_fn = generate_completion_fn
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG

    # Common API response wrapper keys that contain arrays
    ARRAY_WRAPPER_KEYS = ["results", "items", "data", "records", "entries", "list", "values"]

    def _get_nested_value(self, data: Any, path: str) -> Any:
        """
        Extract a nested value from data using dot notation path.
        Returns the value if found, None if path is invalid.

        Examples:
            _get_nested_value({"a": {"b": 1}}, "a.b") -> 1
            _get_nested_value({"items": [{"id": 1}]}, "items.0.id") -> 1
            _get_nested_value({"results": [{"id": 1}]}, "0.id") -> 1  # Auto-unwrap

        Handles common API patterns where arrays are wrapped in keys like
        'results', 'items', 'data', etc.
        """
        if not path:
            return data

        parts = path.split(".")
        current = data

        for i, part in enumerate(parts):
            if current is None:
                return None

            # Handle dict access
            if isinstance(current, dict):
                # Check if part exists directly
                if part in current:
                    current = current.get(part)
                # If part is a numeric index and dict has a common array wrapper key,
                # try to unwrap and access the array
                elif part.isdigit():
                    unwrapped = self._try_unwrap_array(current)
                    if unwrapped is not None:
                        try:
                            idx = int(part)
                            if 0 <= idx < len(unwrapped):
                                current = unwrapped[idx]
                                log.debug(f"[Executor] Auto-unwrapped array, accessed index {idx}")
                            else:
                                return None
                        except (ValueError, TypeError):
                            return None
                    else:
                        return None
                else:
                    return None

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

    def _try_unwrap_array(self, data: Dict[str, Any]) -> Optional[List[Any]]:
        """
        Try to find and return an array from common wrapper keys.

        Many APIs return arrays wrapped in objects like:
        - {"results": [...]}
        - {"items": [...]}
        - {"data": [...]}

        Returns the array if found, None otherwise.
        """
        for key in self.ARRAY_WRAPPER_KEYS:
            if key in data and isinstance(data[key], list):
                return data[key]
        return None

    def _find_tool_result(
        self,
        step_result: Dict[str, Any],
        tool_name: str
    ) -> Optional[Any]:
        """
        Find tool result with flexible name matching.

        Handles common naming variations:
        - Exact match: fireflies_fireflies_get_transcripts
        - Hyphen/underscore variants: fireflies-fireflies-get_transcripts
        - Normalized match (all underscores or all hyphens)
        """
        # Try exact match first
        if tool_name in step_result:
            return step_result[tool_name]

        # Normalize the tool name (convert hyphens to underscores)
        normalized_name = tool_name.replace("-", "_")
        if normalized_name in step_result:
            return step_result[normalized_name]

        # Try hyphenated version
        hyphenated_name = tool_name.replace("_", "-")
        if hyphenated_name in step_result:
            return step_result[hyphenated_name]

        # Try partial matching - find any key that contains the core tool name
        # Extract the core name (last part after underscores/hyphens)
        core_parts = re.split(r'[_\-]', tool_name)
        if len(core_parts) > 1:
            # Try matching by suffix (e.g., "get_transcripts" matches "fireflies_get_transcripts")
            for key in step_result.keys():
                key_parts = re.split(r'[_\-]', key)
                # Check if the last N parts match
                if len(key_parts) >= len(core_parts):
                    if key_parts[-len(core_parts):] == core_parts[-len(core_parts):]:
                        log.debug(f"[Executor] Matched tool '{tool_name}' to '{key}' via partial match")
                        return step_result[key]

        # If only one tool in step results, use it (common case)
        if len(step_result) == 1:
            only_key = list(step_result.keys())[0]
            log.debug(f"[Executor] Using only available tool '{only_key}' for reference '{tool_name}'")
            return step_result[only_key]

        return None

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
        # Groups: (1) step_num, (2) tool_name (allowing hyphens and underscores), (3) optional property path
        # Tool names can contain: letters, digits, underscores, hyphens
        pattern = r"\{\{step_(\d+)_result_([\w\-]+)((?:\.[a-zA-Z0-9_]+)*)\}\}"

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

            # Try to find the tool result with flexible matching
            result = self._find_tool_result(step_result, tool_name)

            if result is None:
                log.warning(f"[Executor] Tool '{tool_name}' not found in step {step_num} results, keeping reference as-is")
                log.debug(f"[Executor] Available tools in step {step_num}: {list(step_result.keys())}")
                return match.group(0)

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

    async def _execute_llm_critique(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Execute the special __llm_critique__ tool that reviews content before actions.

        Parameters:
        - content: The content to review (can be a step reference)
        - criteria: List of criteria to check (e.g., ["professional tone", "no sensitive data"])
        - action_on_issues: What to do if issues found: "revise" | "halt" | "warn"
        - revision_instructions: Optional custom instructions for revision

        Returns:
        - status: "approved" | "revised" | "halted"
        - original_content: The original content
        - revised_content: The revised content (if action_on_issues="revise")
        - issues: List of issues found
        - passed_criteria: List of criteria that passed
        """
        tool_name = tool_call.name

        if not self.generate_completion_fn:
            log.error("[Executor] LLM critique requested but no generate_completion_fn provided")
            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "error",
                "error": "LLM critique not available",
                "result": None,
                "files": [],
                "embeds": [],
            }

        # Resolve parameter references to get actual data
        resolved_params = self._resolve_parameter_references(tool_call.parameters, context)

        content = resolved_params.get("content", "")
        criteria = resolved_params.get("criteria", [])
        action_on_issues = resolved_params.get("action_on_issues", "warn")
        revision_instructions = resolved_params.get("revision_instructions", "")

        # Handle criteria as string (JSON array) or list
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except json.JSONDecodeError:
                criteria = [c.strip() for c in criteria.split(",") if c.strip()]

        # Emit status
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "llm_critique",
                        "description": "Reviewing content for quality...",
                        "done": False,
                    },
                }
            )

        try:
            # Build the LLM prompt for critique
            criteria_list = "\n".join([f"- {c}" for c in criteria]) if criteria else "- Professional tone\n- Clear and concise\n- No sensitive data exposed"

            system_prompt = """You are a quality reviewer that checks content before it gets sent to external services (email, Slack, etc.).
Your job is to ensure the content meets quality standards and doesn't contain issues.

Be strict but fair. Only flag genuine issues that would negatively impact the communication.

CRITICAL AUTOMATIC CHECKS (always apply these):
1. If content includes a "Subject:" line:
   - FAIL if subject contains JSON (starts with { or [, or contains "body": or "content":)
   - FAIL if subject is longer than 100 characters
   - FAIL if subject contains newlines or the full email body
   - WARN if subject is generic like "Meeting Summary" when it should be specific
2. If content looks like JSON being sent as message body:
   - FAIL if the content is raw JSON that should be human-readable text
   - FAIL if the content starts with { and contains keys like "body", "subject", "content"

Respond in JSON format:
{
  "approved": true/false,
  "issues": ["list of issues found, empty if approved"],
  "passed_criteria": ["list of criteria that passed"],
  "severity": "none" | "minor" | "major" | "critical",
  "suggested_revision": "revised content if issues found, null if approved"
}"""

            user_prompt = f"""Review the following content against these criteria:

CRITERIA TO CHECK:
{criteria_list}

CONTENT TO REVIEW:
{content}

Analyze the content and provide your assessment."""

            # Call the LLM for critique
            critique_response = await self.generate_completion_fn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Parse the critique response
            critique_data = self._parse_critique_response(critique_response)

            log.info(f"[Executor] LLM critique completed: approved={critique_data.get('approved')}, severity={critique_data.get('severity')}")

            # Determine final status based on critique and action_on_issues
            approved = critique_data.get("approved", True)
            issues = critique_data.get("issues", [])
            severity = critique_data.get("severity", "none")
            suggested_revision = critique_data.get("suggested_revision")

            result = {
                "original_content": content,
                "approved": approved,
                "issues": issues,
                "passed_criteria": critique_data.get("passed_criteria", []),
                "severity": severity,
            }

            if approved:
                result["status"] = "approved"
                result["final_content"] = content
                status_msg = "Content approved"
            elif action_on_issues == "halt" and severity in ["major", "critical"]:
                result["status"] = "halted"
                result["final_content"] = None
                result["halt_reason"] = f"Content review failed with {severity} issues: {', '.join(issues)}"
                status_msg = f"Content halted: {severity} issues found"
                log.warning(f"[Executor] Critique halted execution: {issues}")
            elif action_on_issues == "revise" and suggested_revision:
                # Use the suggested revision or generate a new one
                if revision_instructions:
                    # Generate custom revision
                    revised_content = await self._generate_revision(
                        content, issues, revision_instructions
                    )
                else:
                    revised_content = suggested_revision

                result["status"] = "revised"
                result["final_content"] = revised_content
                result["revised_content"] = revised_content
                status_msg = "Content revised"
                log.info(f"[Executor] Critique revised content due to: {issues}")
            else:
                # Warn but continue with original
                result["status"] = "warned"
                result["final_content"] = content
                result["warnings"] = issues
                status_msg = f"Content approved with warnings: {len(issues)} issues noted"
                log.warning(f"[Executor] Critique warnings: {issues}")

            if self.event_emitter:
                await self.event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "action": "llm_critique",
                            "description": status_msg,
                            "done": True,
                        },
                    }
                )

            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "success" if result["status"] != "halted" else "halted",
                "result": result,
                "error": result.get("halt_reason") if result["status"] == "halted" else None,
                "files": [],
                "embeds": [],
            }

        except Exception as e:
            log.error(f"[Executor] LLM critique failed: {e}")
            # On critique failure, fail-safe: do NOT auto-approve (security)
            return {
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "status": "error",
                "result": {
                    "status": "critique_error",
                    "original_content": content,
                    "final_content": None,
                    "approved": False,  # Fail-safe: don't approve on error
                    "issues": ["Critique system failed - content not approved for safety"],
                    "error": str(e),
                },
                "error": f"Critique failed: {e}",
                "files": [],
                "embeds": [],
            }

    def _parse_critique_response(self, response: str) -> Dict[str, Any]:
        """Parse the critique LLM response into structured data."""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback: Parse as free text
        response_lower = response.lower()
        approved = "approved" in response_lower and "not approved" not in response_lower
        return {
            "approved": approved,
            "issues": [] if approved else ["Content needs review"],
            "passed_criteria": [],
            "severity": "none" if approved else "minor",
            "suggested_revision": None,
        }

    async def _generate_revision(
        self,
        original_content: str,
        issues: List[str],
        revision_instructions: str,
    ) -> str:
        """Generate a revised version of content based on issues found."""
        system_prompt = """You are a content editor. Revise the given content to address the issues while preserving the original meaning and intent.
Output ONLY the revised content, nothing else."""

        issues_text = "\n".join([f"- {issue}" for issue in issues])
        user_prompt = f"""ORIGINAL CONTENT:
{original_content}

ISSUES TO ADDRESS:
{issues_text}

REVISION INSTRUCTIONS:
{revision_instructions}

Please provide the revised content:"""

        revised = await self.generate_completion_fn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return revised.strip()

    # Email-sending tools that need parameter validation
    EMAIL_TOOLS = [
        "google-workspace_gmail_send_message",
        "gmail_send_message",
        "outlook_send_email",
        "email_send",
    ]

    def _validate_and_fix_email_params(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Validate and fix email parameters before sending.

        Checks for common issues:
        - Subject is JSON or contains body content
        - Subject is too long (> 200 chars)
        - Subject contains newlines
        - Body is empty or looks malformed

        Returns:
            (fixed_params, list of issues found)
        """
        issues = []
        fixed_params = params.copy()

        # Check subject line
        subject = params.get("subject", "")
        if subject:
            # Check if subject looks like JSON
            if subject.strip().startswith("{") or subject.strip().startswith("["):
                issues.append(f"Subject appears to be JSON data (starts with '{subject[:20]}...')")
                # Try to extract a proper subject from JSON
                try:
                    parsed = json.loads(subject)
                    if isinstance(parsed, dict):
                        # Try common keys for subject/title
                        for key in ["subject", "title", "name", "summary"]:
                            if key in parsed and isinstance(parsed[key], str):
                                fixed_params["subject"] = parsed[key][:150]
                                log.warning(f"[Executor] Fixed malformed subject from JSON key '{key}'")
                                break
                        else:
                            # Generate a default subject
                            fixed_params["subject"] = "Email from Workflow"
                            log.warning("[Executor] Could not extract subject from JSON, using default")
                except json.JSONDecodeError:
                    # Not valid JSON but starts with { - truncate to first line
                    first_line = subject.split("\n")[0][:100]
                    if len(first_line) < len(subject):
                        fixed_params["subject"] = first_line
                        log.warning(f"[Executor] Truncated malformed subject to: {first_line}")

            # Check if subject is too long (likely contains body)
            elif len(subject) > 200:
                issues.append(f"Subject too long ({len(subject)} chars, max 200)")
                # Truncate to first sentence or 100 chars
                truncated = subject[:100]
                if ". " in truncated:
                    truncated = truncated[:truncated.index(". ") + 1]
                fixed_params["subject"] = truncated
                log.warning(f"[Executor] Truncated long subject to: {truncated}")

            # Check for newlines in subject
            elif "\n" in subject:
                issues.append("Subject contains newlines")
                fixed_params["subject"] = subject.split("\n")[0].strip()
                log.warning("[Executor] Removed newlines from subject")

        # Check body
        body = params.get("body", "")
        if not body or (isinstance(body, str) and len(body.strip()) < 10):
            issues.append("Email body is empty or too short")

        # Check if body looks like JSON metadata instead of content
        if body and isinstance(body, str) and body.strip().startswith('{"'):
            try:
                parsed_body = json.loads(body)
                if isinstance(parsed_body, dict):
                    # Try to extract actual body content
                    for key in ["body", "content", "text", "message", "email_body"]:
                        if key in parsed_body and isinstance(parsed_body[key], str):
                            fixed_params["body"] = parsed_body[key]
                            issues.append(f"Body was JSON wrapper, extracted from '{key}' key")
                            log.warning(f"[Executor] Extracted body content from JSON key '{key}'")
                            break
            except json.JSONDecodeError:
                pass

        if issues:
            log.warning(f"[Executor] Email validation issues: {issues}")

        return fixed_params, issues

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute a single tool call and return the result"""
        tool_name = tool_call.name

        # Handle special built-in tools
        if tool_name == self.LLM_COMPOSE_TOOL:
            return await self._execute_llm_compose(tool_call, context)

        if tool_name == self.LLM_CRITIQUE_TOOL:
            return await self._execute_llm_critique(tool_call, context)

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

        # Validate and fix email parameters before sending
        if any(email_tool in tool_name for email_tool in self.EMAIL_TOOLS):
            filtered_params, validation_issues = self._validate_and_fix_email_params(
                tool_name, filtered_params
            )
            if validation_issues:
                log.warning(f"[Executor] Email params fixed: {validation_issues}")

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

    async def _execute_single_tool_with_retry(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Execute a single tool call with retry logic.

        Retries are attempted when:
        - Tool execution fails with a retryable error
        - Tool returns an empty result (if retry_on_empty is enabled)

        Uses exponential backoff between retry attempts.
        """
        tool_name = tool_call.name
        max_retries = self.retry_config.max_retries
        attempt = 0
        last_result = None

        while attempt <= max_retries:
            # Execute the tool
            result = await self._execute_single_tool(tool_call, context)
            last_result = result

            # Check if successful with valid result
            status = result.get("status")
            error = result.get("error")
            tool_result = result.get("result")

            # Success with non-empty result - return immediately
            if status == "success" and not self.retry_config._is_empty_result(tool_result):
                if attempt > 0:
                    log.info(f"[Executor] Tool '{tool_name}' succeeded on retry attempt {attempt}")
                return result

            # Check if we should retry
            if attempt < max_retries:
                should_retry = self.retry_config.should_retry(tool_name, error, tool_result)

                if should_retry:
                    delay = self.retry_config.get_delay(attempt)
                    retry_reason = "error" if error else "empty result"

                    log.warning(
                        f"[Executor] Tool '{tool_name}' {retry_reason}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )

                    # Emit retry status to client
                    if self.event_emitter:
                        await self.event_emitter(
                            {
                                "type": "status",
                                "data": {
                                    "action": "tool_retry",
                                    "description": f"Retrying {tool_name} ({attempt + 1}/{max_retries})...",
                                    "done": False,
                                },
                            }
                        )

                    # Wait before retry
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                else:
                    # Not retryable, return result as-is
                    if error:
                        log.info(f"[Executor] Tool '{tool_name}' error is not retryable: {error}")
                    break
            else:
                # Max retries reached
                if attempt > 0:
                    log.warning(
                        f"[Executor] Tool '{tool_name}' failed after {max_retries} retries"
                    )

                    # Update error message to indicate retry exhaustion
                    if last_result.get("status") == "error":
                        last_result["error"] = f"{last_result.get('error', 'Unknown error')} (after {max_retries} retries)"

                    # Emit final status
                    if self.event_emitter:
                        await self.event_emitter(
                            {
                                "type": "status",
                                "data": {
                                    "action": "tool_retry",
                                    "description": f"Tool {tool_name} failed after {max_retries} retries",
                                    "done": True,
                                },
                            }
                        )
                break

        return last_result

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

        # Create tasks for all tool calls in this step (parallel execution with retry)
        tasks = [self._execute_single_tool_with_retry(tc, context) for tc in step.tool_calls]

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
    def is_valid_tool(tool_name: str) -> bool:
        return tool_name in tools_dict or tool_name in ParallelToolExecutor.BUILTIN_TOOLS

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
