"""
Workflow Router - Intelligently routes tasks to the appropriate handler.

Routes:
1. Simple tasks → Regular LLM conversation
2. Complex tasks with predefined patterns → Template workflows
3. Complex tasks without patterns → Auto-planned workflows
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MAIN", logging.INFO))


class TaskComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class RouteType(Enum):
    SIMPLE_LLM = "simple_llm"
    PREDEFINED_WORKFLOW = "predefined_workflow"
    AUTO_PLANNED_WORKFLOW = "auto_planned_workflow"


@dataclass
class WorkflowStep:
    """A single step in a predefined workflow."""
    tool_name: str
    description: str
    parameters_template: Dict[str, Any]
    use_llm_compose: bool = False
    compose_prompt: str = ""
    compose_output_key: str = "content"


@dataclass
class PredefinedWorkflow:
    """A predefined workflow template for common task patterns."""
    id: str
    name: str
    description: str
    trigger_patterns: List[str]  # Regex patterns to match user queries
    required_tools: List[str]  # Tools that must be available
    steps: List[WorkflowStep]

    def matches_query(self, query: str) -> bool:
        """Check if this workflow matches the user query."""
        query_lower = query.lower()
        for pattern in self.trigger_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def has_required_tools(self, available_tools: List[str]) -> bool:
        """Check if all required tools are available."""
        return all(tool in available_tools for tool in self.required_tools)


@dataclass
class RouteDecision:
    """The routing decision made by the router."""
    route_type: RouteType
    complexity: TaskComplexity
    workflow_id: Optional[str] = None
    workflow: Optional[PredefinedWorkflow] = None
    confidence: float = 0.0
    reasoning: str = ""


# Registry of predefined workflows
PREDEFINED_WORKFLOWS: Dict[str, PredefinedWorkflow] = {}


def register_workflow(workflow: PredefinedWorkflow) -> None:
    """Register a predefined workflow."""
    PREDEFINED_WORKFLOWS[workflow.id] = workflow
    log.info(f"[Router] Registered workflow: {workflow.id} - {workflow.name}")


def get_workflow(workflow_id: str) -> Optional[PredefinedWorkflow]:
    """Get a predefined workflow by ID."""
    return PREDEFINED_WORKFLOWS.get(workflow_id)


def list_workflows() -> List[PredefinedWorkflow]:
    """List all registered workflows."""
    return list(PREDEFINED_WORKFLOWS.values())


# ============================================================================
# Predefined Workflow Templates
# ============================================================================

# Fireflies Meeting Summary to Email
FIREFLIES_TO_EMAIL = PredefinedWorkflow(
    id="fireflies_to_email",
    name="Fireflies Meeting Summary to Email",
    description="Get Fireflies meeting transcript and send a formatted summary email",
    trigger_patterns=[
        r"fireflies.*meeting.*email",
        r"meeting.*summary.*email",
        r"send.*meeting.*summary",
        r"email.*meeting.*transcript",
        r"fireflies.*send.*email",
    ],
    required_tools=["fireflies_fireflies_get_transcripts", "google-workspace_gmail_send_message"],
    steps=[
        WorkflowStep(
            tool_name="fireflies_fireflies_get_transcripts",
            description="Fetch recent meeting transcripts",
            parameters_template={"limit": 5, "mine": True},
        ),
        WorkflowStep(
            tool_name="__llm_compose__",
            description="Generate beautiful email content",
            parameters_template={
                "data": "{{step_1_result_fireflies_fireflies_get_transcripts}}",
                "prompt": "Write a professional, well-formatted email summarizing this meeting. Include: 1) Meeting title and date, 2) Key discussion points, 3) Decisions made, 4) Action items with assignees. Use proper email formatting with greeting and signature.",
                "output_key": "email_body",
            },
            use_llm_compose=True,
        ),
        WorkflowStep(
            tool_name="__llm_compose__",
            description="Generate email subject line",
            parameters_template={
                "data": "{{step_1_result_fireflies_fireflies_get_transcripts}}",
                "prompt": "Generate a concise, professional email subject line for this meeting summary. Just output the subject line, nothing else.",
                "output_key": "subject",
            },
            use_llm_compose=True,
        ),
        WorkflowStep(
            tool_name="google-workspace_gmail_send_message",
            description="Send the composed email",
            parameters_template={
                "to": "{{USER_EMAIL}}",
                "subject": "{{step_3_result___llm_compose__.subject}}",
                "body": "{{step_2_result___llm_compose__.email_body}}",
            },
        ),
    ],
)

# Fireflies Meeting Summary to Slack
FIREFLIES_TO_SLACK = PredefinedWorkflow(
    id="fireflies_to_slack",
    name="Fireflies Meeting Summary to Slack",
    description="Get Fireflies meeting transcript and post a formatted summary to Slack",
    trigger_patterns=[
        r"fireflies.*meeting.*slack",
        r"meeting.*summary.*slack",
        r"post.*meeting.*slack",
        r"slack.*meeting.*transcript",
        r"send.*meeting.*slack",
    ],
    required_tools=["fireflies_fireflies_get_transcripts", "slack_slack_send_message"],
    steps=[
        WorkflowStep(
            tool_name="fireflies_fireflies_get_transcripts",
            description="Fetch recent meeting transcripts",
            parameters_template={"limit": 5, "mine": True},
        ),
        WorkflowStep(
            tool_name="__llm_compose__",
            description="Generate Slack message",
            parameters_template={
                "data": "{{step_1_result_fireflies_fireflies_get_transcripts}}",
                "prompt": "Write a concise Slack message summarizing this meeting. Use Slack markdown formatting (*bold*, _italic_, bullet points). Include: meeting title, key points, and action items. Keep it scannable and professional.",
                "output_key": "message",
            },
            use_llm_compose=True,
        ),
        WorkflowStep(
            tool_name="slack_slack_send_message",
            description="Post to Slack",
            parameters_template={
                "channel": "{{SLACK_CHANNEL}}",
                "text": "{{step_2_result___llm_compose__.message}}",
            },
        ),
    ],
)

# Calendar Events to Email Reminder
CALENDAR_TO_EMAIL = PredefinedWorkflow(
    id="calendar_to_email",
    name="Calendar Events Email Reminder",
    description="Get upcoming calendar events and send a reminder email",
    trigger_patterns=[
        r"calendar.*email",
        r"upcoming.*events.*email",
        r"send.*schedule",
        r"email.*calendar",
        r"remind.*meetings",
    ],
    required_tools=["google-workspace_google-calendar_list_events", "google-workspace_gmail_send_message"],
    steps=[
        WorkflowStep(
            tool_name="google-workspace_google-calendar_list_events",
            description="Fetch upcoming calendar events",
            parameters_template={"max_results": 10},
        ),
        WorkflowStep(
            tool_name="__llm_compose__",
            description="Generate reminder email",
            parameters_template={
                "data": "{{step_1_result_google-workspace_google-calendar_list_events}}",
                "prompt": "Write a helpful email reminder about upcoming calendar events. Format each event clearly with date, time, title, and location. Group by day if multiple days. Use professional tone.",
                "output_key": "email_body",
            },
            use_llm_compose=True,
        ),
        WorkflowStep(
            tool_name="google-workspace_gmail_send_message",
            description="Send reminder email",
            parameters_template={
                "to": "{{USER_EMAIL}}",
                "subject": "Your Upcoming Schedule",
                "body": "{{step_2_result___llm_compose__.email_body}}",
            },
        ),
    ],
)

# Register default workflows
register_workflow(FIREFLIES_TO_EMAIL)
register_workflow(FIREFLIES_TO_SLACK)
register_workflow(CALENDAR_TO_EMAIL)


# ============================================================================
# Router Logic
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are a task complexity classifier. Analyze the user's query and determine:
1. Whether it's a SIMPLE task (can be answered directly) or COMPLEX task (requires tools/workflows)
2. If COMPLEX, identify which predefined workflow pattern it matches (if any)

SIMPLE tasks:
- General questions, explanations, advice
- Creative writing, brainstorming
- Code explanations or reviews
- Casual conversation

COMPLEX tasks (require tools):
- Fetching data from external services (Fireflies, Calendar, etc.)
- Sending emails or messages
- Multi-step workflows
- Anything requiring real-time data or actions

Available predefined workflows:
{workflows}

Respond in JSON format:
{{
  "complexity": "simple" or "complex",
  "workflow_match": "workflow_id" or null,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""


async def classify_task(
    query: str,
    available_tools: List[str],
    generate_completion_fn: Callable,
) -> RouteDecision:
    """
    Use LLM to classify the task complexity and match to workflows.
    """
    # Build workflow descriptions for the prompt
    workflow_descriptions = []
    for wf in PREDEFINED_WORKFLOWS.values():
        if wf.has_required_tools(available_tools):
            workflow_descriptions.append(f"- {wf.id}: {wf.description}")

    workflows_text = "\n".join(workflow_descriptions) if workflow_descriptions else "No predefined workflows available"

    system_prompt = ROUTER_SYSTEM_PROMPT.format(workflows=workflows_text)
    user_prompt = f"User query: {query}"

    try:
        response = await generate_completion_fn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Parse JSON response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())

            complexity = TaskComplexity(data.get("complexity", "simple"))
            workflow_id = data.get("workflow_match")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            # Determine route type
            if complexity == TaskComplexity.SIMPLE:
                route_type = RouteType.SIMPLE_LLM
                workflow = None
            elif workflow_id and workflow_id in PREDEFINED_WORKFLOWS:
                workflow = PREDEFINED_WORKFLOWS[workflow_id]
                if workflow.has_required_tools(available_tools):
                    route_type = RouteType.PREDEFINED_WORKFLOW
                else:
                    route_type = RouteType.AUTO_PLANNED_WORKFLOW
                    workflow = None
            else:
                route_type = RouteType.AUTO_PLANNED_WORKFLOW
                workflow = None

            return RouteDecision(
                route_type=route_type,
                complexity=complexity,
                workflow_id=workflow_id,
                workflow=workflow,
                confidence=confidence,
                reasoning=reasoning,
            )

    except Exception as e:
        log.warning(f"[Router] Classification failed: {e}, falling back to pattern matching")

    # Fallback: Use pattern matching
    return classify_task_by_pattern(query, available_tools)


def classify_task_by_pattern(
    query: str,
    available_tools: List[str],
) -> RouteDecision:
    """
    Fallback classification using pattern matching (no LLM call).
    """
    # Check for tool-related keywords
    tool_keywords = [
        "send", "email", "post", "slack", "meeting", "calendar",
        "fireflies", "transcript", "schedule", "remind", "fetch",
        "get", "create", "update", "delete", "search", "find"
    ]

    query_lower = query.lower()
    has_tool_keywords = any(kw in query_lower for kw in tool_keywords)

    if not has_tool_keywords:
        return RouteDecision(
            route_type=RouteType.SIMPLE_LLM,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.7,
            reasoning="No tool-related keywords found",
        )

    # Check for predefined workflow matches
    for workflow in PREDEFINED_WORKFLOWS.values():
        if workflow.matches_query(query) and workflow.has_required_tools(available_tools):
            return RouteDecision(
                route_type=RouteType.PREDEFINED_WORKFLOW,
                complexity=TaskComplexity.COMPLEX,
                workflow_id=workflow.id,
                workflow=workflow,
                confidence=0.8,
                reasoning=f"Matched predefined workflow: {workflow.name}",
            )

    # Complex task without predefined workflow
    return RouteDecision(
        route_type=RouteType.AUTO_PLANNED_WORKFLOW,
        complexity=TaskComplexity.COMPLEX,
        confidence=0.6,
        reasoning="Tool keywords found but no predefined workflow match",
    )


def build_execution_plan_from_workflow(
    workflow: PredefinedWorkflow,
    user_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an execution plan from a predefined workflow template.
    Substitutes user context variables into the template.
    """
    steps = []

    for i, step in enumerate(workflow.steps, start=1):
        # Deep copy and substitute variables in parameters
        parameters = _substitute_variables(step.parameters_template, user_context)

        tool_call = {
            "id": f"{step.tool_name}_{i}",
            "name": step.tool_name,
            "parameters": parameters,
            "depends_on": [f"{workflow.steps[i-2].tool_name}_{i-1}"] if i > 1 else [],
        }

        steps.append({
            "step_number": i,
            "description": step.description,
            "tool_calls": [tool_call],
        })

    return {
        "reasoning": f"Using predefined workflow: {workflow.name}",
        "execution_plan": {
            "steps": steps,
        },
    }


def _substitute_variables(template: Any, context: Dict[str, Any]) -> Any:
    """Recursively substitute {{VARIABLE}} patterns in template."""
    if isinstance(template, str):
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result
    elif isinstance(template, dict):
        return {k: _substitute_variables(v, context) for k, v in template.items()}
    elif isinstance(template, list):
        return [_substitute_variables(item, context) for item in template]
    else:
        return template
