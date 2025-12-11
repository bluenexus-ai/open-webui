from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class ExecutionStatus(Enum):
    """Status of a tool call or execution step"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ToolCall:
    """Represents a single tool call within an execution step"""

    id: str
    name: str
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ExecutionStep:
    """Represents a step in the execution plan containing parallel-executable tools"""

    step_number: int
    tool_calls: List[ToolCall]
    description: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING

    @property
    def tool_ids(self) -> Set[str]:
        return {tc.id for tc in self.tool_calls}

    @property
    def tool_names(self) -> Set[str]:
        return {tc.name for tc in self.tool_calls}


@dataclass
class ExecutionPlan:
    """Complete execution plan with multiple steps"""

    steps: List[ExecutionStep]
    reasoning: str = ""
    total_tools: int = 0

    def get_step_by_number(self, step_number: int) -> Optional[ExecutionStep]:
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_next_pending_step(self) -> Optional[ExecutionStep]:
        for step in self.steps:
            if step.status == ExecutionStatus.PENDING:
                return step
        return None

    def is_complete(self) -> bool:
        return all(
            step.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED)
            for step in self.steps
        )


@dataclass
class ExecutionContext:
    """Context passed between execution steps"""

    sources: List[Dict[str, Any]] = field(default_factory=list)
    skip_files: bool = False
    step_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def get_result_for_tool(self, step_number: int, tool_name: str) -> Optional[Any]:
        """Get the result of a specific tool from a previous step"""
        if step_number in self.step_results:
            return self.step_results[step_number].get(tool_name)
        return None

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def get_all_results_flat(self) -> Dict[str, Any]:
        """Get all results as a flat dictionary keyed by tool name"""
        results = {}
        for step_results in self.step_results.values():
            results.update(step_results)
        return results
