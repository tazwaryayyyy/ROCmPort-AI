from typing import TypedDict, Optional, List, Any, Annotated
import operator


class MigrationState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    cuda_code: str
    kernel_name: str
    simple_mode: bool

    # ── Intermediate results (pydantic model instances stored as Any) ──────────
    analyzer_result: Optional[Any]    # AnalyzerResult
    translator_result: Optional[Any]  # TranslatorResult
    optimizer_result: Optional[Any]   # OptimizerResult
    tester_result: Optional[Any]      # TesterResult

    # ── Control flow ───────────────────────────────────────────────────────────
    iteration: int         # incremented each time optimizer node runs
    max_iterations: int    # default 3
    should_retry: bool

    # ── Output ─────────────────────────────────────────────────────────────────
    migration_success: bool
    final_report: dict

    # ── SSE event accumulator — LangGraph appends each node's new events ───────
    # operator.add reducer: each node returns a *list of new events*;
    # LangGraph concatenates them into a single growing list.
    events: Annotated[List[dict], operator.add]
