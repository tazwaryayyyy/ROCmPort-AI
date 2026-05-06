# pylint: disable=broad-exception-caught
import asyncio
import json
from typing import Literal

from langgraph.graph import StateGraph, END

from backend.graph.state import MigrationState

# ─── Node implementations ──────────────────────────────────────────────────────


async def analyze_node(state: MigrationState) -> dict:
    from backend.agents import analyzer as analyzer_agent

    new_events = [
        {
            "agent": "analyzer",
            "status": "running",
            "message": "Scanning CUDA code for kernels, APIs, and hardware-specific issues...",
            "detail": None,
            "iteration": state.get("iteration", 0),
        }
    ]

    try:
        result = await asyncio.to_thread(analyzer_agent.run, state["cuda_code"])
    except Exception as exc:
        new_events.append(
            {
                "agent": "analyzer",
                "status": "failed",
                "message": "Analysis failed",
                "detail": str(exc),
                "iteration": state.get("iteration", 0),
            }
        )
        return {"analyzer_result": None, "events": new_events, "migration_success": False}

    detail_parts = [
        f"Found {len(result.kernels_found)} kernel(s): {', '.join(result.kernels_found)}",
        f"Workload: {result.workload_type.value}",
        f"Difficulty: {result.difficulty} - {result.difficulty_reason}",
    ]
    if result.warp_size_issue:
        detail_parts.append(f"WARP SIZE ISSUE: {result.warp_size_detail}")
    if result.sharding_detected:
        detail_parts.append(
            "Multi-GPU sharding detected; review if needed on MI300X memory capacity."
        )
    if result.prediction:
        detail_parts.append(result.prediction)

    new_events.append(
        {
            "agent": "analyzer",
            "status": "done",
            "message": (
                f"Found {len(result.kernels_found)} kernel(s) | "
                f"{result.workload_type.value} workload | Difficulty: {result.difficulty}"
            ),
            "detail": "\n".join(detail_parts),
            "iteration": state.get("iteration", 0),
        }
    )

    return {"analyzer_result": result, "events": new_events}


async def translate_node(state: MigrationState) -> dict:
    from backend.agents import translator as translator_agent

    new_events = [
        {
            "agent": "translator",
            "status": "running",
            "message": "Running hipify-clang (pass 1) then LLM correction (pass 2)...",
            "detail": None,
            "iteration": state.get("iteration", 0),
        }
    ]

    analyzer_result = state.get("analyzer_result")
    if analyzer_result is None:
        new_events.append(
            {
                "agent": "translator",
                "status": "failed",
                "message": "Translation skipped — analysis did not complete",
                "detail": None,
                "iteration": state.get("iteration", 0),
            }
        )
        return {"translator_result": None, "events": new_events}

    try:
        result = await asyncio.to_thread(
            translator_agent.run, state["cuda_code"], analyzer_result
        )
    except Exception as exc:
        new_events.append(
            {
                "agent": "translator",
                "status": "failed",
                "message": "Translation failed",
                "detail": str(exc),
                "iteration": state.get("iteration", 0),
            }
        )
        return {"translator_result": None, "events": new_events}

    new_events.append(
        {
            "agent": "translator",
            "status": "done",
            "message": (
                f"{result.total_changes} changes "
                f"({result.hipify_changes} hipify + {result.llm_changes} LLM)"
            ),
            "detail": (
                f"Total changes: {result.total_changes} "
                f"({result.hipify_changes} hipify, {result.llm_changes} LLM)\n"
                f"Warp size corrected: {analyzer_result.warp_size_issue}\n"
                "Kernel launch syntax updated"
            ),
            "iteration": state.get("iteration", 0),
        }
    )

    return {"translator_result": result, "events": new_events}


async def optimize_node(state: MigrationState) -> dict:
    from backend.agents import optimizer as optimizer_agent

    # bump on each optimizer invocation
    iteration = state.get("iteration", 0) + 1
    analyzer_result = state.get("analyzer_result")
    translator_result = state.get("translator_result")
    prev_tester_result = state.get("tester_result")  # set on retry path
    is_retry = prev_tester_result is not None

    new_events: list[dict] = []

    # On retry: emit coordinator decision + optimizer retrying signals
    if is_retry:
        new_events.append(
            {
                "agent": "coordinator",
                "status": "running",
                "message": "Performance regressed, retrying optimizer with profiler feedback...",
                "detail": f"Profiler feedback: {prev_tester_result.notes}",
                "iteration": iteration,
            }
        )
        new_events.append(
            {
                "agent": "optimizer",
                "status": "retrying",
                "message": f"Trying alternative optimization strategy (iteration {iteration})...",
                "detail": f"Previous strategy regressed. Feedback: {prev_tester_result.notes}",
                "iteration": iteration,
            }
        )
    else:
        new_events.append(
            {
                "agent": "optimizer",
                "status": "running",
                "message": f"Applying AMD MI300X-specific optimizations (iteration {iteration})...",
                "detail": None,
                "iteration": iteration,
            }
        )

    if translator_result is None:
        new_events.append(
            {
                "agent": "optimizer",
                "status": "failed",
                "message": "Optimization skipped — translation did not complete",
                "detail": None,
                "iteration": iteration,
            }
        )
        return {"optimizer_result": None, "iteration": iteration, "events": new_events}

    previous_feedback = prev_tester_result.notes if is_retry else None

    try:
        result = await asyncio.to_thread(
            optimizer_agent.run,
            translator_result.hip_code,  # always start from translated base
            analyzer_result,
            iteration,
            previous_feedback,
        )
    except Exception as exc:
        new_events.append(
            {
                "agent": "optimizer",
                "status": "failed",
                "message": "Optimization failed" if not is_retry else "Re-optimization failed",
                "detail": str(exc),
                "iteration": iteration,
            }
        )
        return {"optimizer_result": None, "iteration": iteration, "events": new_events}

    new_events.append(
        {
            "agent": "optimizer",
            "status": "done",
            "message": f"{len(result.changes)} optimization(s) applied",
            "detail": "\n".join(f"- {c['description']}" for c in result.changes),
            "iteration": iteration,
        }
    )

    return {"optimizer_result": result, "iteration": iteration, "events": new_events}


async def test_node(state: MigrationState) -> dict:
    from backend.agents import tester as tester_agent

    iteration = state.get("iteration", 0)
    analyzer_result = state.get("analyzer_result")
    optimizer_result = state.get("optimizer_result")

    new_events = [
        {
            "agent": "tester",
            "status": "running",
            "message": f"Compiling with hipcc and profiling with rocprof (iteration {iteration})...",
            "detail": None,
            "iteration": iteration,
        }
    ]

    if optimizer_result is None:
        new_events.append(
            {
                "agent": "tester",
                "status": "failed",
                "message": "Testing skipped — optimization did not complete",
                "detail": None,
                "iteration": iteration,
            }
        )
        return {
            "tester_result": None,
            "migration_success": False,
            "events": new_events,
        }

    try:
        result = await asyncio.to_thread(
            tester_agent.run,
            optimizer_result.optimized_code,
            analyzer_result,
            iteration,
            state.get("kernel_name", "custom"),
        )
    except Exception as exc:
        new_events.append(
            {
                "agent": "tester",
                "status": "failed",
                "message": "Testing failed",
                "detail": str(exc),
                "iteration": iteration,
            }
        )
        return {
            "tester_result": None,
            "migration_success": False,
            "events": new_events,
        }

    if not result.success:
        new_events.append(
            {
                "agent": "tester",
                "status": "failed",
                "message": "Compilation or profiling failed",
                "detail": result.notes,
                "iteration": iteration,
            }
        )
        return {
            "tester_result": result,
            "migration_success": False,
            "events": new_events,
        }

    if result.speedup < 0.95:
        new_events.append(
            {
                "agent": "tester",
                "status": "failed",
                "message": f"Iteration {iteration}: {result.speedup}x vs baseline HIP (regression)",
                "detail": (
                    f"Bandwidth utilized: {result.bandwidth_utilized}%\n"
                    f"{result.notes}"
                ),
                "iteration": iteration,
            }
        )
    else:
        new_events.append(
            {
                "agent": "tester",
                "status": "done",
                "message": f"Iteration {iteration}: {result.speedup}x vs baseline HIP",
                "detail": (
                    f"Execution time: {result.execution_ms:.1f}ms\n"
                    f"Memory bandwidth: {result.bandwidth_utilized:.1f}% utilized\n"
                    f"Bottleneck type: {result.bottleneck}\n"
                    f"{result.notes}"
                ),
                "iteration": iteration,
            }
        )

    return {"tester_result": result, "events": new_events}


async def coordinate_node(state: MigrationState) -> dict:
    from backend.agents.coordinator import (
        calculate_cost_estimate,
        simplify_explanation,
        _build_amd_explanation,
    )
    from backend.models import FinalReport, CostEstimate

    analyzer_result = state.get("analyzer_result")
    translator_result = state.get("translator_result")
    optimizer_result = state.get("optimizer_result")
    tester_result = state.get("tester_result")
    iteration = state.get("iteration", 0)
    simple_mode = state.get("simple_mode", False)

    new_events = [
        {
            "agent": "coordinator",
            "status": "running",
            "message": "Generating migration report...",
            "detail": None,
            "iteration": iteration,
        }
    ]

    # Hard failure path — one or more agents did not complete
    if tester_result is None or translator_result is None or optimizer_result is None:
        new_events.append(
            {
                "agent": "coordinator",
                "status": "failed",
                "message": "Pipeline did not complete successfully",
                "detail": "One or more agents failed before the report could be generated.",
                "iteration": iteration,
            }
        )
        return {
            "migration_success": False,
            "final_report": {},
            "events": new_events,
        }

    amd_explanation = _build_amd_explanation(analyzer_result, tester_result)

    try:
        cost_estimate = calculate_cost_estimate(analyzer_result)
    except Exception:
        from backend.models import CostEstimate
        cost_estimate = CostEstimate(
            manual_porting_weeks="3-6 weeks",
            rocmport_minutes="Varies by kernel",
            estimated_savings="$20,000-$50,000",
            complexity_factor="Medium",
        )

    total_changes = translator_result.total_changes + \
        len(optimizer_result.changes)

    temp_report = FinalReport(
        migration_success=tester_result.success,
        speedup=tester_result.speedup,
        bandwidth_utilized=tester_result.bandwidth_utilized,
        total_changes=total_changes,
        bottleneck=tester_result.bottleneck,
        amd_advantage_explanation=amd_explanation,
        iterations=iteration,
        hip_code=translator_result.hip_code,
        optimized_code=optimizer_result.optimized_code,
        verification=tester_result.verification,
        static_risk_report=analyzer_result.static_risk_report if analyzer_result else None,
        data_source=tester_result.data_source or "simulated",
    )

    simplified = simplify_explanation(
        temp_report) if simple_mode else amd_explanation

    report = FinalReport(
        migration_success=tester_result.success,
        speedup=tester_result.speedup,
        bandwidth_utilized=tester_result.bandwidth_utilized,
        total_changes=total_changes,
        bottleneck=tester_result.bottleneck,
        amd_advantage_explanation=amd_explanation,
        iterations=iteration,
        hip_code=translator_result.hip_code,
        optimized_code=optimizer_result.optimized_code,
        verification=tester_result.verification,
        cost_estimate=cost_estimate,
        simplified_explanation=simplified,
        static_risk_report=analyzer_result.static_risk_report if analyzer_result else None,
        data_source=tester_result.data_source or "simulated",
    )

    report_dict = report.model_dump()

    new_events.append(
        {
            "agent": "coordinator",
            "status": "done",
            "message": "Migration complete",
            "detail": json.dumps(report_dict),
            "iteration": iteration,
        }
    )

    return {
        "migration_success": report.migration_success,
        "final_report": report_dict,
        "events": new_events,
    }


# ─── Conditional routing ───────────────────────────────────────────────────────

def should_retry_decision(state: MigrationState) -> Literal["retry", "done"]:
    """Route to optimizer (retry) or coordinator (done)."""
    tester_result = state.get("tester_result")
    if tester_result is None:
        return "done"
    if not getattr(tester_result, "success", True):
        return "done"  # hard compile/run failure — let coordinator report it
    speedup = float(getattr(tester_result, "speedup", 1.0) or 1.0)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if speedup < 0.95 and iteration < max_iter:
        return "retry"
    return "done"


# ─── Graph builder ─────────────────────────────────────────────────────────────

def build_pipeline():
    """Build and compile the LangGraph StateGraph for the migration pipeline."""
    graph = StateGraph(MigrationState)

    graph.add_node("analyzer",    analyze_node)
    graph.add_node("translator",  translate_node)
    graph.add_node("optimizer",   optimize_node)
    graph.add_node("tester",      test_node)
    graph.add_node("coordinator", coordinate_node)

    graph.set_entry_point("analyzer")
    graph.add_edge("analyzer",   "translator")
    graph.add_edge("translator", "optimizer")
    graph.add_edge("optimizer",  "tester")

    graph.add_conditional_edges(
        "tester",
        should_retry_decision,
        {
            "retry": "optimizer",    # performance regression + iterations remaining
            "done":  "coordinator",  # acceptable result or hard failure
        },
    )

    graph.add_edge("coordinator", END)

    return graph.compile()


# Module-level compiled pipeline (reused across requests)
pipeline = build_pipeline()
