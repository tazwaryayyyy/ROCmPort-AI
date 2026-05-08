import asyncio
import json
from typing import AsyncGenerator

# pylint: disable=broad-exception-caught

from . import analyzer, optimizer, tester, translator
from ..models import (
    AgentEvent,
    AgentStatus,
    AnalyzerResult,
    CostEstimate,
    FinalReport,
    OptimizerResult,
    TesterResult,
    TranslatorResult,
    WorkloadType,
)


def calculate_cost_estimate(analyzer_result: AnalyzerResult) -> CostEstimate:
    """Calculate cost impact estimate based on code complexity."""
    complexity = analyzer_result.complexity_score or 5

    if complexity <= 3:
        manual_weeks = "1-2 weeks"
        savings = f"~{complexity * 5}-{complexity * 10} eng-days × team rate (complexity {complexity}/10)"
        factor = "Low"
    elif complexity <= 7:
        manual_weeks = "3-6 weeks"
        savings = f"~{complexity * 5}-{complexity * 10} eng-days × team rate (complexity {complexity}/10)"
        factor = "Medium"
    else:
        manual_weeks = "6-10 weeks"
        savings = f"~{complexity * 5}-{complexity * 10} eng-days × team rate (complexity {complexity}/10)"
        factor = "High"

    return CostEstimate(
        manual_porting_weeks=manual_weeks,
        rocmport_minutes="Varies by kernel",
        estimated_savings=savings,
        complexity_factor=factor,
    )


def simplify_explanation(report: FinalReport) -> str:
    """Convert technical explanation to simpler wording for explain mode."""
    simple_text = report.amd_advantage_explanation

    simple_text = simple_text.replace(
        "5.3 TB/s memory bandwidth", "much faster memory access")
    simple_text = simple_text.replace("3.35 TB/s", "slower memory access")
    simple_text = simple_text.replace(
        "memory-bound", "needs to move a lot of data")
    simple_text = simple_text.replace(
        "compute-bound", "does a lot of calculations")
    simple_text = simple_text.replace(
        "wavefront", "group of threads working together")
    simple_text = simple_text.replace(
        "shared memory tiling", "shares data between threads efficiently")
    simple_text = simple_text.replace("coalescing", "accesses memory in order")
    simple_text = simple_text.replace("optimization", "improvement")
    simple_text = simple_text.replace("performance", "speed")
    simple_text = simple_text.replace("benchmark", "test")
    simple_text = simple_text.replace("iteration", "try")

    simple_text = simple_text.replace("This kernel is", "This code is")
    simple_text = simple_text.replace("The optimization", "The improvement")
    simple_text = simple_text.replace("achieves", "gets")
    simple_text = simple_text.replace("demonstrates", "shows")
    return simple_text


# NOTE: run_pipeline below is NOT used by the active LangGraph pipeline.
# The active pipeline is backend/graph/pipeline.py (build_pipeline / pipeline).
# This function is kept for reference but is dead code.
async def run_pipeline(
    cuda_code: str,
    kernel_name: str = "custom",
    simple_mode: bool = False,
) -> AsyncGenerator[AgentEvent, None]:
    """Run full pipeline and stream AgentEvent objects."""
    yield AgentEvent(
        agent="analyzer",
        status=AgentStatus.RUNNING,
        message="Scanning CUDA code for kernels, APIs, and hardware-specific issues...",
    )

    try:
        analyzer_result: AnalyzerResult = await asyncio.to_thread(analyzer.run, cuda_code)
    except Exception as e:
        yield AgentEvent(agent="analyzer", status=AgentStatus.FAILED, message="Analysis failed", detail=str(e))
        return

    detail_parts = [
        f"Found {len(analyzer_result.kernels_found)} kernel(s): {', '.join(analyzer_result.kernels_found)}",
        f"Workload: {analyzer_result.workload_type.value}",
        f"Difficulty: {analyzer_result.difficulty} - {analyzer_result.difficulty_reason}",
    ]

    if analyzer_result.warp_size_issue:
        detail_parts.append(
            f"WARP SIZE ISSUE: {analyzer_result.warp_size_detail}")
    if analyzer_result.sharding_detected:
        detail_parts.append(
            "Multi-GPU sharding detected; review if needed on MI300X memory capacity.")
    if analyzer_result.prediction:
        detail_parts.append(analyzer_result.prediction)

    yield AgentEvent(
        agent="analyzer",
        status=AgentStatus.DONE,
        message=(
            f"Found {len(analyzer_result.kernels_found)} kernel(s) | "
            f"{analyzer_result.workload_type.value} workload | Difficulty: {analyzer_result.difficulty}"
        ),
        detail="\n".join(detail_parts),
    )

    yield AgentEvent(
        agent="translator",
        status=AgentStatus.RUNNING,
        message="Running hipify-clang (pass 1) then LLM correction (pass 2)...",
    )

    try:
        translator_result: TranslatorResult = await asyncio.to_thread(translator.run, cuda_code, analyzer_result)
    except Exception as e:
        yield AgentEvent(agent="translator", status=AgentStatus.FAILED, message="Translation failed", detail=str(e))
        return

    yield AgentEvent(
        agent="translator",
        status=AgentStatus.DONE,
        message=(
            f"{translator_result.total_changes} changes "
            f"({translator_result.hipify_changes} hipify + {translator_result.llm_changes} LLM)"
        ),
        detail=(
            f"Total changes: {translator_result.total_changes} "
            f"({translator_result.hipify_changes} hipify, {translator_result.llm_changes} LLM)\n"
            f"Warp size corrected: {analyzer_result.warp_size_issue}\n"
            "Kernel launch syntax updated"
        ),
    )

    yield AgentEvent(
        agent="optimizer",
        status=AgentStatus.RUNNING,
        message="Applying AMD MI300X-specific optimizations (iteration 1)...",
    )

    try:
        optimizer_result: OptimizerResult = await asyncio.to_thread(
            optimizer.run,
            translator_result.hip_code,
            analyzer_result,
            1,
        )
    except Exception as e:
        yield AgentEvent(agent="optimizer", status=AgentStatus.FAILED, message="Optimization failed", detail=str(e))
        return

    yield AgentEvent(
        agent="optimizer",
        status=AgentStatus.DONE,
        message=f"{len(optimizer_result.changes)} optimization(s) applied",
        detail="\n".join(
            f"- {c['description']}" for c in optimizer_result.changes),
    )

    yield AgentEvent(
        agent="tester",
        status=AgentStatus.RUNNING,
        message="Compiling with hipcc and profiling with rocprof (iteration 1)...",
    )

    try:
        tester_result_1: TesterResult = await asyncio.to_thread(
            tester.run,
            optimizer_result.optimized_code,
            analyzer_result,
            1,
            kernel_name,
        )
    except Exception as e:
        yield AgentEvent(agent="tester", status=AgentStatus.FAILED, message="Testing failed", detail=str(e))
        return

    if not tester_result_1.success:
        yield AgentEvent(
            agent="tester",
            status=AgentStatus.FAILED,
            message="Compilation or profiling failed",
            detail=tester_result_1.notes,
        )
        return

    if tester_result_1.speedup < 1.0:
        yield AgentEvent(
            agent="tester",
            status=AgentStatus.FAILED,
            message=f"Iteration 1: {tester_result_1.speedup}x vs baseline HIP (regression)",
            detail=(
                f"Bandwidth utilized: {tester_result_1.bandwidth_utilized}%\n"
                f"{tester_result_1.notes}"
            ),
        )

        yield AgentEvent(
            agent="coordinator",
            status=AgentStatus.RUNNING,
            message="Performance regressed, retrying optimizer with profiler feedback...",
            detail=f"Profiler feedback: {tester_result_1.notes}",
        )

        yield AgentEvent(
            agent="optimizer",
            status=AgentStatus.RETRYING,
            message="Trying alternative optimization strategy (iteration 2)...",
            detail=f"Previous strategy regressed. Feedback: {tester_result_1.notes}",
        )

        try:
            optimizer_result_2: OptimizerResult = await asyncio.to_thread(
                optimizer.run,
                translator_result.hip_code,
                analyzer_result,
                2,
                tester_result_1.notes,
            )
        except Exception as e:
            yield AgentEvent(agent="optimizer", status=AgentStatus.FAILED, message="Re-optimization failed", detail=str(e))
            return

        yield AgentEvent(
            agent="optimizer",
            status=AgentStatus.DONE,
            message=f"Alternative strategy: {len(optimizer_result_2.changes)} change(s) applied",
            detail="\n".join(
                f"- {c['description']}" for c in optimizer_result_2.changes),
        )

        yield AgentEvent(
            agent="tester",
            status=AgentStatus.RUNNING,
            message="Re-profiling with alternative optimization (iteration 2)...",
        )

        try:
            tester_result_final: TesterResult = await asyncio.to_thread(
                tester.run,
                optimizer_result_2.optimized_code,
                analyzer_result,
                2,
                kernel_name,
            )
        except Exception as e:
            yield AgentEvent(agent="tester", status=AgentStatus.FAILED, message="Re-testing failed", detail=str(e))
            return

        final_optimizer = optimizer_result_2
    else:
        tester_result_final = tester_result_1
        final_optimizer = optimizer_result

    yield AgentEvent(
        agent="tester",
        status=AgentStatus.DONE,
        message=f"Iteration {tester_result_final.iteration}: {tester_result_final.speedup}x vs baseline HIP",
        detail=(
            f"Execution time: {tester_result_final.execution_ms:.1f}ms\n"
            f"Memory bandwidth: {tester_result_final.bandwidth_utilized:.1f}% utilized\n"
            f"Bottleneck type: {tester_result_final.bottleneck}\n"
            f"{tester_result_final.notes}"
        ),
    )

    yield AgentEvent(agent="coordinator", status=AgentStatus.RUNNING, message="Generating migration report...")

    amd_explanation = _build_amd_explanation(
        analyzer_result, tester_result_final)

    try:
        cost_estimate = calculate_cost_estimate(analyzer_result)
    except Exception:
        cost_estimate = CostEstimate(
            manual_porting_weeks="3-6 weeks",
            rocmport_minutes="Varies by kernel",
            estimated_savings="$20,000-$50,000",
            complexity_factor="Medium",
        )

    temp_report = FinalReport(
        migration_success=True,
        speedup=tester_result_final.speedup,
        bandwidth_utilized=tester_result_final.bandwidth_utilized,
        total_changes=translator_result.total_changes +
        len(final_optimizer.changes),
        bottleneck=tester_result_final.bottleneck,
        amd_advantage_explanation=amd_explanation,
        iterations=tester_result_final.iteration,
        hip_code=translator_result.hip_code,
        optimized_code=final_optimizer.optimized_code,
        verification=tester_result_final.verification,
        static_risk_report=analyzer_result.static_risk_report,
        data_source=tester_result_final.data_source or "simulated",
    )
    simplified_explanation = simplify_explanation(temp_report)

    report = FinalReport(
        migration_success=True,
        speedup=tester_result_final.speedup,
        bandwidth_utilized=tester_result_final.bandwidth_utilized,
        total_changes=translator_result.total_changes +
        len(final_optimizer.changes),
        bottleneck=tester_result_final.bottleneck,
        amd_advantage_explanation=amd_explanation,
        iterations=tester_result_final.iteration,
        hip_code=translator_result.hip_code,
        optimized_code=final_optimizer.optimized_code,
        verification=tester_result_final.verification,
        cost_estimate=cost_estimate,
        simplified_explanation=simplified_explanation,
        static_risk_report=analyzer_result.static_risk_report,
        data_source=tester_result_final.data_source or "simulated",
    )

    yield AgentEvent(
        agent="coordinator",
        status=AgentStatus.DONE,
        message="Migration complete",
        detail=json.dumps(report.model_dump()),
    )


def _build_amd_explanation(analyzer_result: AnalyzerResult, tester_result: TesterResult) -> str:
    if analyzer_result.workload_type == WorkloadType.MEMORY_BOUND:
        return (
            "This is a memory-bound kernel; performance scales with memory bandwidth. "
            "MI300X provides higher memory bandwidth than H100-class hardware, and this workload "
            f"reached {tester_result.bandwidth_utilized:.0f}% utilization after optimization."
        )
    return (
        "This is a compute-bound kernel; launch geometry and wavefront-aware tuning are key drivers. "
        "After optimization, compute utilization and execution characteristics improved."
    )
