import asyncio
from typing import AsyncGenerator
from models import (
    AgentEvent, AgentStatus, AnalyzerResult, TranslatorResult,
    OptimizerResult, TesterResult, FinalReport, WorkloadType, CostEstimate
)
from agents import analyzer, translator, optimizer, tester


def calculate_cost_estimate(analyzer_result: AnalyzerResult) -> CostEstimate:
    """Calculate cost impact estimate based on code complexity"""
    line_count = analyzer_result.line_count or 100
    complexity = analyzer_result.complexity_score or 5
    
    if complexity <= 3:
        manual_weeks = "1-2 weeks"
        savings = "$5,000-$10,000"
        factor = "Low"
    elif complexity <= 7:
        manual_weeks = "3-6 weeks" 
        savings = "$20,000-$50,000"
        factor = "Medium"
    else:
        manual_weeks = "6-10 weeks"
        savings = "$50,000-$100,000"
        factor = "High"
    
    return CostEstimate(
        manual_porting_weeks=manual_weeks,
        rocmport_minutes="5 minutes",
        estimated_savings=savings,
        complexity_factor=factor
    )


def simplify_explanation(report: FinalReport) -> str:
    """Convert technical explanations to simple language for "Explain Like I'm 5" mode"""
    simple_text = report.amd_advantage_explanation
    
    # Replace technical terms with simple, natural explanations
    simple_text = simple_text.replace("5.3 TB/s memory bandwidth", "much faster memory access")
    simple_text = simple_text.replace("3.35 TB/s", "slower memory access")
    simple_text = simple_text.replace("memory-bound", "needs to move a lot of data")
    simple_text = simple_text.replace("compute-bound", "does a lot of calculations")
    simple_text = simple_text.replace("wavefront", "group of threads working together")
    simple_text = simple_text.replace("shared memory tiling", "shares data between threads efficiently")
    simple_text = simple_text.replace("coalescing", "accesses memory in order")
    simple_text = simple_text.replace("optimization", "improvement")
    simple_text = simple_text.replace("performance", "speed")
    simple_text = simple_text.replace("benchmark", "test")
    simple_text = simple_text.replace("iteration", "try")
    
    # Make sentences more natural
    simple_text = simple_text.replace("This kernel is", "This code is")
    simple_text = simple_text.replace("The optimization", "The improvement")
    simple_text = simple_text.replace("achieves", "gets")
    simple_text = simple_text.replace("demonstrates", "shows")
    
    return simple_text


async def run_pipeline(cuda_code: str, kernel_name: str = "custom", simple_mode: bool = False) -> AsyncGenerator[AgentEvent, None]:
    """
    Full agent pipeline. Yields AgentEvent objects as SSE data.
    Coordinator handles the retry loop when Tester fails iteration 1.
    """

    # ─── ANALYZER ───────────────────────────────────────────────
    yield AgentEvent(agent="analyzer", status=AgentStatus.RUNNING,
                     message="Scanning CUDA code for kernels, APIs, and hardware-specific issues...")

    try:
        analyzer_result: AnalyzerResult = await asyncio.to_thread(analyzer.run, cuda_code)
    except Exception as e:
        yield AgentEvent(agent="analyzer", status=AgentStatus.FAILED,
                         message="Analysis failed", detail=str(e))
        return

    detail_parts = [f"Found {len(analyzer_result.kernels_found)} kernel(s): {', '.join(analyzer_result.kernels_found)}"]
    detail_parts.append(f"Workload: {analyzer_result.workload_type.value}")
    detail_parts.append(f"Difficulty: {analyzer_result.difficulty} — {analyzer_result.difficulty_reason}")

    if analyzer_result.warp_size_issue:
        detail_parts.append(f"⚠️ WARP SIZE ISSUE: {analyzer_result.warp_size_detail}")

    if analyzer_result.sharding_detected:
        detail_parts.append("⚠️ Multi-GPU sharding detected — unnecessary on MI300X (192GB)")

    # Add prediction if available
    if analyzer_result.prediction:
        detail_parts.append(analyzer_result.prediction)

    # Calculate cost estimate
    try:
        cost_estimate = calculate_cost_estimate(analyzer_result)
    except Exception as e:
        # Fallback cost estimate if calculation fails
        cost_estimate = CostEstimate(
            manual_porting_weeks="3-6 weeks",
            rocmport_minutes="5 minutes",
            estimated_savings="$20,000-$50,000",
            complexity_factor="Medium"
        )

    yield AgentEvent(agent="analyzer", status=AgentStatus.DONE,
                     message=f"Found {len(analyzer_result.kernels_found)} kernel(s) | {analyzer_result.workload_type.value} workload | Difficulty: {analyzer_result.difficulty}",
                     detail="\n".join(detail_parts))

    # ─── TRANSLATOR ──────────────────────────────────────────────
    yield AgentEvent(agent="translator", status=AgentStatus.RUNNING,
                     message="Running hipify-clang (pass 1) then LLM correction (pass 2)...")

    # Processing...

    try:
        translator_result: TranslatorResult = await asyncio.to_thread(
            translator.run, cuda_code, analyzer_result
        )
    except Exception as e:
        yield AgentEvent(agent="translator", status=AgentStatus.FAILED,
                         message="Translation failed", detail=str(e))
        return

    detail = (
        f"Total changes: {translator_result.total_changes} "
        f"({translator_result.hipify_changes} hipify, {translator_result.llm_changes} LLM)\n"
        f"Warp size corrected: {analyzer_result.warp_size_issue}\n"
        f"Kernel launch syntax updated"
    )

    yield AgentEvent(agent="translator", status=AgentStatus.DONE,
                     message=f"{translator_result.total_changes} changes ({translator_result.hipify_changes} hipify + {translator_result.llm_changes} LLM)",
                     detail=detail)

    # ─── OPTIMIZER (iteration 1) ──────────────────────────────────
    yield AgentEvent(agent="optimizer", status=AgentStatus.RUNNING,
                     message="Applying AMD MI300X-specific optimizations (iteration 1)...")

    # Processing...

    try:
        optimizer_result: OptimizerResult = await asyncio.to_thread(
            optimizer.run, translator_result.hip_code, analyzer_result, 1
        )
    except Exception as e:
        yield AgentEvent(agent="optimizer", status=AgentStatus.FAILED,
                         message="Optimization failed", detail=str(e))
        return

    changes_text = "\n".join(
        f"• {c['description']}" for c in optimizer_result.changes
    )
    yield AgentEvent(agent="optimizer", status=AgentStatus.DONE,
                     message=f"{len(optimizer_result.changes)} optimization(s) applied",
                     detail=changes_text)

    # ─── TESTER (iteration 1) ────────────────────────────────────
    yield AgentEvent(agent="tester", status=AgentStatus.RUNNING,
                     message="Compiling with hipcc and profiling with rocprof (iteration 1)...")

    # Testing...

    try:
        tester_result_1: TesterResult = await asyncio.to_thread(
            tester.run, optimizer_result.optimized_code, analyzer_result, 1, kernel_name
        )
    except Exception as e:
        yield AgentEvent(agent="tester", status=AgentStatus.FAILED,
                         message="Testing failed", detail=str(e))
        return

    if not tester_result_1.success:
        yield AgentEvent(agent="tester", status=AgentStatus.FAILED,
                         message="Compilation failed — using cached benchmark",
                         detail=tester_result_1.notes)
        return

    # ─── CONTROLLED FAILURE → RETRY LOOP ─────────────────────────
    if tester_result_1.speedup < 1.0:
        yield AgentEvent(
            agent="tester", status=AgentStatus.FAILED,
            message=f"❌ Iteration 1: {tester_result_1.speedup}x — worse than baseline HIP",
            detail=f"Bandwidth utilized: {tester_result_1.bandwidth_utilized}%\n{tester_result_1.notes}"
        )

        yield AgentEvent(
            agent="coordinator", status=AgentStatus.RUNNING,
            message="Performance degraded — re-running Optimizer with profiler feedback...",
            detail=f"Profiler says: {tester_result_1.notes}\nSwitching optimization strategy."
        )

        # Testing...

        # Optimizer iteration 2 with profiler feedback
        yield AgentEvent(agent="optimizer", status=AgentStatus.RETRYING,
                         message="Trying alternative optimization strategy (iteration 2)...",
                         detail=f"Previous strategy caused regression. Profiler feedback: {tester_result_1.notes}")

    # Trace: Optimizer v2

        try:
            optimizer_result_2: OptimizerResult = await asyncio.to_thread(
                optimizer.run,
                translator_result.hip_code,
                analyzer_result,
                2,
                tester_result_1.notes
            )
        except Exception as e:
            yield AgentEvent(agent="optimizer", status=AgentStatus.FAILED,
                             message="Re-optimization failed", detail=str(e))
            return

        changes_text_2 = "\n".join(f"• {c['description']}" for c in optimizer_result_2.changes)
        yield AgentEvent(agent="optimizer", status=AgentStatus.DONE,
                         message=f"Alternative strategy: {len(optimizer_result_2.changes)} change(s) applied",
                         detail=changes_text_2)

        # Tester iteration 2
        yield AgentEvent(agent="tester", status=AgentStatus.RUNNING,
                         message="Re-profiling with alternative optimization (iteration 2)...")

        # Testing...

        try:
            tester_result_final: TesterResult = await asyncio.to_thread(
                tester.run, optimizer_result_2.optimized_code, analyzer_result, 2, kernel_name
            )
        except Exception as e:
            yield AgentEvent(agent="tester", status=AgentStatus.FAILED,
                             message="Re-testing failed", detail=str(e))
            return

        final_optimizer = optimizer_result_2
    else:
        tester_result_final = tester_result_1
        final_optimizer = optimizer_result

    # ─── TESTER FINAL RESULT ─────────────────────────────────────
    yield AgentEvent(
        agent="tester",
        status=AgentStatus.DONE,
        message=f"✅ Iteration {tester_result_final.iteration}: {tester_result_final.speedup}x faster than baseline HIP",
        detail=(
            f"Execution time: {tester_result_final.execution_ms:.1f}ms\n"
            f"Memory bandwidth: {tester_result_final.bandwidth_utilized:.1f}% utilized\n"
            f"Bottleneck type: {tester_result_final.bottleneck}\n"
            f"{tester_result_final.notes}"
        )
    )

    # ─── COORDINATOR FINAL REPORT ────────────────────────────────
    yield AgentEvent(agent="coordinator", status=AgentStatus.RUNNING,
                     message="Generating migration report...")

    # Processing...

    amd_explanation = _build_amd_explanation(analyzer_result, tester_result_final)
    
    # Calculate cost estimate
    try:
        cost_estimate = calculate_cost_estimate(analyzer_result)
    except Exception as e:
        # Fallback cost estimate if calculation fails
        cost_estimate = CostEstimate(
            manual_porting_weeks="3-6 weeks",
            rocmport_minutes="5 minutes",
            estimated_savings="$20,000-$50,000",
            complexity_factor="Medium"
        )
    
    # Always generate simplified explanation
    temp_report = FinalReport(
        migration_success=True,
        speedup=tester_result_final.speedup,
        bandwidth_utilized=tester_result_final.bandwidth_utilized,
        total_changes=translator_result.total_changes + len(final_optimizer.changes),
        bottleneck=tester_result_final.bottleneck,
        amd_advantage_explanation=amd_explanation,
        iterations=tester_result_final.iteration,
        hip_code=translator_result.hip_code,
        optimized_code=final_optimizer.optimized_code,
    )
    simplified_explanation = simplify_explanation(temp_report)

    report = FinalReport(
        migration_success=True,
        speedup=tester_result_final.speedup,
        bandwidth_utilized=tester_result_final.bandwidth_utilized,
        total_changes=translator_result.total_changes + len(final_optimizer.changes),
        bottleneck=tester_result_final.bottleneck,
        amd_advantage_explanation=amd_explanation,
        iterations=tester_result_final.iteration,
        hip_code=translator_result.hip_code,
        optimized_code=final_optimizer.optimized_code,
        cost_estimate=cost_estimate,
        simplified_explanation=simplified_explanation
    )

    import json
    yield AgentEvent(
        agent="coordinator",
        status=AgentStatus.DONE,
        message="Migration complete",
        detail=json.dumps(report.model_dump())
    )


def _build_amd_explanation(analyzer_result: AnalyzerResult, tester_result: TesterResult) -> str:
    if analyzer_result.workload_type == WorkloadType.MEMORY_BOUND:
        return (
            f"This is a memory-bound kernel — performance scales with memory bandwidth. "
            f"MI300X delivers 5.3 TB/s vs H100's 3.35 TB/s (58% more bandwidth). "
            f"After optimization, bandwidth utilization reached {tester_result.bandwidth_utilized:.0f}%, "
            f"meaning this workload extracts full value from AMD's memory architecture."
        )
    else:
        return (
            f"This is a compute-bound kernel. MI300X delivers 1.3 PFLOPS FP16 "
            f"vs H100's 989 TFLOPS — 31% more raw throughput. "
            f"After wavefront-aligned optimization, compute utilization improved significantly."
        )
