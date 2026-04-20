# pylint: disable=broad-exception-caught

from backend.agents.analyzer import AnalyzerResult, WorkloadType
from backend.agents.tester import run as run_tester
from backend.agents.coordinator import run_pipeline
from backend.models import PortRequest, ColdStartRequest, AggregateMetricsRequest
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import json
import asyncio
import zipfile
import io
import os
import difflib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


app = FastAPI(
    title="ROCmPort AI",
    description="CUDA-to-ROCm migration assistant with iterative testing and optimization.",
    version="1.0.0",
    contact={
        "name": "Tazwar Ahnaf Enan",
        "url": "https://github.com/tazwaryayyyy",
        "email": "tazwardevp@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ROCmPort AI"}


@app.post("/port")
async def port_cuda_code(req: PortRequest):
    """
    Main endpoint. Streams SSE events as the agent pipeline runs.
    Each event is a JSON AgentEvent object.
    """
    if not req.cuda_code or len(req.cuda_code.strip()) < 10:
        raise HTTPException(status_code=400, detail="No CUDA code provided")

    async def event_stream():
        try:
            async for event in run_pipeline(req.cuda_code, req.kernel_name or "custom", req.simple_mode or False):
                data = json.dumps(event.model_dump())
                yield f"data: {data}\n\n"
                # Let the client breathe between events
                await asyncio.sleep(0.05)
        except Exception as e:
            error_event = {
                "agent": "coordinator",
                "status": "failed",
                "message": "Pipeline error",
                "detail": str(e)
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


async def _collect_pipeline_events(cuda_code: str, kernel_name: str, simple_mode: bool = False) -> tuple[list[dict], dict | None]:
    """Collect all pipeline events and extract final report payload when present."""
    events: list[dict] = []
    final_report = None

    async for event in run_pipeline(cuda_code, kernel_name, simple_mode):
        dumped = event.model_dump()
        events.append(dumped)
        if dumped.get("agent") == "coordinator" and dumped.get("status") == "done" and dumped.get("detail"):
            try:
                final_report = json.loads(dumped["detail"])
            except (json.JSONDecodeError, TypeError):
                final_report = None

    return events, final_report


def _has_adaptation_loop(events: list[dict]) -> bool:
    """Return True when the run shows retry-based adaptation behavior."""
    saw_regression = any(
        e.get("agent") == "tester" and e.get(
            "status") == "failed" and "regression" in str(e.get("message", "")).lower()
        for e in events
    )
    saw_retry = any(
        e.get("agent") == "optimizer" and e.get("status") == "retrying"
        for e in events
    )
    return saw_regression and saw_retry


@app.post("/cold-start")
async def cold_start_run(req: ColdStartRequest):
    """
    Single-run endpoint for unknown pasted CUDA input.
    Returns full trace plus summary trust signals.
    """
    if not req.cuda_code or len(req.cuda_code.strip()) < 10:
        raise HTTPException(status_code=400, detail="No CUDA code provided")

    events, report = await _collect_pipeline_events(req.cuda_code, req.kernel_name or "unknown_input", False)

    if report is None:
        raise HTTPException(
            status_code=500, detail="Pipeline completed without final report")

    return {
        "success": True,
        "kernel_name": req.kernel_name or "unknown_input",
        "adaptation_loop_observed": _has_adaptation_loop(events),
        "event_count": len(events),
        "report": report,
        "events": events,
    }


@app.post("/aggregate-metric")
async def aggregate_metric(req: AggregateMetricsRequest):
    """
    Evaluate multiple kernels and return one aggregate metric:
    average speedup vs baseline HIP.
    """
    kernels_dir = os.path.join(os.path.dirname(__file__), "demo_kernels")
    requested = req.kernel_names or []

    available: dict[str, str] = {}
    for fname in os.listdir(kernels_dir):
        if fname.endswith(".cu"):
            kname = fname.replace(".cu", "")
            with open(os.path.join(kernels_dir, fname), encoding="utf-8") as f:
                available[kname] = f.read()

    selected_names = requested if requested else sorted(available.keys())
    selected_names = [name for name in selected_names if name in available]

    if not selected_names:
        raise HTTPException(
            status_code=400, detail="No valid kernels selected for aggregation")

    runs = []
    speedups = []

    for name in selected_names:
        events, report = await _collect_pipeline_events(available[name], name, False)
        if report is None:
            continue

        speedup = float(report.get("speedup", 0.0) or 0.0)
        speedups.append(speedup)
        runs.append({
            "kernel": name,
            "speedup": speedup,
            "adaptation_loop_observed": _has_adaptation_loop(events),
            "iterations": report.get("iterations", 1),
        })

    if not speedups:
        raise HTTPException(
            status_code=500, detail="Unable to produce aggregate metric from selected kernels")

    avg_speedup = round(sum(speedups) / len(speedups), 3)
    avg_improvement_pct = round((avg_speedup - 1.0) * 100.0, 2)

    return {
        "success": True,
        "baseline": "straight hipify output with minimal compile edits",
        "kernel_count": len(speedups),
        "aggregate_metric": {
            "average_speedup_vs_baseline": avg_speedup,
            "average_improvement_percent": avg_improvement_pct,
        },
        "runs": runs,
    }


@app.post("/recompile")
async def recompile_edited_code(req: dict):
    """
    Recompile endpoint for human override feature.
    Accepts edited HIP code and re-runs tester.
    """
    try:
        edited_code = req.get("edited_code")
        kernel_name = req.get("kernel_name", "custom")

        if not edited_code or len(edited_code.strip()) < 10:
            raise HTTPException(status_code=400, detail="No HIP code provided")

        # Create a mock analyzer result for testing
        analyzer_result = AnalyzerResult(
            kernels_found=["test_kernel"],
            cuda_apis=["hipMalloc", "hipMemcpy"],
            warp_size_issue=False,
            warp_size_detail=None,
            workload_type=WorkloadType.MEMORY_BOUND,
            sharding_detected=False,
            difficulty="Easy",
            difficulty_reason="Simple test kernel"
        )

        # Run tester with edited code
        tester_result = await asyncio.to_thread(run_tester, edited_code, analyzer_result, 2, kernel_name)

        return {
            "success": True,
            "result": tester_result.model_dump()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recompilation failed: {str(e)}") from e


@app.post("/export")
async def export_migration_package(req: dict):
    """
    Export endpoint for GitHub PR simulation.
    Returns a zip file with diff and migration report.
    """
    try:
        migration_report = req.get("migration_report", {})
        if not isinstance(migration_report, dict):
            migration_report = {}

        original_cuda = str(req.get("original_cuda") or "")
        # Fallback to report content when frontend omits final_rocm.
        final_rocm = str(req.get("final_rocm")
                         or migration_report.get("optimized_code") or "")

        if not final_rocm.strip():
            raise HTTPException(
                status_code=400, detail="No ROCm code provided for export")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add professional unified diff
            diff = difflib.unified_diff(
                original_cuda.splitlines(keepends=True),
                final_rocm.splitlines(keepends=True),
                fromfile="original.cu",
                tofile="optimized.hip"
            )
            diff_text = "".join(diff)
            zf.writestr("migration.diff", diff_text)

            # Include source snapshots for easier review in PRs.
            zf.writestr("original.cu", original_cuda)
            zf.writestr("optimized.hip", final_rocm)

            # Add migration report as markdown
            md_report = f"""# ROCmPort AI Migration Report

## Performance Results
- Speedup: {migration_report.get('speedup', 'N/A')}x
- Bandwidth Utilization: {migration_report.get('bandwidth_utilized', 'N/A')}%
- Total Changes: {migration_report.get('total_changes', 'N/A')}

## AMD Advantage Explanation
{migration_report.get('amd_advantage_explanation', 'N/A')}

## Cost Impact
{migration_report.get('cost_estimate', 'N/A')}

Generated by ROCmPort AI.
"""
            zf.writestr("migration_report.md", md_report)

        zip_content = zip_buffer.getvalue()

        from fastapi.responses import Response
        return Response(
            content=zip_content,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=rocmport_migration.zip"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Export failed: {str(e)}") from e


@app.get("/demo-kernels")
async def list_demo_kernels():
    kernels_dir = os.path.join(os.path.dirname(__file__), "demo_kernels")
    kernels = {}
    for fname in os.listdir(kernels_dir):
        if fname.endswith(".cu"):
            name = fname.replace(".cu", "")
            with open(os.path.join(kernels_dir, fname), encoding="utf-8") as f:
                kernels[name] = f.read()
    return kernels


# Serve frontend if built
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path,
              html=True), name="frontend")
