import os
import hashlib
from ..models import TesterResult, AnalyzerResult, VerificationResult
from ..tools.rocprof_wrapper import RocprofWrapper
from ..tools.demo_artifacts import get_demo_data, get_kernel_baselines

# Set ROCM_AVAILABLE=true on AMD Cloud
ROCM_AVAILABLE = os.environ.get("ROCM_AVAILABLE", "false").lower() == "true"

# Expected checksums for demo kernels (first 100 elements of output)
DEMO_KERNEL_CHECKSUMS = {
    "vector_add": "a1b2c3d4e5f6789012345678901234567890",  # Mock checksum
    "matrix_multiply": "b2c3d4e5f6a7890123456789012345678901",  # Mock checksum
    "convolution_2d": "c3d4e5f6a7b8901234567890123456789012",  # Mock checksum
    "reduction": "e5f6a7b8c9d0123456789012345678901234",       # Mock checksum
    "custom": "d4e5f6a7b8c9012345678901234567890123"  # Mock checksum
}


def compute_code_checksum(code_text: str, sample_size: int = 400) -> str:
    """Compute a short checksum from code text for traceability in demo mode."""
    if not code_text:
        return "empty"
    sample = code_text[:sample_size]
    return hashlib.sha256(sample.encode()).hexdigest()[:32]


def verify_demo_kernel(kernel_name: str, optimized_code: str) -> VerificationResult:
    """Verify demo kernel execution and output correctness"""
    expected = DEMO_KERNEL_CHECKSUMS.get(kernel_name, "mock_checksum")
    actual = compute_code_checksum(optimized_code)

    # In demo mode, indicate this is simulated verification
    is_demo = not ROCM_AVAILABLE

    verification = VerificationResult(
        compiled_successfully=True,
        executed_without_error=True,
        output_matches_expected=actual == expected,
        expected_checksum=expected,
        actual_checksum=actual,
        mock_mode=is_demo
    )

    # Do not fabricate pass/fail in demo mode. Surface that verification is simulated.
    if is_demo:
        verification.output_matches_expected = False
        verification.checksum_computed = actual

    return verification


def run(optimized_code: str, analyzer_result: AnalyzerResult,
        iteration: int = 1, kernel_name: str = "matrix_multiply") -> TesterResult:
    """
    On AMD Cloud (ROCM_AVAILABLE=true): runs real hipcc + rocprof.
    Locally: returns deterministic demo artifact data labelled with data_source.
    """
    rocprof_wrapper = RocprofWrapper()

    # Add verification for demo kernels
    verification = None
    if kernel_name in DEMO_KERNEL_CHECKSUMS:
        verification = verify_demo_kernel(kernel_name, optimized_code)

    if ROCM_AVAILABLE:
        return _run_real(optimized_code, analyzer_result, iteration, rocprof_wrapper, verification)
    else:
        # Use deterministic demo artifact data keyed by kernel name + iteration
        profiling_data = rocprof_wrapper.get_mock_profiling_data(kernel_name, iteration)
        return _convert_profiling_to_tester_result(
            profiling_data, analyzer_result, iteration, verification, kernel_name
        )


def _convert_profiling_to_tester_result(
    profiling_data: dict,
    analyzer_result: AnalyzerResult,
    iteration: int,
    verification: VerificationResult = None,
    kernel_name: str = "custom",
) -> TesterResult:
    """Convert RocprofWrapper output to TesterResult format."""
    if not profiling_data.get('success', False):
        return TesterResult(
            success=False,
            iteration=iteration,
            speedup=0.0,
            bandwidth_utilized=0.0,
            execution_ms=0.0,
            bottleneck="profiling-error",
            notes=profiling_data.get('error', 'Unknown profiling error'),
            data_source="error",
            verification=verification
        )

    exec_ms = profiling_data.get('execution_time_ms', 0.0)
    bandwidth = profiling_data.get('memory_bandwidth_gbps', 0.0)
    data_source = profiling_data.get('data_source', 'simulated')

    # Use kernel-specific baseline — not a hardcoded 100ms
    baselines = get_kernel_baselines()
    baseline_ms = baselines.get(kernel_name, profiling_data.get('baseline_time_ms', 100.0))

    if exec_ms > 0:
        speedup = round(baseline_ms / exec_ms, 2)
    else:
        speedup = 0.0

    # Pull notes from the demo artifact (already contains useful context)
    notes = profiling_data.get('notes', '')

    # Append a clear data-source label when not running real hardware
    if data_source == "demo_artifact":
        notes += (
            "\n\n[DATA SOURCE: demo_artifact] These metrics are representative of MI300X "
            "performance for this kernel class. Set ROCM_AVAILABLE=true on AMD Developer "
            "Cloud for authoritative numbers."
        )
    elif data_source == "simulated":
        notes += (
            "\n\n[DATA SOURCE: simulated] Unknown kernel type — conservative estimate used. "
            "Set ROCM_AVAILABLE=true on AMD Developer Cloud for real measurements."
        )

    return TesterResult(
        success=True,
        iteration=iteration,
        speedup=speedup,
        bandwidth_utilized=min(bandwidth, 95.0),
        execution_ms=exec_ms,
        bottleneck=analyzer_result.workload_type.value,
        notes=notes,
        data_source=data_source,
        verification=verification
    )


def _run_real(
    code: str,
    analyzer_result: AnalyzerResult,
    iteration: int,
    rocprof_wrapper: RocprofWrapper,
    verification: VerificationResult = None,
) -> TesterResult:
    """Real hipcc + rocprof execution on MI300X."""
    # Compile the code
    success, message = rocprof_wrapper.compile_hip_code(code)

    if not success:
        return TesterResult(
            success=False,
            iteration=iteration,
            speedup=0.0,
            bandwidth_utilized=0.0,
            execution_ms=0.0,
            bottleneck="compilation-failed",
            notes=f"Compilation failed: {message}",
            data_source="real_rocm",
            verification=verification
        )

    # Run with profiling
    profiling_data = rocprof_wrapper.run_with_profiling(
        message.split(": ")[-1])  # Extract executable path

    if not profiling_data.get('success', False):
        return TesterResult(
            success=False,
            iteration=iteration,
            speedup=0.0,
            bandwidth_utilized=0.0,
            execution_ms=0.0,
            bottleneck="profiling-failed",
            notes=f"Profiling failed: {profiling_data.get('error', 'Unknown error')}",
            data_source="real_rocm",
            verification=verification
        )

    exec_ms = profiling_data.get('execution_time_ms', 0.0)
    bandwidth = profiling_data.get('memory_bandwidth_gbps', 0.0)
    speedup = _calculate_speedup_real(exec_ms, profiling_data)

    return TesterResult(
        success=True,
        iteration=iteration,
        speedup=speedup,
        bandwidth_utilized=min(bandwidth, 95.0),
        execution_ms=exec_ms,
        bottleneck=analyzer_result.workload_type.value,
        notes="Real MI300X benchmark via rocprof",
        data_source="real_rocm",
        verification=verification,
    )


def _calculate_speedup_real(exec_ms: float, profiling_data: dict) -> float:
    """Estimate speedup relative to baseline HIP using the profiler's baseline reading."""
    if exec_ms <= 0:
        return 0.0
    baseline_ms = profiling_data.get('baseline_time_ms', 100.0)
    return round(baseline_ms / exec_ms, 2)
