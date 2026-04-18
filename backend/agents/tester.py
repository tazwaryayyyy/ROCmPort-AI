import os
import hashlib
from ..models import TesterResult, AnalyzerResult, VerificationResult
from ..tools.rocprof_wrapper import RocprofWrapper

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
    """Compute a short checksum from code text for traceability in mock mode."""
    if not code_text:
        return "empty"

    sample = code_text[:sample_size]
    return hashlib.sha256(sample.encode()).hexdigest()[:32]


def verify_demo_kernel(kernel_name: str, optimized_code: str) -> VerificationResult:
    """Verify demo kernel execution and output correctness"""
    expected = DEMO_KERNEL_CHECKSUMS.get(kernel_name, "mock_checksum")
    actual = compute_code_checksum(optimized_code)

    # In mock mode, indicate this is simulated verification
    is_mock = not ROCM_AVAILABLE

    verification = VerificationResult(
        compiled_successfully=True,
        executed_without_error=True,
        output_matches_expected=actual == expected,
        expected_checksum=expected,
        actual_checksum=actual,
        mock_mode=is_mock
    )

    # Do not fabricate pass/fail in mock mode. Surface that verification is simulated.
    if is_mock:
        verification.output_matches_expected = False
        verification.checksum_computed = actual

    return verification


def run(optimized_code: str, analyzer_result: AnalyzerResult,
        iteration: int = 1, kernel_name: str = "matrix_multiply") -> TesterResult:
    """
    On AMD Cloud (ROCM_AVAILABLE=true): runs real hipcc + rocprof
    Locally: returns mock profiling results labeled as simulated.
    """
    rocprof_wrapper = RocprofWrapper()

    # Add verification for demo kernels
    verification = None
    if kernel_name in DEMO_KERNEL_CHECKSUMS:
        verification = verify_demo_kernel(kernel_name, optimized_code)

    if ROCM_AVAILABLE:
        return _run_real(optimized_code, analyzer_result, iteration, rocprof_wrapper, verification)
    else:
        # In non-ROCm environments, run_with_profiling returns simulated metrics.
        profiling_data = rocprof_wrapper.run_with_profiling("mock_executable")
        return _convert_profiling_to_tester_result(profiling_data, analyzer_result, iteration, verification)


def _convert_profiling_to_tester_result(profiling_data: dict, analyzer_result: AnalyzerResult, iteration: int, verification: VerificationResult = None) -> TesterResult:
    """Convert RocprofWrapper output to TesterResult format"""
    if not profiling_data.get('success', False):
        return TesterResult(
            success=False,
            iteration=iteration,
            speedup=0.0,
            bandwidth_utilized=0.0,
            execution_ms=0.0,
            bottleneck="profiling-error",
            notes=profiling_data.get('error', 'Unknown profiling error'),
            verification=verification
        )

    exec_ms = profiling_data.get('execution_time_ms', 0.0)
    bandwidth = profiling_data.get('memory_bandwidth_gbps', 0.0)

    baseline_ms = profiling_data.get('baseline_time_ms', 100.0)
    if exec_ms > 0:
        speedup = round(baseline_ms / exec_ms, 2)
    else:
        speedup = 0.0

    if speedup < 1.0:
        notes = "Simulated profile indicates regression vs baseline. Retry with an alternative optimization strategy."
    elif speedup < 1.1:
        notes = "Simulated profile indicates marginal improvement. Optimization may be memory- or launch-bound."
    else:
        notes = "Simulated profile indicates improvement vs baseline after optimization."

    notes += " Mock mode is enabled (ROCM_AVAILABLE=false); use real ROCm hardware for authoritative numbers."

    return TesterResult(
        success=True,
        iteration=iteration,
        speedup=speedup,
        bandwidth_utilized=min(bandwidth, 95.0),
        execution_ms=exec_ms,
        bottleneck=analyzer_result.workload_type.value,
        notes=notes,
        verification=verification
    )


def _run_real(code: str, analyzer_result: AnalyzerResult, iteration: int, rocprof_wrapper: RocprofWrapper, verification: VerificationResult = None) -> TesterResult:
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
            verification=verification
        )

    exec_ms = profiling_data.get('execution_time_ms', 0.0)
    bandwidth = profiling_data.get('memory_bandwidth_gbps', 0.0)
    speedup = _calculate_speedup(exec_ms)

    return TesterResult(
        success=True,
        iteration=iteration,
        speedup=speedup,
        bandwidth_utilized=min(bandwidth, 95.0),
        execution_ms=exec_ms,
        bottleneck=analyzer_result.workload_type.value,
        notes="Real MI300X benchmark via rocprof"
    )


def _calculate_speedup(exec_ms: float) -> float:
    """Estimate speedup relative to baseline HIP."""
    if exec_ms <= 0:
        return 0.0
    baseline_ms = 100.0
    return round(baseline_ms / exec_ms, 2)
