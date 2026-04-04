import subprocess
import tempfile
import os
import re


class HipifyWrapper:
    """Wrapper for hipify-clang tool with Python fallback"""
    
    def __init__(self):
        pass
    
    def hipify_code(self, cuda_code: str) -> tuple[str, list[dict]]:
        """
        Try to run real hipify-clang if available.
        Falls back to Python-based pattern replacement.
        Returns (hip_code, list of changes made)
        """
        # Try real hipify first
        if self._hipify_available():
            result = self._run_real_hipify(cuda_code)
            if result:
                return result

        # Fallback: Python pattern replacement
        return self._python_hipify(cuda_code)
    
    def _hipify_available(self) -> bool:
        try:
            result = subprocess.run(
                ["hipify-clang", "--version"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_real_hipify(self, cuda_code: str) -> tuple[str, list[dict]] | None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
                f.write(cuda_code)
                tmp_path = f.name

            # Use -- separator to pass compiler flags to the internal Clang parser
            # This is critical for Clang-based tools to distinguish tool flags from compiler flags.
            cmd = ["hipify-clang", tmp_path, "--", "-nocudalib", "-nocudainc", "-arch=sm_60"]
            
            # Debug log for build engineering
            print(f"DEBUG: Running hipify-clang command: {' '.join(cmd)}")
            
            # Set environment variable just in case hipify-clang invokes nvcc internally
            env = os.environ.copy()
            env['NVCC_APPEND_FLAGS'] = '-nocudalib -arch=sm_60'
            
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=30,
                env=env
            )

            if result.returncode != 0:
                print(f"DEBUG: hipify-clang failed with return code {result.returncode}")
                print(f"DEBUG: stderr: {result.stderr}")

            if result.returncode == 0 and result.stdout:
                changes = self._detect_changes(cuda_code, result.stdout, source="hipify-clang")
                return result.stdout, changes

            return None
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _python_hipify(self, cuda_code: str) -> tuple[str, list[dict]]:
        """Python-based hipify — handles the mechanical replacements."""
        hip_code = cuda_code
        changes = []

        for cuda_api, hip_api in HIPIFY_MAP.items():
            if cuda_api in hip_code and cuda_api != hip_api:
                count = hip_code.count(cuda_api)
                hip_code = hip_code.replace(cuda_api, hip_api)
                changes.append({
                    "old": cuda_api,
                    "new": hip_api,
                    "count": count,
                    "source": "hipify",
                    "confidence": "high"
                })

        # Fix kernel launch syntax: kernel<<<blocks, threads>>> → hipLaunchKernelGGL
        # Keep it as-is for now — LLM handles complex launch syntax
        # Simple <<<>>> launches are valid in HIP too

        return hip_code, changes

    def _detect_changes(self, original: str, converted: str, source: str) -> list[dict]:
        """Detect what changed between original and converted code."""
        changes = []
        orig_lines = original.splitlines()
        conv_lines = converted.splitlines()

        for i, (o, c) in enumerate(zip(orig_lines, conv_lines)):
            if o != c:
                changes.append({
                    "line": i + 1,
                    "old": o.strip(),
                    "new": c.strip(),
                    "source": source,
                    "confidence": "high"
                })

        return changes


# Legacy function for backward compatibility
def run_hipify(cuda_code: str) -> tuple[str, list[dict]]:
    """Legacy function - use HipifyWrapper.hipify_code instead"""
    wrapper = HipifyWrapper()
    return wrapper.hipify_code(cuda_code)


# Common CUDA → HIP replacements hipify handles
HIPIFY_MAP = {
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaSuccess": "hipSuccess",
    "cudaError_t": "hipError_t",
    "cudaGetLastError": "hipGetLastError",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaEventCreate": "hipEventCreate",
    "cudaEventRecord": "hipEventRecord",
    "cudaEventSynchronize": "hipEventSynchronize",
    "cudaEventElapsedTime": "hipEventElapsedTime",
    "cudaEventDestroy": "hipEventDestroy",
    "cudaEvent_t": "hipEvent_t",
    "cudaStream_t": "hipStream_t",
    "cudaStreamCreate": "hipStreamCreate",
    "cudaStreamDestroy": "hipStreamDestroy",
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "__syncthreads": "__syncthreads",   # same in HIP
}
