import subprocess
import tempfile
import os
import re
from typing import Dict, List, Tuple


class RocprofWrapper:
    """Wrapper for AMD rocprof profiler and hipcc compiler"""

    def __init__(self):
        self.rocm_available = os.getenv(
            "ROCM_AVAILABLE", "false").lower() == "true"
        self.hipcc_path = os.getenv("HIPCC_PATH", "hipcc")
        self.rocprof_path = os.getenv("ROCPROF_PATH", "rocprof")

    def compile_hip_code(self, hip_code: str, output_file: str = None) -> Tuple[bool, str]:
        """Compile HIP code using hipcc"""
        if not self.rocm_available:
            return True, "Mock compilation successful (ROCm not available)"

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.hip', delete=False) as f:
                f.write(hip_code)
                temp_file = f.name

            if output_file is None:
                output_file = temp_file.replace('.hip', '.out')

            # Add  and --offload-arch=gfx942 to solve "Cannot find libdevice for sm_52" error
            # This ensures compilation works even if CUDA device libraries are missing.
            cmd = [self.hipcc_path, '-o', output_file,
                   temp_file, '--offload-arch=gfx942']

            # Set environment variable just in case hipcc invokes nvcc internally
            env = os.environ.copy()
            env['NVCC_APPEND_FLAGS'] = ' --offload-arch=gfx942'

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, env=env, check=False)

            if result.returncode == 0:
                return True, f"Compilation successful: {output_file}"
            else:
                return False, f"Compilation failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"Compilation error: {str(e)}"
        finally:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
            except OSError:
                pass

    def run_with_profiling(self, executable_path: str, args: List[str] = None) -> Dict:
        """Run executable with rocprof profiling"""
        if not self.rocm_available:
            # Caller should use get_mock_profiling_data(kernel_name, iteration) directly.
            return {"success": False, "error": "ROCm not available; use get_mock_profiling_data(kernel_name, iteration) instead", "execution_time_ms": 0}

        try:
            if args is None:
                args = []

            # Run with rocprof stats timing
            cmd = [self.rocprof_path, '--stats', '--', executable_path] + args
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, check=False)

            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip(
                ) or "rocprof exited with a non-zero status"
                return {
                    "success": False,
                    "error": f"Profiling failed: {detail}",
                    "execution_time_ms": 0,
                }

            # Parse rocprof output
            profiling_data = self._parse_rocprof_output(
                result.stdout, result.stderr)

            return profiling_data

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Profiling timed out", "execution_time_ms": 0}
        except (OSError, subprocess.SubprocessError) as e:
            return {"success": False, "error": f"Profiling error: {str(e)}", "execution_time_ms": 0}

    def _parse_rocprof_output(self, stdout: str, _stderr: str) -> Dict:
        """Parse rocprof --stats CSV output (Name,Calls,TotalDurationNs,AverageNs,Percentage)."""
        import csv
        import io
        try:
            metrics: Dict = {}
            reader = csv.DictReader(io.StringIO(stdout))
            for row in reader:
                name = row.get("Name", "")
                # Skip ROCm runtime helper kernels
                if "__amd_rocclr" in name:
                    continue
                avg_ns_str = row.get("AverageNs", "") or ""
                if avg_ns_str.strip():
                    avg_ns = float(avg_ns_str)
                    if avg_ns > 0:
                        metrics["execution_time_ms"] = round(
                            avg_ns / 1_000_000, 6)
                        metrics["memory_bandwidth_gbps"] = 0.0
                        metrics["gpu_utilization_percent"] = 0.0
                        metrics["sq_waves"] = 0
                        break

            if not metrics:
                return {
                    "success": False,
                    "error": "rocprof output contained no parseable kernel rows",
                    "execution_time_ms": 0,
                }

            metrics["success"] = True
            return metrics

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse rocprof output: {str(e)}",
                "execution_time_ms": 0,
            }

    def get_mock_profiling_data(self, kernel_name: str = "custom", iteration: int = 1) -> Dict:
        """Public accessor for deterministic demo profiling data used by testing layer."""
        return self._get_demo_profiling_data(kernel_name, iteration)

    def _get_demo_profiling_data(self, kernel_name: str = "custom", iteration: int = 1) -> Dict:
        """
        Return deterministic per-kernel demo profiling data.

        Replaces random.uniform() with representative MI300X values keyed by kernel name
        and iteration number. Every entry is tagged with data_source so the caller and
        the UI can show an honest provenance badge instead of fabricated numbers.
        """
        from .demo_artifacts import get_demo_data
        data = get_demo_data(kernel_name, iteration)
        data['success'] = True
        return data

    def get_hardware_info(self) -> Dict:
        """Get AMD GPU hardware information"""
        if not self.rocm_available:
            return {
                'gpu_name': 'AMD MI300X (Mock)',
                'compute_units': 120,
                'memory_size_gb': 192,
                'memory_bandwidth_tb_s': 5.3,
                'wavefront_size': 64
            }

        try:
            # Try to get real GPU info using rocminfo or similar
            cmd = ['rocminfo']
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=False)

            if result.returncode == 0:
                return self._parse_rocminfo(result.stdout)
            else:
                return self._get_mock_hardware_info()

        except (OSError, subprocess.SubprocessError):
            return self._get_mock_hardware_info()

    def _parse_rocminfo(self, output: str) -> Dict:
        """Parse rocminfo output to extract hardware info."""
        info = self._get_mock_hardware_info()  # safe MI300X defaults
        name_match = re.search(r'^\s*Name:\s+(.+)$', output, re.MULTILINE)
        if name_match:
            info['gpu_name'] = name_match.group(1).strip()
        cu_match = re.search(r'^\s*Compute Unit:\s+(\d+)',
                             output, re.MULTILINE)
        if cu_match:
            info['compute_units'] = int(cu_match.group(1))
        wf_match = re.search(
            r'^\s*Wavefront Size:\s+(\d+)', output, re.MULTILINE)
        if wf_match:
            info['wavefront_size'] = int(wf_match.group(1))
        return info

    def _get_mock_hardware_info(self) -> Dict:
        """Mock hardware info for MI300X"""
        return {
            'gpu_name': 'AMD MI300X',
            'compute_units': 120,
            'memory_size_gb': 192,
            'memory_bandwidth_tb_s': 5.3,
            'wavefront_size': 64,
            'l2_cache_size_kb': 16384,
            'l1_cache_size_kb': 128
        }
