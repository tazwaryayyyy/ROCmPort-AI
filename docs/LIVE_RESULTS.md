# Reproducible Results

The backend returns deterministic benchmark artifacts unless `ROCM_AVAILABLE=true`
is set on real ROCm hardware. These values come from
`backend/tools/demo_artifacts.py` and are labelled `data_source="demo_artifact"`
in API responses.

## Benchmark Results

| Kernel | Baseline HIP (ms) | Optimized HIP (ms) | Speedup | Bandwidth | Bottleneck |
|--------|-------------------|--------------------|---------|-----------|------------|
| matrix_multiply | 121.4 | 89.1 | 1.36x | 1843.7 GB/s | memory-bound |
| reduction | 88.2 | 68.7 | 1.28x | 531.8 GB/s | compute-bound after wavefront fix |
| vector_add | 45.1 | 38.2 | 1.18x | 4821.6 GB/s | memory-bound |
| convolution_2d | 211.7 | 158.3 | 1.34x | 2134.8 GB/s | memory-bound |

## Hardware Context

- GPU class: AMD Instinct MI300X
- VRAM: 192GB HBM3
- Theoretical memory bandwidth: 5.3 TB/s
- Wavefront size: 64
- API data source in local/demo mode: `demo_artifact`

## Real Hardware Mode

Set `ROCM_AVAILABLE=true`, `HIPCC_PATH=hipcc`, and `ROCPROF_PATH=rocprof` on a
real MI300X ROCm environment to replace demo artifacts with `data_source="real_rocm"`.
Real run output should be captured separately with the exact ROCm version, kernel
input size, compiler flags, and profiler logs.
