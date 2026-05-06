# Live Results — AMD Instinct MI300X (gfx942), ROCm 7.2

All kernels compiled with `hipcc --offload-arch=gfx942 -O3` and 
benchmarked on real AMD DevCloud hardware. No simulated data.

## Benchmark Results

| Kernel | Input Size | Baseline HIP (ms) | Optimized HIP (ms) | Speedup | Notes |
|--------|------------|-------------------|-------------------|---------|-------|
| matrix_multiply | 512x512 fp32 | 0.068 | 0.026 | **2.61x** | Shared memory tiling |
| reduction | 16M elements fp32 | — | 0.019 | — | Wavefront-64 fix verified PASS |
| vector_add | 32M elements fp32 | — | 0.099 | — | 4077.6 GB/s (77% MI300X peak) |

## Hardware Configuration

- **GPU**: AMD Instinct MI300X VF (gfx942)
- **VRAM**: 192GB HBM3
- **Platform**: AMD Developer Cloud (ATL1 region)
- **ROCm**: 7.2
- **Compiler**: hipcc (clang++ --offload-arch=gfx942)
- **data_source**: real_rocm

## Key Findings

**matrix_multiply**: Shared memory tiling with LDS padding ([32][33] 
to avoid bank conflicts) delivers 2.61x over naive global memory access 
on gfx942. The wavefront-64 aligned block size (256 threads) is critical 
for this result.

**reduction**: AMD wavefront-64 aware final stage produces correct results. 
The original CUDA kernel with hardcoded warp-32 assumption silently skips 
lanes 32-63 and returns a wrong sum. ROCmPort AI catches this at static 
scan before any compilation attempt.

**vector_add**: 4077.6 GB/s achieved on a memory-bound kernel — 77% of 
MI300X's 5.3 TB/s theoretical HBM3 peak. This demonstrates the bandwidth 
advantage of MI300X over H100 (3.35 TB/s peak) for memory-bound workloads.

## Correctness Verification
All kernels executed without runtime errors on gfx942.
