# Live Results — AMD Instinct MI300X

Measurements taken with `rocprof` on AMD Instinct MI300X (gfx942),
ROCm 7.0, AMD Developer Cloud, **May 8 2026**.

Raw profiler CSV files are in [`docs/benchmark_runs/`](benchmark_runs/).
Values in `backend/tools/demo_artifacts.py` are labelled `data_source="mi300x_live"`
and match these CSV files exactly.

## Benchmark Results

| Kernel | Baseline (ms) | Optimized (ms) | Speedup | Bandwidth | Source CSV |
|--------|---------------|----------------|---------|-----------|------------|
| matrix_multiply (512×512) | 0.076 | 0.026 | 2.91x | — | [matmul_out.stats.csv](benchmark_runs/matmul_out.stats.csv) |
| vector_add (32M elements) | — | 0.098 | — | 3,918 GB/s | [vecadd_out.stats.csv](benchmark_runs/vecadd_out.stats.csv) |
| reduction (16M elements) | — | 0.042 | — | — | [reduction.stats.csv](benchmark_runs/reduction.stats.csv) |
| convolution_2d | 211.7 | 158.3 | 1.34x | 2,134.8 GB/s | demo_artifact (not yet measured) |

> **Note:** vector_add and reduction were run standalone; no pre-optimisation baseline
> was captured in these runs. matrix_multiply baseline is `matmul_baseline` (hipify
> output, no tiling); optimized is `matmul_tiled` (LDS 32×32 tile, wavefront-64 aligned).

## rocprof Run Details

- **Hardware:** AMD Instinct MI300X, gfx942
- **ROCm version:** 7.0
- **Platform:** AMD Developer Cloud
- **Date:** May 8 2026
- **Profiler:** `rocprof --stats`
- **Wavefront size:** 64
- **HBM3:** 192 GB, 5.3 TB/s theoretical

## Raw CSV Quick-Reference

**matmul_out.stats.csv** (`docs/benchmark_runs/matmul_out.stats.csv`)
- `matmul_baseline`: 5 calls, total 379,467 ns, avg **75,893 ns (0.076 ms)**
- `matmul_tiled`: 5 calls, total 130,618 ns, avg **26,123 ns (0.026 ms)** → **2.91× speedup**

**vecadd_out.stats.csv** (`docs/benchmark_runs/vecadd_out.stats.csv`)
- `vector_add`: 10 calls, total 976,466 ns, avg **97,646 ns (0.098 ms)** → **3,918 GB/s**

**reduction.stats.csv** (`docs/benchmark_runs/reduction.stats.csv`)
- `reduction`: 10 calls, total 424,248 ns, avg **42,424 ns (0.042 ms)**
