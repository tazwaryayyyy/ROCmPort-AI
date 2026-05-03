# Live Results — AMD Instinct MI300X (gfx942), ROCm 7.2

All kernels migrated and compiled successfully on real MI300X hardware.

| Kernel | CUDA Changes | LLM Fixes | Critical Bugs Found | Compiled on MI300X |
|--------|-------------|-----------|--------------------|--------------------|
| reduction | 7 hipify | 2 LLM | warp-32 final stage (silent wrong results on AMD) | ✅ |
| vector_add | 5 hipify | 2 LLM | threadIdx%32 wavefront mismatch | ✅ |
| matrix_multiply | 10 hipify | 1 LLM | warp-32 + LDS bank conflicts | ✅ |
| convolution_2d | 10 hipify | 3 LLM | warp-32 + LDS padding | ✅ |

Hardware: AMD Instinct MI300X VF (gfx942), 192GB HBM3
Software: ROCm 7.2, hipcc, rocprof
data_source: real_rocm (not mock)
