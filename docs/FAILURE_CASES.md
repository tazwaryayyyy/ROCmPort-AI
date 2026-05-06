# Failure Cases

This document records known failure modes with reproducible context.

## FC-001: Inline PTX in CUDA Kernel

### Why this matters
Kernels that embed inline PTX are a realistic migration boundary. hipify can translate CUDA APIs, but it cannot preserve NVIDIA-specific assembly semantics on AMD.

### Original CUDA pattern (simplified)
```cpp
__device__ __forceinline__ unsigned lane_id() {
  unsigned lane;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  return lane;
}
```

### Typical migration output
- CUDA runtime calls are translated.
- Inline PTX block is left unchanged or translated into invalid code for HIP compilation.

### Observed failure mode
- Compile error under hipcc due to unsupported PTX instruction syntax.
- In some cases, compile succeeds after manual edits but semantics differ because lane behavior assumptions are NVIDIA-specific.

### Root cause
- Inline PTX is vendor-specific and outside mechanical translation scope.
- Warp-level assumptions in PTX often rely on 32-lane behavior and NVIDIA ISA details.

### What is required to fix
1. Replace inline PTX with HIP or portable intrinsics.
2. Rework lane-level logic for wavefront-64 behavior where required.
3. Add correctness tests for edge lanes and reduction boundaries.
4. Re-profile after rewrite to confirm no occupancy regressions.

### Trust note
This is a deliberate example of where ROCmPort AI should report risk, not pretend full automation.

## Failure Case: Library-Heavy CUDA Code (CUB, Thrust, cuDNN)

**Input type**: CUDA kernels that call into CUB, Thrust, or cuDNN directly

**Example pattern**:
```cpp
#include <cub/cub.cuh>
cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
```

**What happens**: hipify-clang renames the include to `<hipcub/hipcub.hpp>` and the namespace to `hipcub`. ROCmPort AI passes this through. The translation is mechanically correct.

**The limitation**: hipCUB API coverage is not 1:1 with CUB. Some primitives behave differently under ROCm, and performance characteristics differ significantly due to wavefront width. ROCmPort AI does not currently benchmark library calls against rocPRIM equivalents.

**What ROCmPort AI does**: flags the library dependency in the static scan, marks it HIGH risk, and recommends manual review by a ROCm-experienced engineer.

**What ROCmPort AI does not do**: guarantee correctness or performance parity for library-heavy code without human validation.

**Fix requirement**: Manual comparison of CUB vs hipCUB primitive behavior for the specific use case, or replacement with rocPRIM equivalents.

## Failure Case: Flash Attention — Warp Shuffle Intrinsics

**Kernel**: Simplified Flash Attention forward pass (Dao et al. 2022 style)
**File**: backend/demo_kernels/flash_attention_simplified.cu

**Bugs detected by ROCmPort AI static scan**:
- `__shfl_down` with implicit warp-32 offset=16 — on AMD wavefront-64, 
  the final reduction should use offset=32 first
- Softmax reduction terminates at 16 lanes — silently wrong on gfx942

**What hipify does**: renames cudaFree to hipFree, cuda headers to hip headers. 
Does NOT fix the shuffle semantics.

**What ROCmPort AI does**: flags both shuffle calls as HIGH risk, 
identifies the offset=16 assumption, suggests wavefront-64 aware rewrite.

**Status**: Compiled and executed on AMD Instinct MI300X (gfx942), ROCm 7.2.
Numerical correctness not verified — requires reference CPU implementation.

**Fix required**: Replace `__shfl_down(x, 16)` with two-stage reduction:
  `__shfl_down(x, 32)` then `__shfl_down(x, 16)` for wavefront-64.
