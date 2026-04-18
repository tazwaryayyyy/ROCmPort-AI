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
