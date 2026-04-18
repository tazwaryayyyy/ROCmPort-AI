# ROCmPort AI Benchmarking Guide

This document defines how to report performance without overclaiming.

## Reporting Principles

- Compare against a clearly stated baseline.
- Use reproducible runs with fixed input sizes and environment details.
- Include correctness checks before accepting performance numbers.
- Report failures and non-improving cases, not only wins.

## Baseline Definitions

Use one of these and name it explicitly in each table:

- Baseline A: Straight `hipify-clang` output with minimal manual edits.
- Baseline B: Existing hand-written HIP version from the team.

Recommended: use Baseline A for measuring migration automation value.

Quick answer format for live review:

- Q: What is your baseline?
- A: Straight hipify output with minimal compile edits (Baseline A), measured on the same hardware and inputs.

## Required Environment Metadata

Always include:

- GPU model (for example MI300X) and memory size.
- ROCm version, compiler version, and profiler version.
- OS and driver versions.
- Kernel launch parameters and input sizes.
- Number of runs and aggregation rule (median recommended).

## Required Measurement Fields

For each kernel tested, provide:

- Kernel name and workload shape.
- Baseline latency.
- Optimized latency.
- Speedup ratio.
- Correctness status (pass/fail and checksum or tolerance).
- Notes on optimization strategy.

Example table format:

| Kernel | Shape | Baseline (ms) | Optimized (ms) | Speedup | Correctness | Notes |
|---|---|---:|---:|---:|---|---|
| matrix_multiply | 1024x1024 | 12.4 | 9.5 | 1.31x | pass | LDS tiling + wavefront-aware launch |

Include non-win cases in the same table. Example:

| Kernel | Shape | Baseline (ms) | Optimized (ms) | Speedup | Correctness | Notes |
|---|---|---:|---:|---:|---|---|
| sparse_scatter | 4M elements | 6.0 | 6.3 | 0.95x | pass | Irregular access pattern; optimization did not help |

## Reproducibility Checklist

Before publishing numbers, verify all items:

- Same input set for baseline and optimized runs.
- Warm-up runs excluded or consistently handled.
- At least 3 measured runs (prefer 5+) with median reported.
- No hidden manual edits after optimization output unless documented.
- Full command lines and profiler artifacts retained.

## Evidence Package for Review

A technical review package should include:

- CUDA source input.
- Baseline HIP output.
- Optimized HIP output.
- Compile logs and profiler summaries.
- Final report explaining what changed and why.

## Interpreting Results Responsibly

- Some kernels will regress or fail initially; this is normal for migration.
- Improvement ranges vary by memory behavior, occupancy, and control-flow patterns.
- Do not claim universal speedups.

Preferred claim style:

"ROCmPort AI improved X out of Y tested kernels against a stated baseline under reproducible MI300X conditions."

## Current Repository Status

The repository includes demo kernels intended to exercise migration behavior.
Treat any sample numbers as demonstrations unless accompanied by full reproducibility artifacts from your environment.
