# Judge Mode Walkthrough

Use this sequence during technical evaluation with the current React UI and
FastAPI SSE stream.

## Goal

Make every claim falsifiable and tied to fields returned by the backend.

## Flow

1. Open `http://localhost:8000/index.html`.
2. Choose or paste a CUDA kernel.
3. Run ROCmPort AI and watch the five agent cards:
   analyzer, translator, optimizer, tester, coordinator.
4. Confirm the tester event reports speedup, bandwidth, bottleneck, and data source.
5. Confirm the coordinator event produces the final report JSON in its SSE `detail`.
6. Use `/benchmark-report` for reproducible demo-artifact metrics and data-source labels.
7. Show a limited-gain case such as `vector_add` and explain the bandwidth-bound result.

## Baseline Policy

- Primary baseline: straight hipify output with minimal required compile edits.
- Demo-mode baselines come from `backend/tools/demo_artifacts.py`.
- Real hardware baselines require `ROCM_AVAILABLE=true` and captured `hipcc`/`rocprof` logs.
- Never mix `demo_artifact` and `real_rocm` numbers in the same result table.

## Visible Artifacts In Current UI

- CUDA source input.
- Agent event stream.
- Tester summary: execution time, bandwidth utilization, bottleneck, notes.
- Final summary footer: changes made, critical bugs found, compile/migration success, data source.

## Additional Artifacts Available By API

- `/benchmark-report`: reproducible benchmark summary and static risk scans.
- `/export`: migration diff, original CUDA, optimized HIP, and report markdown.
- `/demo-kernels`: source for bundled demo kernels.

## Pass/Fail Criteria

A demo is credible if:

- Every speedup is tied to its `data_source`.
- The baseline definition is stated before showing speedup.
- Static risk findings match the analyzer event or `/benchmark-report`.
- At least one non-perfect or limited-gain case is included.
