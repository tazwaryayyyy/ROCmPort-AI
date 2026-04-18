# Judge Mode Walkthrough

Use this sequence during technical evaluation.

## Goal
Make every claim falsifiable and easy to verify.

## Flow
1. Show raw CUDA input.
2. Run baseline translation only (straight hipify output).
3. Show baseline compile/profiler result.
4. Run full ROCmPort AI loop.
5. Show each agent event and decisions.
6. Compare final output against the declared baseline.
7. Show one weak result (small gain or no gain) and explain why.

## Baseline Policy
- Primary baseline: straight hipify output with minimal required compile edits.
- Never switch baselines mid-demo.
- Repeat baseline definition before showing speedup.

## Required Artifacts
- CUDA source.
- Baseline HIP output.
- Optimized HIP output.
- Compile logs.
- Profiler summary.
- Final report with rationale.

## Suggested Script
- "Here is the original CUDA kernel."
- "Here is baseline HIP produced by hipify only."
- "Now we run the orchestration loop and show each decision."
- "This is the final code diff and measured result versus baseline."
- "Here is a case where gain is limited, and why."

## Pass/Fail Criteria
A demo is credible if:
- Baseline is explicit.
- Intermediate artifacts are visible.
- At least one non-win case is included.
- Reasoning matches observed profiler data.
