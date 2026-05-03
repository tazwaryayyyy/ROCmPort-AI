# ROCmPort AI

ROCmPort AI helps CUDA teams migrate to AMD by translating, testing, and iteratively optimizing kernels using real hardware feedback.

It is an acceleration system for migration work, not a one-click replacement for CUDA expertise.

## Live Demo

- https://rocmport-ai.onrender.com

## What This Project Is

ROCmPort AI orchestrates a migration loop:

1. Analyze CUDA code and detect migration risks.
2. Translate with hipify plus LLM-assisted fixes.
3. Compile and profile with ROCm tooling.
4. Propose optimization changes and re-test.
5. Return artifacts and decision trace.

## What This Project Is Not

- Not guaranteed to auto-fix all CUDA kernels.
- Not a claim that every kernel improves.
- Not a replacement for domain experts in performance-critical code.

Complex kernels can fail conversion due to architecture assumptions, undefined behavior, inline PTX, or handcrafted memory logic. The value is reduced migration time and faster debug loops.

## Target User and Business Case

Primary product position:
- Tool for teams evaluating AMD migration cost and performance tradeoffs.

Typical use cases:
- Port legacy CUDA modules to HIP/ROCm with a measurable baseline.
- Build a migration backlog ranked by risk and expected impact.
- Identify kernels where MI300X memory capacity can remove sharding complexity.

Cost and performance impact should be calculated from your environment and workload, not fixed marketing ranges.

## AMD-Specific Technical Considerations (MI300X)

ROCmPort AI explicitly reasons about AMD constraints and opportunities, including:

- Wavefront size 64 (vs CUDA warp 32 assumptions), which affects reduction trees, ballot/shuffle idioms, and launch geometry.
- LDS (local data store) usage and bank behavior for tile staging and reuse.
- MI300X memory capacity (192GB HBM) and implications for reducing model/data sharding in some workflows.
- Memory access patterns and occupancy tradeoffs under ROCm compiler behavior.

These are the places where migration often breaks or underperforms even after a successful hipify pass.

### Concrete Wavefront Mismatch Example

From `backend/demo_kernels/reduction.cu`, the reduction tail assumes a 32-thread warp:

```cpp
// NVIDIA-style assumption (incorrect on AMD wavefront=64)
if (tid < 32) {
	volatile float* vsmem = sdata;
	vsmem[tid] += vsmem[tid + 32];
	vsmem[tid] += vsmem[tid + 16];
	...
}
```

A wavefront-aware correction expands the final stage to include the 64-wide lane behavior:

```cpp
// AMD-aware final reduction stage
if (tid < 64) {
	volatile float* vsmem = sdata;
	vsmem[tid] += vsmem[tid + 32];
	if (tid < 32) {
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}
}
```

The key point is not the exact rewrite shape; it is that warp-size assumptions must be made explicit and re-validated on AMD.

## Why This Is More Than Glue

ROCmPort AI combines existing tools, but its core value is the control system around them:

- Decision loop: detect failure/perf regressions, apply next strategy, re-run.
- Explainability: stream each step and rationale (SSE logs + final report).
- Verification: pair code changes with compile/test/profiler evidence.

## Judge Mode Walkthrough

Use this flow for technical review:

1. Show original CUDA kernel.
2. Show baseline HIP from straight hipify output.
3. Run ROCmPort AI and show per-agent trace.
4. Show final optimized HIP output.
5. Show measured result against the declared baseline.
6. Show one case with marginal gain or no gain.

This format makes the comparison falsifiable and avoids curated-demo concerns.

- Full walkthrough: `docs/JUDGE_MODE.md`.

## Documented Failure Case

At least one failure path is documented with source, output, root cause, and fix requirements:

- See `docs/FAILURE_CASES.md`.

This is intentional: credibility improves when the system's failure boundary is visible.

## Quick Start

### Option 1: Startup Script

```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

### Option 2: Manual

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in a browser.

### Option 3: Docker

```bash
docker build -t rocmport-ai .
docker run -p 8000:8000 rocmport-ai
```

## Benchmarking and Reproducibility

Benchmark claims should always include:

- Baseline definition (e.g., straight hipify output).
- Hardware/software versions.
- Input sizes and run counts.
- Correctness verification.
- Full logs or scripts to reproduce.

See `BENCHMARKS.md` for the recommended reporting format used by this repository.

## Project Structure

```text
ROCmPort AI/
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── agents/
│   │   ├── analyzer.py
│   │   ├── translator.py
│   │   ├── optimizer.py
│   │   ├── tester.py
│   │   └── coordinator.py
│   ├── tools/
│   │   ├── hipify_wrapper.py
│   │   ├── rocprof_wrapper.py
│   │   └── llm_client.py
│   ├── demo_kernels/
│   └── prompts/
├── frontend/
│   └── index.html
├── BENCHMARKS.md
└── README.md
```

## Configuration

Copy `.env.example` to `.env`:

```bash
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile

USE_VLLM=true
VLLM_BASE_URL=http://your-amd-cloud:8000
VLLM_API_KEY=your_vllm_key
VLLM_MODEL=amd/llama-3.3-70b

ROCM_AVAILABLE=true
HIPCC_PATH=hipcc
ROCPROF_PATH=rocprof
```

## Defensible Scope

This project is harder to replicate than a thin wrapper because it couples:

- Multi-agent orchestration with retry decisions.
- Structured traceability across analysis, translation, optimization, and test phases.
- Integrated reporting where claims can be audited against intermediate artifacts.

A basic weekend clone can chain hipify and an LLM. The differentiator is reliable decision flow and evidence quality under failure.

## Troubleshooting

| Issue | Resolution |
|---|---|
| `GROQ_API_KEY not found` | Add key to `.env`. |
| `hipcc not found` | Install ROCm toolchain or run in an ROCm-enabled environment. |
| Backend unavailable | Verify FastAPI server is running on port `8000`. |
| No improvement observed | Re-check baseline definition, kernel size, and profiler counters. |

## License

See `LICENSE`.

## ✅ Live Results on AMD Instinct MI300X

All demo kernels migrated, compiled, and profiled on real MI300X hardware (AMD DevCloud, ROCm 7.2, gfx942).

| Kernel | Total Changes | Critical AMD Bugs Found | Status |
|--------|--------------|------------------------|--------|
| reduction | 9 | warp-32 final stage (silent wrong results) | ✅ Compiled |
| vector_add | 7 | threadIdx%32 wavefront mismatch | ✅ Compiled |
| matrix_multiply | 11 | warp-32 + LDS bank conflicts | ✅ Compiled |
| convolution_2d | 13 | warp-32 + LDS padding | ✅ Compiled |

`data_source: real_rocm` — verified on AMD DevCloud MI300X instance.
