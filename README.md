# ROCmPort AI

ROCmPort AI helps CUDA teams migrate to AMD by translating, testing, and iteratively optimizing kernels using real hardware feedback.

It is an acceleration system for migration work, not a one-click replacement for CUDA expertise.

## Live Demo

-  **Backend Demo**: https://rocmport-ai.onrender.com
-  **HuggingFace Space**: https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/ROCmPort-AI

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

## Fine-tuned Model

[tazwarrrr/rocmport-qwen-wavefront-finetuned](https://huggingface.co/tazwarrrr/rocmport-qwen-wavefront-finetuned)  
Qwen2.5-Coder-7B-Instruct fine-tuned on 153 CUDA→ROCm wavefront-64 bug examples.  
Trained on AMD Instinct MI300X (gfx942) with ROCm 6.2 in 79 seconds.

## Dataset

Training data: [tazwarrrr/cuda-to-rocm-wavefront-bugs](https://huggingface.co/datasets/tazwarrrr/cuda-to-rocm-wavefront-bugs)  
170 expert-curated CUDA→ROCm wavefront-64 bug examples across 6 categories.
See `dataset/upload_dataset.py` for how it was compiled and uploaded.

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

## LLM Configuration

| Agent | Model | Why |
|-------|-------|-----|
| Analyzer | Qwen2.5-Coder-32B | Purpose-built for code reasoning |
| Translator | Qwen2.5-Coder-32B | Best-in-class CUDA/HIP translation |
| Optimizer | Qwen2.5-Coder-32B | Hardware-aware optimization proposals |
| Tester | llama-3.3-70b | Fast log parsing, cost-efficient fallback |

**Primary**: Qwen2.5-Coder-32B via HuggingFace Inference API  
**Production**: Qwen2.5-Coder-32B via vLLM on AMD MI300X DevCloud  
**Estimated cost**: ~$0.003 per kernel migration (8K tokens avg)

---

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


## Why Not Just Use hipify?

hipify-clang is AMD's official translation tool. ROCmPort AI uses it as a first pass. The problem is what hipify cannot catch.

**The reduction kernel example:**

hipify successfully translates `reduction.cu` — it compiles, it runs, it returns a result. No errors. But the result is silently wrong on AMD hardware.

The root cause: line 59 assumes `warpSize=32` in the final unrolled reduction stage. On AMD, wavefront size is 64. Lanes 32–63 are skipped entirely in the final summation. The output looks plausible but is numerically incorrect.

hipify has no knowledge of this. It performs mechanical API renaming. It cannot reason about hardware architecture assumptions baked into kernel logic.

ROCmPort AI catches this before execution:

- Static scanner flags line 59 as CRITICAL risk: "hardcoded warp-32 conditional — assumes NVIDIA warpSize=32. On AMD wavefront=64 this silently skips lanes 32–63"
- LLM correction pass rewrites the final reduction stage to be wavefront-64 aware
- Compiler + rocprof verification confirms the fix compiles and executes correctly on gfx942

This is the gap between "it compiles" and "it is correct."

## ✅ Live Results on AMD Instinct MI300X

All demo kernels migrated, compiled, and profiled on real MI300X hardware (AMD DevCloud, ROCm 7.2, gfx942).

| Kernel | Input | Baseline HIP | Optimized HIP | Speedup |
|--------|-------|-------------|---------------|---------|
| matrix_multiply | 512x512 fp32 | 0.068ms | 0.026ms | 2.61x |
| reduction | 16M elements | — | 0.019ms | PASS (correct) |
| vector_add | 32M elements | — | 0.099ms | 4077.6 GB/s |

Hardware: AMD Instinct MI300X VF (gfx942), 192GB HBM3, ROCm 7.2

## License

See `LICENSE`.