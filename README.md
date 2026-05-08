# ⚡ ROCmPort AI

A multi-agent pipeline that migrates CUDA kernels to AMD ROCm/HIP — catching the bugs that `hipify` misses, compiling with `hipcc`, profiling with `rocprof` on real MI300X hardware, and iterating until the output is correct and fast.

---

## Live Demo

- **Backend API**: https://rocmport-ai.onrender.com
- **HuggingFace Space**: https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/ROCmPort-AI

---

## The Gap hipify Doesn't Close

`hipify-clang` translates CUDA API calls mechanically. It cannot detect that `if (tid < 32)` in a warp reduction silently skips lanes 32–63 on AMD wavefront-64. The code compiles. The output is wrong. No errors. No warnings.

**ROCmPort AI catches this before execution.**

```cpp
// NVIDIA assumption — silently wrong on AMD (wavefront = 64)
if (tid < 32) {
    vsmem[tid] += vsmem[tid + 32];  // lanes 32-63 never participate
    ...
}

// AMD-aware correction
if (tid < 64) {
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

---

## What ROCmPort AI Does

1. **Analyze** — scan CUDA kernel for AMD-specific risks (wavefront size, ballot/shuffle idioms, shared memory layout)
2. **Translate** — run hipify + LLM-assisted fixes for bugs hipify can't detect
3. **Compile** — build with `hipcc` targeting gfx942, surface real errors
4. **Profile** — run `rocprof` and measure actual throughput on MI300X
5. **Optimize** — propose changes based on profiler feedback, re-test
6. **Report** — stream full decision trace with per-agent rationale

If the optimized output underperforms the baseline, the coordinator retries the optimizer (max 3 iterations) before returning the best result found.

---

## Reproducible Demo Results

These numbers are deterministic `demo_artifact` values returned by the backend when `ROCM_AVAILABLE=false`. Set `ROCM_AVAILABLE=true` on real MI300X hardware to collect `data_source=real_rocm` results.

| Kernel | Input | Baseline HIP | Optimized HIP | Result |
|--------|-------|-------------|---------------|--------|
| matrix_multiply | demo artifact | 121.4ms | 89.1ms | **1.36x speedup** |
| reduction | demo artifact | 88.2ms | 68.7ms | **1.28x speedup** |
| vector_add | demo artifact | 45.1ms | 38.2ms | **1.18x speedup** |

Hardware class: AMD Instinct MI300X, 192GB HBM3, wavefront=64

---

## The Dataset No One Else Built

**170 expert-curated CUDA→ROCm correctness bugs** across 6 categories. Every example includes the original CUDA, the still-broken `hipify` output, and the correct AMD version — with a precise explanation of why the bug manifests on gfx942.

| Category | Count | Description |
|----------|-------|-------------|
| `warp_size_hardcoded_32` | 50 | `tid & 31`, `tid >> 5`, loop bounds |
| `threadidx_modulo_warpsize` | 30 | `threadIdx.x % 32` for lane ID |
| `shared_memory_no_padding` | 30 | Arrays sized for 32-thread warps |
| `reduction_loop_bound_32` | 20 | Shuffle loops missing offset=32 step |
| `ballot_sync_warp_assumptions` | 20 | `uint32_t` truncating 64-bit ballot |
| `shfl_down_sync_mask_assumptions` | 20 | 32-bit mask on 64-lane wavefront |

📦 **[tazwarrrr/cuda-to-rocm-wavefront-bugs](https://huggingface.co/datasets/tazwarrrr/cuda-to-rocm-wavefront-bugs)**

---

## The Model Trained on AMD Hardware

Qwen2.5-Coder-7B-Instruct fine-tuned with LoRA (r=16) on the wavefront bug dataset — trained on an AMD Instinct MI300X via AMD Developer Cloud in 79 seconds. Final loss: 1.189, token accuracy: 81%.

🤖 **[tazwarrrr/rocmport-qwen-wavefront-finetuned](https://huggingface.co/tazwarrrr/rocmport-qwen-wavefront-finetuned)**

---

## Agent Architecture

```
CUDA Input
    │
    ▼
┌─────────────┐
│   Analyzer  │  Detect wavefront bugs, classify risk
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Translator │  hipify + LLM fix for missed bugs
└──────┬──────┘
       │
       ▼
┌─────────────┐     speedup < 0.95?
│  Optimizer  │ ◄──────────────────┐
└──────┬──────┘                    │
       │                           │
       ▼                           │
┌─────────────┐     retry (max 3)  │
│   Tester    │ ───────────────────┘
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Coordinator │  Final report + artifacts
└─────────────┘
```

| Agent | Model | Role |
|-------|-------|------|
| Analyzer | Qwen2.5-Coder-32B | Code risk analysis |
| Translator | Qwen2.5-Coder-32B | CUDA→HIP translation |
| Optimizer | Qwen2.5-Coder-32B | Hardware-aware optimization |
| Tester | Llama-3.3-70B | Log parsing, compile verification |

---

## AMD-Specific Technical Considerations

ROCmPort AI reasons explicitly about MI300X constraints:

- **Wavefront size 64** — affects reduction trees, ballot/shuffle idioms, launch geometry
- **LDS bank behavior** — tile staging and reuse patterns
- **192GB HBM3** — opportunities to eliminate model sharding in some workflows
- **gfx942 occupancy** — memory access pattern tradeoffs under ROCm compiler

---

## Why This Is Hard to Replicate

A basic clone can chain `hipify` and an LLM. The differentiator is:

- **Decision loop** — detect failure/perf regression, apply next strategy, re-run
- **Explainability** — stream each agent's reasoning via SSE in real time
- **Verification** — every code change paired with compile + profiler evidence
- **Dataset** — 170 labeled correctness bugs that don't exist anywhere else
- **Fine-tuned model** — trained on real AMD hardware on a purpose-built dataset

---

## Quick Start

```bash
# Windows
start.bat

# Linux/Mac
./start.sh

# Manual
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
. .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
# Add GROQ_API_KEY
npm --prefix frontend install
npm --prefix frontend run build
python -m uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000/index.html` in a browser.

### Docker

```bash
docker build -t rocmport-ai .
docker run -p 8000:8000 rocmport-ai
```

---

## Configuration

```bash
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile

# AMD DevCloud vLLM (production)
USE_VLLM=true
VLLM_BASE_URL=http://your-amd-cloud:8000
VLLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
ROCM_AVAILABLE=true
```

---

## Documented Failure Cases

At least one failure path is documented with source, output, root cause, and fix requirements. See [`docs/FAILURE_CASES.md`](docs/FAILURE_CASES.md).

Credibility improves when the system's failure boundary is visible.

---

## Judge Mode

For technical review, use this flow:

1. Show original CUDA kernel
2. Show baseline HIP from straight `hipify` output
3. Run ROCmPort AI — watch per-agent trace stream
4. Show final optimized HIP output
5. Show measured result vs declared baseline
6. Show one case with marginal gain or no gain

Full walkthrough: [`docs/JUDGE_MODE.md`](docs/JUDGE_MODE.md)

---

## Project Structure

```
ROCmPort AI/
├── backend/
│   ├── agents/          # analyzer, translator, optimizer, tester, coordinator
│   ├── tools/           # hipify_wrapper, rocprof_wrapper, llm_client
│   ├── demo_kernels/    # reduction.cu, matrix_multiply.cu, vector_add.cu
│   └── graph/           # LangGraph StateGraph pipeline
├── dataset/
│   ├── upload_dataset.py
│   └── finetune_qwen.py
├── docs/
│   ├── LIVE_RESULTS.md
│   ├── FAILURE_CASES.md
│   └── JUDGE_MODE.md
├── frontend/
└── BENCHMARKS.md
```

---

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `GROQ_API_KEY not found` | Add key to `.env` |
| `hipcc not found` | Install ROCm toolchain or use ROCm-enabled environment |
| Backend unavailable | Verify FastAPI running on port 8000 |
| No improvement observed | Check baseline definition and profiler counters |

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE)
