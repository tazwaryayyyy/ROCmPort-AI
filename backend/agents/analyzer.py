import json
import re
from models import AnalyzerResult, WorkloadType
from tools.llm_client import LLMClient
from tools.json_utils import safe_json_loads

llm_client = LLMClient()

def chat_complete(messages: list, temperature: float = 0.7, max_tokens: int = 4000) -> str:
    """Wrapper for LLM client chat completion"""
    return llm_client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)

def generate_prediction(workload_type: WorkloadType, line_count: int) -> str:
    """Generate performance prediction based on workload analysis"""
    if workload_type == WorkloadType.MEMORY_BOUND:
        return "🧠 Prediction: This kernel is memory-bound → HIGH potential gain on MI300X (5.3 TB/s vs H100 3.35 TB/s bandwidth)"
    elif workload_type == WorkloadType.COMPUTE_BOUND:
        return "🧠 Prediction: This kernel is compute-bound → MODERATE gain on MI300X (wavefront efficiency improvements)"
    else:
        return "🧠 Prediction: Unknown workload type → LIMITED gain prediction without further analysis"

SYSTEM_PROMPT = """You are an expert CUDA and GPU architecture engineer analyzing CUDA code before porting it to AMD ROCm/HIP.

Your job is to deeply analyze CUDA code and output a structured JSON analysis. Be specific and technical.

CRITICAL things to detect:
1. All CUDA kernel functions (__global__ functions)
2. All CUDA API calls (cudaMalloc, cudaMemcpy, cudaFree, etc.)
3. Warp size assumptions - NVIDIA warp = 32, AMD wavefront = 64. This causes SILENT BUGS.
   Look for: warpSize, __shfl_*, __ballot_sync, hardcoded 32 in thread calculations, WARP_SIZE defines
4. Workload type classification:
   - memory-bound: lots of global memory reads/writes, low arithmetic intensity
   - compute-bound: lots of math operations, high reuse of loaded data
5. Multi-GPU sharding code (written for NVIDIA's 80GB limit - unnecessary on MI300X 192GB)
6. Porting difficulty
7. Code complexity estimation (line count, nested loops, memory access patterns)

Respond ONLY with this exact JSON structure, no markdown, no extra text:
{
  "kernels_found": ["kernel1", "kernel2"],
  "cuda_apis": ["cudaMalloc", "cudaMemcpy"],
  "warp_size_issue": true,
  "warp_size_detail": "Line 23: hardcoded warpSize=32 in block reduction. AMD wavefront=64 -- this will produce incorrect results.",
  "workload_type": "memory-bound",
  "sharding_detected": false,
  "difficulty": "Medium",
  "difficulty_reason": "Warp-level primitives require manual rewriting beyond hipify scope",
  "line_count": 150,
  "complexity_score": 7
}"""


def run(cuda_code: str) -> AnalyzerResult:
    # Count lines for complexity estimation
    line_count = len([line for line in cuda_code.split('\n') if line.strip()])
    
    try:
        raw = chat_complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this CUDA code:\n\n```cuda\n{cuda_code}\n```"}
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        data = safe_json_loads(raw)
    except Exception:
        # Fallback to defaults on LLM/parse failure
        data = {
            "kernels_found": ["unknown_kernel"],
            "cuda_apis": [],
            "warp_size_issue": False,
            "workload_type": "memory-bound",
            "sharding_detected": False,
            "difficulty": "Medium",
            "difficulty_reason": "Analysis failed, using safe defaults",
            "line_count": line_count,
            "complexity_score": 5
        }
    
    workload_type = WorkloadType(data.get("workload_type", "unknown"))
    prediction = generate_prediction(workload_type, line_count)

    return AnalyzerResult(
        kernels_found=data.get("kernels_found", []),
        cuda_apis=data.get("cuda_apis", []),
        warp_size_issue=data.get("warp_size_issue", False),
        warp_size_detail=data.get("warp_size_detail"),
        workload_type=workload_type,
        sharding_detected=data.get("sharding_detected", False),
        difficulty=data.get("difficulty", "Medium"),
        difficulty_reason=data.get("difficulty_reason", ""),
        prediction=prediction,
        line_count=data.get("line_count", line_count),
        complexity_score=data.get("complexity_score", 5)
    )
