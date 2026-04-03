import json
import re
from models import OptimizerResult, AnalyzerResult, WorkloadType
from tools.llm_client import LLMClient

llm_client = LLMClient()

def chat_complete(messages: list) -> str:
    """Wrapper for LLM client chat completion"""
    return llm_client.chat_completion(messages)

ALLOWED_OPTIMIZATIONS = """
You may ONLY suggest these specific, well-known AMD MI300X optimizations:
1. Shared memory tiling: Replace naive global memory access with 32x32 shared memory tiles (__shared__)
2. Block size adjustment: Change thread block size to 256 for MI300X wavefront alignment (multiple of 64)
3. Memory coalescing: Fix non-coalesced global memory access patterns (ensure stride-1 access)
4. Kernel fusion: Identify two adjacent kernels that can be merged to reduce memory round-trips
5. LDS bank conflict avoidance: Add padding to shared memory arrays to avoid 32-bank conflicts
6. Remove GPU sharding: If code splits work across GPUs due to 80GB limit, remove -- MI300X has 192GB
7. Loop unrolling: Add #pragma unroll for small fixed-size loops

DO NOT invent optimizations. Stick strictly to the list above.
DO NOT suggest anything you are not 100% certain will improve AMD performance.
If the code is already well-optimized, say so -- fewer changes is better than wrong ones.
"""

SYSTEM_PROMPT = f"""You are an AMD MI300X performance engineer. You receive HIP code and apply AMD-specific optimizations.

{ALLOWED_OPTIMIZATIONS}

Return ONLY this JSON, no markdown:
{{
  "optimized_code": "the complete optimized HIP code",
  "changes": [
    {{
      "description": "Replaced global memory access with shared memory tile (32x32)",
      "impact": "Reduces global memory bandwidth pressure, better L2 cache utilization"
    }}
  ]
}}

Be conservative. 2-3 high-confidence changes beat 10 uncertain ones."""


def run(hip_code: str, analyzer_result: AnalyzerResult,
        iteration: int = 1, previous_feedback: str = None) -> OptimizerResult:

    context = f"""
Optimize this HIP code for AMD MI300X.

Hardware context:
- MI300X: 192GB HBM3, 5.3 TB/s bandwidth, wavefront size = 64
- Workload classification: {analyzer_result.workload_type.value}
- {"MEMORY-BOUND: prioritize memory coalescing and shared memory tiling" if analyzer_result.workload_type == WorkloadType.MEMORY_BOUND else "COMPUTE-BOUND: prioritize arithmetic efficiency and register usage"}
"""

    if iteration == 2 and previous_feedback:
        context += f"""
ITERATION 2 -- Previous optimization made performance WORSE.
Profiler feedback: {previous_feedback}
Try a DIFFERENT strategy. If you applied shared memory tiling, try memory coalescing instead.
"""

    context += f"\nHIP code to optimize:\n```\n{hip_code}\n```"

    raw = chat_complete(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(raw)

    return OptimizerResult(
        optimized_code=data.get("optimized_code", hip_code),
        changes=data.get("changes", []),
        iteration=iteration,
    )
