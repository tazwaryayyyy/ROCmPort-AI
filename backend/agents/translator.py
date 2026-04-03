import json
import re
from models import TranslatorResult, AnalyzerResult
from tools.llm_client import LLMClient
from tools.hipify_wrapper import HipifyWrapper

llm_client = LLMClient()
hipify_wrapper = HipifyWrapper()

def chat_complete(messages: list) -> str:
    """Wrapper for LLM client chat completion"""
    return llm_client.chat_completion(messages)

def run_hipify(cuda_code: str) -> str:
    """Wrapper for hipify wrapper"""
    return hipify_wrapper.hipify_code(cuda_code)

SYSTEM_PROMPT = """You are an expert AMD ROCm/HIP engineer. You receive CUDA code that has already gone through hipify (basic syntax replacement) and you fix what hipify missed.

Your specific jobs:
1. Fix warp size assumptions: any code assuming warpSize=32 must be updated for AMD wavefront size of 64
   - Hardcoded 32 in reductions -> use 64 explicitly or warpSize
   - __ballot_sync(0xffffffff, ...) -> __ballot(...)
   - __shfl_sync -> __shfl (HIP equivalent)
2. Fix kernel launch syntax if broken
3. Fix any CUDA intrinsics with no direct HIP equivalent
4. Ensure #include uses hip/hip_runtime.h not cuda_runtime.h

Return ONLY this JSON, no markdown:
{
  "fixed_code": "the complete fixed HIP code here",
  "llm_changes": [
    {
      "description": "Fixed warp size assumption: changed hardcoded 32 to 64 for AMD wavefront",
      "confidence": "high"
    }
  ]
}

If nothing needs fixing beyond what hipify did, return the code unchanged with empty llm_changes array."""


def run(cuda_code: str, analyzer_result: AnalyzerResult) -> TranslatorResult:
    # Pass 1: hipify (mechanical replacements)
    hip_code_pass1, hipify_changes = run_hipify(cuda_code)

    # Pass 2: LLM fixes what hipify missed
    context = f"""
The following code has already been through hipify (basic CUDA->HIP syntax replacement).

Analyzer findings:
- Warp size issue detected: {analyzer_result.warp_size_issue}
- Warp size detail: {analyzer_result.warp_size_detail or 'none'}
- Workload type: {analyzer_result.workload_type}
- CUDA APIs found: {', '.join(analyzer_result.cuda_apis)}

Fix what hipify missed, especially warp size issues.

Code after hipify:
```
{hip_code_pass1}
```
"""

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

    final_code = data.get("fixed_code", hip_code_pass1)
    llm_changes = data.get("llm_changes", [])

    diff_lines = _build_diff(cuda_code, final_code)

    return TranslatorResult(
        hip_code=final_code,
        total_changes=len(hipify_changes) + len(llm_changes),
        hipify_changes=len(hipify_changes),
        llm_changes=len(llm_changes),
        diff_lines=diff_lines,
    )


def _build_diff(original: str, converted: str) -> list[dict]:
    orig_lines = original.splitlines()
    conv_lines = converted.splitlines()
    diff = []
    max_len = max(len(orig_lines), len(conv_lines))
    for i in range(max_len):
        o = orig_lines[i] if i < len(orig_lines) else ""
        c = conv_lines[i] if i < len(conv_lines) else ""
        if o != c:
            diff.append({"line": i + 1, "old": o, "new": c})
    return diff
