"""
static_analyzer.py — Pure-Python wavefront correctness scanner.

Runs BEFORE the LLM sees any code. Zero external dependencies. Typical run time < 5ms.

Detects the six most common categories of CUDA→AMD correctness hazards caused by the
NVIDIA warpSize=32 vs AMD wavefront=64 mismatch. Results are fed as structured pre-analysis
context into the LLM analyzer prompt, making the LLM's job more targeted and auditable.
"""

import re
import time
from typing import List

from ..models import RiskItem, StaticRiskReport


# ---------------------------------------------------------------------------
# Risk pattern definitions
# Each entry: (pattern_name, regex, risk_level, description, amd_fix_hint)
# ---------------------------------------------------------------------------

_PATTERNS: List[tuple] = [
    (
        "warp_size_hardcoded_32_conditional",
        re.compile(r'\btid\s*<\s*32\b|\bthreadIdx\.x\s*<\s*32\b|\bi\s*<\s*32\b', re.MULTILINE),
        "CRITICAL",
        "Hardcoded '<32' in thread conditional — assumes NVIDIA warpSize=32. "
        "On AMD wavefront=64 this silently skips lanes 32–63 in final reduction stages, "
        "producing incorrect results.",
        "Expand final stage: check 'tid < 64' first, then 'tid < 32'. "
        "See AMD wavefront reduction pattern in docs/JUDGE_MODE.md."
    ),
    (
        "warp_size_define_32",
        re.compile(r'#\s*define\s+WARP_SIZE\s+32\b', re.MULTILINE),
        "CRITICAL",
        "#define WARP_SIZE 32 — this constant will produce wrong kernel geometry on AMD. "
        "Wavefront size is 64 on all GCN/CDNA architectures including MI300X.",
        "Change to #define WARP_SIZE 64 or use the runtime constant wavefrontSize "
        "from hipDeviceGetAttribute(HIP_DEVICE_ATTRIBUTE_WAVEFRONT_SIZE)."
    ),
    (
        "shfl_sync_warp_primitive",
        re.compile(r'\b__shfl_sync\b|\b__shfl_up_sync\b|\b__shfl_down_sync\b|\b__shfl_xor_sync\b', re.MULTILINE),
        "CRITICAL",
        "__shfl_sync family requires the 0xffffffff mask to be reinterpreted for 64-lane wavefronts. "
        "hipify replaces the function name but not the mask — lanes 32–63 are excluded.",
        "Replace with __shfl, __shfl_up, __shfl_down, __shfl_xor (no mask arg in HIP). "
        "Verify lane shuffle ranges cover the full 64-lane wavefront."
    ),
    (
        "ballot_sync_mask",
        re.compile(r'\b__ballot_sync\s*\(\s*0x[Ff]+\s*,', re.MULTILINE),
        "CRITICAL",
        "__ballot_sync(0xffffffff, ...) uses a 32-bit full mask. On AMD this is __ballot() "
        "with no mask argument — the 32-bit mask is semantically wrong for a 64-lane wavefront.",
        "Replace __ballot_sync(0xffffffff, cond) with __ballot(cond). "
        "The return type changes from uint32_t to uint64_t — update downstream bitmask logic."
    ),
    (
        "shfl_wavefront_offset_16",
        re.compile(r'\b__shfl(?:_down|_up|_xor)?\s*\([^;]*,\s*16\s*(?:,|\))', re.MULTILINE),
        "HIGH",
        "__shfl* with offset=16 often encodes a 32-lane warp reduction tail. "
        "On AMD wavefront=64 the reduction should include an offset=32 step first.",
        "Audit the shuffle reduction and add a wavefront-64 step, e.g. offset=32 "
        "before offset=16 where the algorithm reduces a full wavefront."
    ),
    (
        "activemask_warp",
        re.compile(r'\b__activemask\s*\(\s*\)', re.MULTILINE),
        "HIGH",
        "__activemask() returns a 32-bit value on NVIDIA. On AMD __activemask() "
        "or __ballot(1) returns a 64-bit value. Storing in uint32_t will truncate lanes 32–63.",
        "Declare the result as uint64_t. Audit all bitmask operations for 64-bit correctness."
    ),
    (
        "threadidx_modulo_warpsize",
        re.compile(r'threadIdx\.x\s*%\s*(?:32|warpSize)\b', re.MULTILINE),
        "HIGH",
        "threadIdx.x % 32 assumes 32-lane warps. On AMD wavefront=64 the lane ID "
        "within a wavefront requires modulo 64.",
        "Use threadIdx.x % 64 or threadIdx.x & 63 for the lane ID within a wavefront."
    ),
    (
        "reduction_loop_stops_at_32",
        re.compile(r'for\s*\([^)]*\bs\s*>\s*32\b', re.MULTILINE),
        "HIGH",
        "Reduction loop terminates at s>32 before manually unrolling the final 32 lanes. "
        "On AMD the loop should terminate at s>64 to correctly handle the 64-lane warp tail.",
        "Change loop bound from s>32 to s>64. Expand the manual unroll below the loop "
        "to cover tid<64 before the tid<32 block."
    ),
    (
        "inline_ptx_block",
        re.compile(r'asm\s+volatile\s*\(', re.MULTILINE),
        "CRITICAL",
        "Inline PTX assembly is NVIDIA-specific ISA. hipify cannot translate PTX semantics. "
        "The kernel may compile under hipcc but will have undefined or incorrect behaviour.",
        "Replace inline PTX with portable HIP intrinsics or CDNA ISA equivalents. "
        "Common cases: lane_id → __lane_id(), __clz → __clz() (same name in HIP)."
    ),
    (
        "cuda_runtime_include",
        re.compile(r'#\s*include\s*[<\"]cuda_runtime(?:_api)?\.h[>\"]', re.MULTILINE),
        "MEDIUM",
        "cuda_runtime.h / cuda_runtime_api.h must be replaced with hip/hip_runtime.h. "
        "hipify handles this mechanically but the check confirms it was applied.",
        "Replace with #include <hip/hip_runtime.h>. "
        "hipify-clang does this automatically in its first pass."
    ),
    (
        "cuda_library_dependency",
        re.compile(r'#\s*include\s*[<"][^>"]*(?:cub|thrust|cudnn)[^>"]*[>"]|\b(?:cub|thrust|cudnn)::', re.MULTILINE),
        "HIGH",
        "CUDA library dependency detected. hipify can rename some CUB/Thrust/cuDNN symbols, "
        "but API coverage and performance behavior are not guaranteed to match rocPRIM/hipCUB/MIOpen.",
        "Manually review the translated library call, compare against rocPRIM/hipCUB/MIOpen, "
        "and add correctness/performance tests for the specific primitive."
    ),
    (
        "shared_memory_no_padding",
        re.compile(r'__shared__\s+\w+\s+\w+\s*\[\s*\d+\s*\]', re.MULTILINE),
        "MEDIUM",
        "Fixed-size shared memory array detected without padding. AMD LDS has 32 banks of 4B. "
        "Arrays whose inner dimension is a power-of-2 may cause systematic bank conflicts.",
        "Add +1 padding to the inner dimension, e.g., __shared__ float tile[32][33]. "
        "This staggers accesses across banks and eliminates the conflict."
    ),
]


def _find_line_number(code: str, match_start: int) -> int:
    """Convert a character offset into a 1-indexed line number."""
    return code[:match_start].count('\n') + 1


def scan(cuda_code: str) -> StaticRiskReport:
    """
    Scan CUDA source for AMD compatibility hazards.

    Returns a StaticRiskReport with structured RiskItems, counts by severity,
    and the wall-clock scan duration for transparency.
    """
    t0 = time.perf_counter()
    items: List[RiskItem] = []

    for pattern_name, regex, risk_level, description, amd_fix_hint in _PATTERNS:
        for match in regex.finditer(cuda_code):
            line_num = _find_line_number(cuda_code, match.start())
            items.append(RiskItem(
                line=line_num,
                pattern=pattern_name,
                risk_level=risk_level,
                description=description,
                amd_fix_hint=amd_fix_hint,
            ))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    critical = sum(1 for i in items if i.risk_level == "CRITICAL")
    high = sum(1 for i in items if i.risk_level == "HIGH")
    medium = sum(1 for i in items if i.risk_level == "MEDIUM")

    return StaticRiskReport(
        items=items,
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        scan_duration_ms=round(elapsed_ms, 3),
    )


def format_for_llm_prompt(report: StaticRiskReport) -> str:
    """
    Render the static report as a compact context block to inject into LLM prompts.
    Keeps token usage low while giving the LLM grounded, actionable pre-analysis.
    """
    if not report.items:
        return "Static pre-scan: No known AMD compatibility hazards detected."

    lines = [
        f"=== STATIC PRE-SCAN ({report.critical_count} CRITICAL, "
        f"{report.high_count} HIGH, {report.medium_count} MEDIUM) ===",
        "The following hazards were detected by deterministic pattern matching BEFORE LLM analysis.",
        "Confirm and expand on these findings — do NOT contradict them without strong evidence.",
        "",
    ]
    for item in report.items:
        loc = f"line {item.line}" if item.line else "location unknown"
        lines.append(f"[{item.risk_level}] {item.pattern} @ {loc}")
        lines.append(f"  Issue: {item.description}")
        lines.append(f"  Fix:   {item.amd_fix_hint}")
        lines.append("")

    return "\n".join(lines)
