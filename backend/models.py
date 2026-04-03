from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class AgentStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"


class WorkloadType(str, Enum):
    COMPUTE_BOUND = "compute-bound"
    MEMORY_BOUND = "memory-bound"
    UNKNOWN = "unknown"


class PortRequest(BaseModel):
    cuda_code: str
    kernel_name: Optional[str] = "custom"
    simple_mode: Optional[bool] = False  # For "Explain Like I'm 5" feature


class AgentEvent(BaseModel):
    agent: str          # analyzer | translator | optimizer | tester | coordinator
    status: AgentStatus
    message: str
    detail: Optional[str] = None


class VerificationResult(BaseModel):
    compiled_successfully: bool
    executed_without_error: bool
    output_matches_expected: bool
    checksum_computed: Optional[str] = None
    expected_checksum: Optional[str] = None
    actual_checksum: Optional[str] = None
    mock_mode: Optional[bool] = False


class CostEstimate(BaseModel):
    manual_porting_weeks: str
    rocmport_minutes: str
    estimated_savings: str
    complexity_factor: str  # Low | Medium | High


class AnalyzerResult(BaseModel):
    kernels_found: List[str]
    cuda_apis: List[str]
    warp_size_issue: bool
    warp_size_detail: Optional[str]
    workload_type: WorkloadType
    sharding_detected: bool
    difficulty: str     # Easy | Medium | Hard
    difficulty_reason: str
    prediction: Optional[str] = None  # 🧠 Prediction field
    line_count: Optional[int] = None
    complexity_score: Optional[int] = None


class TranslatorResult(BaseModel):
    hip_code: str
    total_changes: int
    hipify_changes: int
    llm_changes: int
    diff_lines: List[dict]   # [{line, old, new, confidence, source}]


class OptimizerResult(BaseModel):
    optimized_code: str
    changes: List[dict]      # [{description, impact}]
    iteration: int


class TesterResult(BaseModel):
    success: bool
    iteration: int
    speedup: float           # vs baseline HIP
    bandwidth_utilized: float   # percentage
    execution_ms: float
    bottleneck: str
    notes: str
    verification: Optional[VerificationResult] = None  # Trust layer verification


class FinalReport(BaseModel):
    migration_success: bool
    speedup: float
    bandwidth_utilized: float
    total_changes: int
    bottleneck: str
    amd_advantage_explanation: str
    iterations: int
    hip_code: str
    optimized_code: str
    cost_estimate: Optional[CostEstimate] = None  # 💰 Cost impact estimator
    simplified_explanation: Optional[str] = None  # For "Explain Like I'm 5" mode
