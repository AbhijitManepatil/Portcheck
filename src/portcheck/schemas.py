"""All pydantic models (spec section 7). Field names are fixed."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Severity = Literal["red", "yellow", "green"]
FixKind = Literal["auto", "manual", "none"]
Vendor = Literal["nvidia", "amd", "cpu"]
DType = Literal["float32", "float16", "bfloat16"]
Verdict = Literal["pass", "fail"]

SEVERITY_ORDER: dict[str, int] = {"red": 0, "yellow": 1, "green": 2}


class Fingerprint(BaseModel):
    vendor: Vendor
    device_name: str | None = None
    device_count: int = 0
    torch_version: str | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    driver_version: str | None = None
    python_version: str
    platform: str
    portcheck_version: str


class Finding(BaseModel):
    rule_id: str
    severity: Severity
    title: str
    file: str
    line: int | None = None
    evidence: str
    action: str
    fix: FixKind


class ScanResult(BaseModel):
    root: str
    files_scanned: int
    grade: Severity
    counts: dict[str, int]
    findings: list[Finding]


class PlanStep(BaseModel):
    step_id: str
    rule_id: str
    kind: Literal["auto", "manual"]
    title: str
    files: list[str]
    detail: str


class Plan(BaseModel):
    scan_id: str
    steps: list[PlanStep]


class Workload(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str = Field(min_length=1)
    revision: str | None = None
    task: str = "text-generation"
    dtype: DType = "float32"
    prompts: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1)
    batch_sizes: list[int] = Field(default_factory=lambda: [1])
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> Workload:
        if self.task != "text-generation":
            raise ValueError(
                f"task '{self.task}' is not supported; v0.1 supports only 'text-generation'"
            )
        if not self.batch_sizes or any(b < 1 for b in self.batch_sizes):
            raise ValueError("batch_sizes must be a non-empty list of positive integers")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must not contain duplicates")
        return self


class BenchRow(BaseModel):
    batch_size: int
    tokens_generated: int
    seconds: float
    tokens_per_second: float
    latency_p50_s: float
    latency_p95_s: float


class Capture(BaseModel):
    label: str
    workload: dict[str, Any]
    workload_sha256: str
    fingerprint: Fingerprint
    num_prompts: int
    fault_injected: bool = False
    bench: list[BenchRow]
    peak_memory_bytes: int | None = None


class CompareMetrics(BaseModel):
    max_abs_diff: float
    mean_rel_error: float
    min_cosine: float
    top1_agreement: float
    greedy_exact_match_rate: float
    mean_matching_prefix_tokens: float


class NoiseFloor(BaseModel):
    max_abs_diff: float = 0.0
    mean_rel_error: float = 0.0


class Thresholds(BaseModel):
    max_abs_diff: float
    min_cosine: float
    min_top1_agreement: float


class Check(BaseModel):
    metric: str
    value: float
    threshold: float
    passed: bool


class CompareResult(BaseModel):
    reference: str
    candidate: str
    baseline_repeat: str | None = None
    dtype: str
    metrics: CompareMetrics
    noise_floor: NoiseFloor
    thresholds: Thresholds
    checks: list[Check]
    verdict: Verdict


class CostRow(BaseModel):
    batch_size: int
    tokens_per_second: float
    latency_p50_s: float
    latency_p95_s: float
    cost_per_million_tokens: float


class CostResult(BaseModel):
    capture: str
    price_per_hour: float
    currency: str
    rows: list[CostRow]
