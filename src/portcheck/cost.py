"""``estimate_cost`` (spec sections 7.7 and 8.7)."""

from __future__ import annotations

import math

from portcheck.envelope import InvalidInputError
from portcheck.schemas import CostResult, CostRow
from portcheck.validate.capture import load_capture


def cost_per_million_tokens(price_per_hour: float, tokens_per_second: float) -> float:
    return round(price_per_hour / (tokens_per_second * 3600) * 1_000_000, 4)


def estimate_cost(capture: str, price_per_hour: float, currency: str = "USD") -> CostResult:
    """Turn a capture's measured throughput into cost per million tokens."""
    if isinstance(price_per_hour, bool) or not isinstance(price_per_hour, (int, float)):
        raise InvalidInputError("price_per_hour must be a number.")
    if not math.isfinite(price_per_hour) or price_per_hour <= 0:
        raise InvalidInputError("price_per_hour must be greater than zero.")
    if not isinstance(currency, str) or not currency.strip():
        raise InvalidInputError("currency must be a non-empty string, for example 'USD'.")
    name, cap, _directory = load_capture(capture)
    rows = []
    for row in cap.bench:
        if row.tokens_per_second <= 0:
            raise InvalidInputError(
                f"Capture '{name}' has no measured throughput at batch size {row.batch_size}."
            )
        rows.append(
            CostRow(
                batch_size=row.batch_size,
                tokens_per_second=row.tokens_per_second,
                latency_p50_s=row.latency_p50_s,
                latency_p95_s=row.latency_p95_s,
                cost_per_million_tokens=cost_per_million_tokens(
                    float(price_per_hour), row.tokens_per_second
                ),
            )
        )
    return CostResult(
        capture=name, price_per_hour=float(price_per_hour), currency=currency.strip(), rows=rows
    )
