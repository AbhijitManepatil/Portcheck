from __future__ import annotations

import pytest

from portcheck import tools
from portcheck.cost import cost_per_million_tokens, estimate_cost


def test_formula_against_hand_computed_value():
    # 2.5 per hour at 128 tokens/s: 128 * 3600 = 460800 tokens per hour;
    # 2.5 / 460800 * 1e6 = 5.42534722... -> 5.4253
    assert cost_per_million_tokens(2.5, 128.0) == 5.4253
    # 3.6 per hour at 1000 tokens/s is exactly 1.0 per million tokens.
    assert cost_per_million_tokens(3.6, 1000.0) == 1.0


def test_rows_follow_the_capture_bench(root, make_capture, base_logits):
    cap = make_capture("ref", base_logits, tokens_per_second=128.0)
    result = estimate_cost(cap, 2.5)
    assert result.capture == cap and result.price_per_hour == 2.5 and result.currency == "USD"
    assert [r.model_dump() for r in result.rows] == [
        {
            "batch_size": 1,
            "tokens_per_second": 128.0,
            "latency_p50_s": 0.24,
            "latency_p95_s": 0.31,
            "cost_per_million_tokens": 5.4253,
        },
        {
            "batch_size": 4,
            "tokens_per_second": 400.0,
            "latency_p50_s": 0.75,
            "latency_p95_s": 0.82,
            "cost_per_million_tokens": 1.7361,  # 2.5 / (400 * 3600) * 1e6 = 1.73611...
        },
    ]


@pytest.mark.parametrize("price", [0, 0.0, -1, float("nan"), float("inf")])
def test_non_positive_price_rejected(root, make_capture, base_logits, price):
    env = tools.estimate_cost(make_capture("ref", base_logits), price)
    assert env["ok"] is False and env["error"]["code"] == "invalid_input"


def test_tool_envelope_currency_and_errors(root, make_capture, base_logits):
    cap = make_capture("ref", base_logits)
    env = tools.estimate_cost(cap, 1.9, "EUR")
    assert env["ok"] and env["run_id"].startswith("cost-") and env["data"]["currency"] == "EUR"
    assert (root / env["result_path"] / "result.json").is_file()
    assert tools.estimate_cost(cap, 1.0, " ")["error"]["code"] == "invalid_input"
    assert tools.estimate_cost(cap, "cheap")["error"]["code"] == "invalid_input"
    assert tools.estimate_cost("capture-20200101T000000Z-abcdef", 1.0)["error"]["code"] == (
        "not_found"
    )
