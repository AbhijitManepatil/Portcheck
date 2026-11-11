"""Drive `portcheck-mcp` with the official MCP client library over stdio.

`test_server.py` speaks raw JSON-RPC; this is a second, independent client, which is
the CPU stand-in for the release check "same findings from the second MCP client".
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from conftest import slim

mcp = pytest.importorskip("mcp")
stdio = pytest.importorskip("mcp.client.stdio")


async def _session(root: str) -> dict:
    params = mcp.StdioServerParameters(
        command=sys.executable,
        args=["-m", "portcheck.server"],
        env=dict(os.environ, PORTCHECK_ROOT=root),
    )
    out: dict = {}
    async with stdio.stdio_client(params) as (read, write):
        async with mcp.ClientSession(read, write) as session:
            await session.initialize()
            out["tools"] = sorted(t.name for t in (await session.list_tools()).tools)

            async def call(name: str, **arguments):
                result = await session.call_tool(name, arguments)
                return json.loads(result.content[0].text)

            out["scan"] = await call("scan_repo", path=".")
            out["plan"] = await call("plan_migration", scan_id=out["scan"]["run_id"])
            out["dry"] = await call("apply_step", step_id="S3", plan_id=out["plan"]["run_id"])
            out["wrong_hash"] = await call(
                "apply_step",
                step_id="S3",
                plan_id=out["plan"]["run_id"],
                dry_run=False,
                diff_sha256="0" * 64,
            )
            out["applied"] = await call(
                "apply_step",
                step_id="S3",
                plan_id=out["plan"]["run_id"],
                dry_run=False,
                diff_sha256=out["dry"]["data"]["diff_sha256"],
            )
            out["outside"] = await call("scan_repo", path="../..")
            out["report"] = await call("build_report", scan_id=out["scan"]["run_id"])
    return out


def test_official_client_gets_the_same_results(fixture_repo, golden):
    before = (fixture_repo / "gpu_stats.py").read_text()
    out = asyncio.run(asyncio.wait_for(_session(str(fixture_repo)), timeout=120))

    assert len(out["tools"]) == 8 and "scan_repo" in out["tools"]
    assert out["scan"]["ok"] and slim(out["scan"]["data"]["findings"]) == golden
    assert [s["kind"] for s in out["plan"]["data"]["steps"]] == ["auto"] * 3 + ["manual"] * 2
    assert out["dry"]["data"]["applied"] is False and out["dry"]["data"]["diff"]
    assert out["wrong_hash"]["error"]["code"] == "diff_mismatch"
    assert out["applied"]["data"]["applied"] is True
    assert out["outside"]["error"]["code"] == "path_outside_root"
    assert out["report"]["ok"]
    after = (fixture_repo / "gpu_stats.py").read_text()
    assert after != before and "except ImportError" in after
