from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys

import pytest

from portcheck import server

EXPECTED = {
    "env_fingerprint": [],
    "scan_repo": ["path"],
    "plan_migration": ["scan_id"],
    "apply_step": ["step_id", "plan_id", "dry_run", "diff_sha256"],
    "capture_run": ["workload", "label", "inject_fault"],
    "compare_captures": ["reference", "candidate", "baseline_repeat"],
    "estimate_cost": ["capture", "price_per_hour", "currency"],
    "build_report": ["scan_id", "compare_id", "cost_ids", "format"],
}
DEFAULTS = {
    "scan_repo": {"path": "."},
    "apply_step": {"dry_run": True, "diff_sha256": None},
    "capture_run": {"inject_fault": None},
    "compare_captures": {"baseline_repeat": None},
    "estimate_cost": {"currency": "USD"},
    "build_report": {"scan_id": None, "compare_id": None, "cost_ids": None, "format": "markdown"},
}
REQUIRED = {
    "plan_migration": ["scan_id"],
    "apply_step": ["step_id", "plan_id"],
    "capture_run": ["workload", "label"],
    "compare_captures": ["reference", "candidate"],
    "estimate_cost": ["capture", "price_per_hour"],
}


def _schema(tool) -> dict:
    return getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}


def _listed() -> dict:
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_exactly_the_eight_tools_are_registered():
    assert sorted(_listed()) == sorted(EXPECTED)
    assert sorted(server.TOOL_NAMES) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_tool_arguments_defaults_and_docstring(name):
    tool = _listed()[name]
    schema = _schema(tool)
    assert list(schema.get("properties", {})) == EXPECTED[name]
    assert sorted(schema.get("required", [])) == sorted(REQUIRED.get(name, []))
    # The Python signature carries the same names and the specified defaults.
    params = inspect.signature(getattr(server, name)).parameters
    assert list(params) == EXPECTED[name]
    defaults = {k: p.default for k, p in params.items() if p.default is not inspect.Parameter.empty}
    assert defaults == DEFAULTS.get(name, {})
    # 1 to 3 sentences that an agent can choose from.
    doc = " ".join((tool.description or "").split())
    assert doc and 1 <= len([s for s in doc.split(". ") if s.strip()]) <= 4
    assert len(doc) < 600


def test_bad_path_returns_ok_false_and_does_not_raise(root):
    for call in (
        lambda: server.scan_repo("../../etc"),
        lambda: server.scan_repo("/etc"),
        lambda: server.capture_run("../outside.yaml", "x"),
        lambda: server.compare_captures("../a", "../b"),
        lambda: server.estimate_cost("../a", 1.0),
    ):
        env = call()
        assert env["ok"] is False and env["error"]["code"] == "path_outside_root"
        assert "data" not in env


def test_tools_never_raise_on_bad_input(root):
    calls = [
        lambda: server.scan_repo("missing-dir"),
        lambda: server.plan_migration("not-an-id"),
        lambda: server.plan_migration(None),
        lambda: server.apply_step("S1", "plan-20200101T000000Z-abcdef"),
        lambda: server.apply_step(None, None, dry_run="yes"),
        lambda: server.capture_run("missing.yaml", "label"),
        lambda: server.capture_run("missing.yaml", "bad label!"),
        lambda: server.compare_captures("", ""),
        lambda: server.compare_captures(None, 3),
        lambda: server.estimate_cost("capture-20200101T000000Z-abcdef", -1),
        lambda: server.build_report(),
        lambda: server.build_report(cost_ids="cost-1"),
        lambda: server.build_report(scan_id=5),
    ]
    for call in calls:
        env = call()
        assert env["ok"] is False
        assert env["error"]["code"] in (
            "invalid_input",
            "not_found",
            "path_outside_root",
        ), env


def test_internal_errors_become_envelopes(root, monkeypatch):
    from portcheck.audit import scanner

    def boom(*_a, **_k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(scanner, "scan_repo", boom)
    env = server.scan_repo(".")
    assert env["ok"] is False and env["error"]["code"] == "internal"
    assert "unexpected" in env["error"]["message"]


def test_tool_output_on_stdout_is_redirected(root, capsys):
    from portcheck import tools

    def noisy(path="."):
        print("stray output")
        return {"ok": True}

    original = tools.scan_repo
    tools.scan_repo = noisy
    try:
        server.scan_repo(".")
    finally:
        tools.scan_repo = original
    captured = capsys.readouterr()
    assert captured.out == "" and "stray output" in captured.err


def _rpc(proc, message: dict) -> None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _read_response(proc, wanted_id: int) -> dict:
    while True:
        line = proc.stdout.readline()
        assert line, "server closed stdout before answering"
        message = json.loads(line)  # every stdout line must be a protocol message
        assert message.get("jsonrpc") == "2.0"
        if message.get("id") == wanted_id:
            return message


def test_server_starts_over_stdio_and_lists_eight_tools(fixture_repo):
    """Drive the real `portcheck-mcp` entry point with raw JSON-RPC over stdio."""
    env = dict(os.environ, PORTCHECK_ROOT=str(fixture_repo))
    proc = subprocess.Popen(
        [sys.executable, "-c", "from portcheck.server import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "portcheck-tests", "version": "0"},
                },
            },
        )
        init = _read_response(proc, 1)
        assert init["result"]["serverInfo"]["name"] == "portcheck"
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = _read_response(proc, 2)["result"]["tools"]
        assert sorted(t["name"] for t in listed) == sorted(EXPECTED)
        assert all(t.get("description") for t in listed)

        _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "scan_repo", "arguments": {"path": "."}},
            },
        )
        result = _read_response(proc, 3)["result"]
        envelope = json.loads(result["content"][0]["text"])
        assert envelope["ok"] is True and envelope["data"]["grade"] == "red"
        assert envelope["data"]["counts"] == {"red": 2, "yellow": 3, "green": 1}

        _rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "scan_repo", "arguments": {"path": "../.."}},
            },
        )
        result = _read_response(proc, 4)["result"]
        envelope = json.loads(result["content"][0]["text"])
        assert envelope["ok"] is False and envelope["error"]["code"] == "path_outside_root"
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
