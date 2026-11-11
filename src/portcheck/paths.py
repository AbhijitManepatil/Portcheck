"""Root resolution, path safety and run ids (spec sections 6.1 and 6.2)."""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from portcheck.envelope import InvalidInputError, NotFoundError, PathOutsideRootError

RUN_KINDS = ("scan", "plan", "capture", "compare", "cost", "report")
RUN_ID_RE = re.compile(r"^(scan|plan|capture|compare|cost|report)-\d{8}T\d{6}Z-[0-9a-f]{6}$")
RESULT_FILE = "result.json"


def get_root() -> Path:
    """The root directory: ``PORTCHECK_ROOT`` if set, else the current directory."""
    env = os.environ.get("PORTCHECK_ROOT")
    return Path(env).resolve() if env else Path.cwd().resolve()


def resolve_in_root(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve ``path`` relative to the root; reject anything that lands outside it."""
    root = root or get_root()
    text = os.fspath(path)
    if "\x00" in text:
        raise InvalidInputError("Path contains a null byte.")
    resolved = (root / text).resolve()
    if resolved != root and root not in resolved.parents:
        raise PathOutsideRootError(f"Path '{text}' resolves outside the root directory '{root}'.")
    return resolved


def rel_to_root(path: Path, root: Path | None = None) -> str:
    """A forward-slash path relative to the root ('.' for the root itself)."""
    root = root or get_root()
    rel = path.resolve().relative_to(root)
    return rel.as_posix() or "."


def runs_dir(root: Path | None = None) -> Path:
    return (root or get_root()) / ".portcheck" / "runs"


def new_run_id(kind: str) -> str:
    if kind not in RUN_KINDS:
        raise InvalidInputError(f"Unknown run kind '{kind}'.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{stamp}-{secrets.token_hex(3)}"


def is_run_id(value: str, kind: str | None = None) -> bool:
    match = RUN_ID_RE.match(value or "")
    return bool(match) and (kind is None or match.group(1) == kind)


def create_run_dir(kind: str) -> tuple[str, Path]:
    """Create a fresh run folder and return ``(run_id, path)``."""
    while True:
        run_id = new_run_id(kind)
        path = runs_dir() / run_id
        try:
            path.mkdir(parents=True, exist_ok=False)
            return run_id, path
        except FileExistsError:
            continue


def run_dir(run_id: str, kind: str) -> Path:
    """The folder of an existing run of the given kind."""
    if not isinstance(run_id, str) or not is_run_id(run_id, kind):
        raise InvalidInputError(f"'{run_id}' is not a valid {kind} run id.")
    path = runs_dir() / run_id
    if not path.is_dir():
        raise NotFoundError(f"No {kind} result with run id '{run_id}' under .portcheck/runs.")
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NotFoundError(
            f"Result file '{path.name}' is missing in '{path.parent.name}'."
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"Result file '{path}' is not valid JSON: {exc}") from exc


def save_result(kind: str, data: dict[str, Any]) -> tuple[str, str]:
    """Write ``data`` to a new run folder. Returns ``(run_id, result_path)``."""
    run_id, path = create_run_dir(kind)
    write_json(path / RESULT_FILE, data)
    return run_id, rel_to_root(path)


def load_result(run_id: str, kind: str) -> dict[str, Any]:
    return read_json(run_dir(run_id, kind) / RESULT_FILE)
