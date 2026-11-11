"""Error types and the return envelope shared by the CLI and the MCP server."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

ERROR_CODES = (
    "path_outside_root",
    "not_found",
    "invalid_input",
    "config_missing",
    "missing_dependency",
    "diff_mismatch",
    "capture_mismatch",
    "internal",
)


class PortcheckError(Exception):
    """An expected failure that maps to one envelope error code."""

    code = "internal"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class PathOutsideRootError(PortcheckError):
    code = "path_outside_root"


class NotFoundError(PortcheckError):
    code = "not_found"


class InvalidInputError(PortcheckError):
    code = "invalid_input"


class ConfigMissingError(PortcheckError):
    code = "config_missing"


class MissingDependencyError(PortcheckError):
    code = "missing_dependency"


class DiffMismatchError(PortcheckError):
    code = "diff_mismatch"


class CaptureMismatchError(PortcheckError):
    code = "capture_mismatch"


def ok_envelope(
    summary: str, data: dict[str, Any], run_id: str | None = None, result_path: str | None = None
) -> dict[str, Any]:
    return {
        "ok": True,
        "run_id": run_id,
        "result_path": result_path,
        "summary": summary,
        "data": data,
    }


def error_envelope(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "run_id": None,
        "result_path": None,
        "summary": message,
        "error": {"code": code, "message": message},
    }


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts)


def guarded(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``fn`` and turn every exception into a failure envelope. Never raises."""
    try:
        return fn()
    except PortcheckError as exc:
        return error_envelope(exc.code, exc.message)
    except ValidationError as exc:
        return error_envelope("invalid_input", _validation_message(exc))
    except Exception as exc:  # noqa: BLE001 - tools must never raise to the client
        traceback.print_exc(file=sys.stderr)
        return error_envelope("internal", f"{type(exc).__name__}: {exc}")
