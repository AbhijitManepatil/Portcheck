"""``apply_step``: dry-run a fix as a diff, then apply it only on a matching hash."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from portcheck.audit.parsers import split_lines
from portcheck.config import load_config
from portcheck.envelope import DiffMismatchError, InvalidInputError, NotFoundError
from portcheck.migrate.fixes import FIXES
from portcheck.migrate.planner import find_step, load_plan, load_scan
from portcheck.paths import get_root, rel_to_root, resolve_in_root


def _unified(old: str, new: str, rel_path: str) -> str:
    lines = difflib.unified_diff(
        split_lines(old, keepends=True),
        split_lines(new, keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    )
    out = []
    for line in lines:
        if line.endswith(("\n", "\r")):
            out.append(line)
        else:
            out.append(line + "\n\\ No newline at end of file\n")
    return "".join(out)


def _write_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` in one step, so an interrupted write never leaves half a file."""
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".portcheck-tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def diff_hash(diff: str) -> str:
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def apply_step(
    step_id: str, plan_id: str, dry_run: bool = True, diff_sha256: str | None = None
) -> dict[str, Any]:
    """Preview (dry run) or apply one auto step of a plan.

    Returns ``{"diff", "diff_sha256", "applied", "files"}``. Nothing is written to the
    repo unless ``dry_run`` is false and ``diff_sha256`` matches the recomputed diff.
    """
    plan = load_plan(plan_id)
    step = find_step(plan, step_id)
    if step.kind != "auto":
        raise InvalidInputError(
            f"Step {step.step_id} ({step.rule_id}) is manual: portcheck cannot apply it. "
            f"{step.detail}"
        )
    fix = FIXES.get(step.rule_id)
    if fix is None:
        raise InvalidInputError(f"No automatic fix exists for rule {step.rule_id}.")
    if not dry_run and not diff_sha256:
        raise InvalidInputError(
            "diff_sha256 is required when dry_run is false. Run the step as a dry run first "
            "and pass the diff_sha256 it returns."
        )

    root = get_root()
    config = load_config(root)
    scan = load_scan(plan.scan_id)

    edits: list[tuple[Path, str, bytes]] = []
    diffs: list[str] = []
    for file in step.files:
        path = resolve_in_root(f"{scan.root}/{file}", root)
        if not path.is_file():
            raise NotFoundError(f"File '{file}' from the plan no longer exists.")
        try:
            raw = path.read_bytes()
            old = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidInputError(f"File '{file}' is not valid UTF-8.") from exc
        # Keep a leading byte-order mark out of the fix's way, then put it back.
        bom = "\ufeff" if old.startswith("\ufeff") else ""
        new = bom + fix(old[len(bom) :], path.name, config)
        if new != old:
            rel = rel_to_root(path, root)
            diffs.append(_unified(old, new, rel))
            edits.append((path, rel, new.encode("utf-8")))

    diff = "".join(diffs)
    digest = diff_hash(diff)
    changed = [rel for _path, rel, _data in edits]
    if dry_run:
        return {"diff": diff, "diff_sha256": digest, "applied": False, "files": changed}
    if digest != diff_sha256:
        raise DiffMismatchError(
            "The diff computed now does not match diff_sha256, so nothing was written. "
            "The files or the config changed since the dry run; run the dry run again and "
            "review the new diff."
        )
    for path, _rel, data in edits:
        _write_atomic(path, data)
    return {"diff": diff, "diff_sha256": digest, "applied": bool(edits), "files": changed}
