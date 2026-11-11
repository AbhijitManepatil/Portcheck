"""Workload file loading and validation (spec section 7.4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from portcheck.envelope import InvalidInputError, NotFoundError
from portcheck.paths import get_root, resolve_in_root
from portcheck.schemas import Workload


@dataclass(frozen=True)
class LoadedWorkload:
    workload: Workload
    sha256: str
    prompt_form: str  # "text" or "input_ids"
    texts: list[str]
    input_ids: list[list[int]]
    model_ref: str  # a local directory, or the hub id as written
    model_is_local: bool
    path: Path

    @property
    def num_prompts(self) -> int:
        return len(self.texts) if self.prompt_form == "text" else len(self.input_ids)


def _parse_prompts(raw: bytes, name: str) -> tuple[str, list[str], list[list[int]]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidInputError(f"Prompts file '{name}' is not valid UTF-8.") from exc
    form: str | None = None
    texts: list[str] = []
    ids: list[list[int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InvalidInputError(f"{name} line {number}: not valid JSON ({exc.msg}).") from exc
        if (
            not isinstance(obj, dict)
            or len(obj) != 1
            or next(iter(obj)) not in ("text", "input_ids")
        ):
            raise InvalidInputError(
                f'{name} line {number}: expected {{"text": "..."}} or {{"input_ids": [...]}}.'
            )
        key = next(iter(obj))
        if form is None:
            form = key
        elif key != form:
            raise InvalidInputError(
                f"{name} line {number}: uses '{key}' but earlier lines use '{form}'. "
                "A prompts file must use one form throughout."
            )
        value = obj[key]
        if key == "text":
            if not isinstance(value, str) or not value:
                raise InvalidInputError(f"{name} line {number}: 'text' must be a non-empty string.")
            texts.append(value)
        else:
            ok = (
                isinstance(value, list)
                and value
                and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)
            )
            if not ok:
                raise InvalidInputError(
                    f"{name} line {number}: 'input_ids' must be a non-empty list of "
                    "non-negative integers."
                )
            ids.append(list(value))
    if form is None:
        raise InvalidInputError(f"Prompts file '{name}' contains no prompts.")
    return form, texts, ids


def _resolve_model(model: str, workload_dir: Path, root: Path) -> tuple[str, bool]:
    candidates = (
        [Path(model)] if Path(model).is_absolute() else [workload_dir / model, root / model]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve()), True
    return model, False


def load_workload(path: str) -> LoadedWorkload:
    """Load a workload file (path relative to the root) and its prompts file."""
    root = get_root()
    wl_path = resolve_in_root(path, root)
    if not wl_path.is_file():
        raise NotFoundError(f"Workload file '{path}' does not exist under the root directory.")
    wl_bytes = wl_path.read_bytes()
    try:
        data = yaml.safe_load(wl_bytes.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise InvalidInputError(f"Workload file '{path}' is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidInputError(f"Workload file '{path}' must contain a mapping.")
    try:
        workload = Workload.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            (".".join(str(p) for p in e["loc"]) + ": " if e["loc"] else "") + e["msg"]
            for e in exc.errors()
        )
        raise InvalidInputError(f"Invalid workload file '{path}': {problems}") from exc

    prompts_path = resolve_in_root(wl_path.parent / workload.prompts, root)
    if not prompts_path.is_file():
        raise NotFoundError(f"Prompts file '{workload.prompts}' does not exist next to '{path}'.")
    prompt_bytes = prompts_path.read_bytes()
    form, texts, ids = _parse_prompts(prompt_bytes, prompts_path.name)

    model_ref, is_local = _resolve_model(workload.model, wl_path.parent, root)
    if not is_local and not workload.revision:
        raise InvalidInputError(
            f"Model '{workload.model}' is not a local directory, so it is treated as a hub id, "
            "and a hub id needs 'revision' set to a commit hash in the workload file."
        )

    digest = hashlib.sha256()
    digest.update(wl_bytes)
    digest.update(b"\x00")
    digest.update(prompt_bytes)
    return LoadedWorkload(
        workload=workload,
        sha256=digest.hexdigest(),
        prompt_form=form,
        texts=texts,
        input_ids=ids,
        model_ref=model_ref,
        model_is_local=is_local,
        path=wl_path,
    )
