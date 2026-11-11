"""File parsers for the audit: Python imports, dependency files and Dockerfiles."""

from __future__ import annotations

import ast
import fnmatch
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

_FALLBACK_IMPORT_RE = re.compile(r"^(\s*)(import|from)\s+([\w.]+)")
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_FROM_RE = re.compile(r"^\s*FROM\s+((?:--\S+\s+)*)(\S+)(.*)$", re.IGNORECASE)
_GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


@dataclass(frozen=True)
class ImportRef:
    """One imported module. ``candidates`` are the dotted names a rule may match."""

    module: str
    line: int
    kind: str  # "import" or "from"
    top_level: bool
    guarded: bool = False  # inside a try block that catches ImportError
    plain: bool = False  # a top-level, unguarded `import x` with one name and no alias
    candidates: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Dependency:
    name: str  # normalised: lowercase, runs of - _ . collapsed to -
    line: int | None
    raw: str


@dataclass(frozen=True)
class DockerFrom:
    line: int
    image: str
    raw: str


# ----------------------------------------------------------------------------- names


def normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def requirement_name(spec: str) -> str | None:
    """The normalised package name at the start of a requirement string, if any."""
    text = spec.strip()
    if not text or text.startswith(("#", "-")) or "://" in text.split("#")[0].split(";")[0][:12]:
        return None
    if "::" in text:  # conda channel prefix, e.g. nvidia::cudatoolkit
        text = text.split("::", 1)[1]
    match = _NAME_RE.match(text)
    return normalise_name(match.group(1)) if match else None


# ----------------------------------------------------------------------- file classes


def is_requirements_file(name: str) -> bool:
    return fnmatch.fnmatchcase(name, "requirements*.txt")


def is_dockerfile(name: str) -> bool:
    return (
        name == "Dockerfile"
        or fnmatch.fnmatchcase(name, "Dockerfile.*")
        or fnmatch.fnmatchcase(name, "*.Dockerfile")
    )


def is_pyproject(name: str) -> bool:
    return name == "pyproject.toml"


def is_environment_file(name: str) -> bool:
    return name in ("environment.yml", "environment.yaml")


# ---------------------------------------------------------------------------- imports


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for node in nodes:
        name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
        if name in _GUARD_EXCEPTIONS:
            return True
    return False


def _walk_imports(
    nodes: list[ast.stmt], top_level: bool, guarded: bool, out: list[ImportRef]
) -> None:
    for node in nodes:
        if isinstance(node, ast.Import):
            plain = (
                top_level and not guarded and len(node.names) == 1 and node.names[0].asname is None
            )
            for alias in node.names:
                out.append(
                    ImportRef(
                        module=alias.name,
                        line=node.lineno,
                        kind="import",
                        top_level=top_level,
                        guarded=guarded,
                        plain=plain,
                        candidates=(alias.name,),
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue  # relative imports never name a third-party package
            cands = (node.module, *(f"{node.module}.{a.name}" for a in node.names))
            out.append(
                ImportRef(
                    module=node.module,
                    line=node.lineno,
                    kind="from",
                    top_level=top_level,
                    guarded=guarded,
                    candidates=cands,
                )
            )
        elif isinstance(node, ast.Try) or node.__class__.__name__ == "TryStar":
            body_guarded = guarded or any(_catches_import_error(h) for h in node.handlers)
            _walk_imports(node.body, False, body_guarded, out)
            for handler in node.handlers:
                _walk_imports(handler.body, False, guarded, out)
            _walk_imports(node.orelse, False, guarded, out)
            _walk_imports(node.finalbody, False, guarded, out)
        else:
            for name in ("body", "orelse", "finalbody"):
                child = getattr(node, name, None)
                if isinstance(child, list) and child and isinstance(child[0], ast.stmt):
                    _walk_imports(child, False, guarded, out)
            for case in getattr(node, "cases", []) or []:
                _walk_imports(case.body, False, guarded, out)


def _fallback_imports(text: str) -> list[ImportRef]:
    out: list[ImportRef] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _FALLBACK_IMPORT_RE.match(line)
        if not match or match.group(3).startswith("."):
            continue
        module = match.group(3)
        cands: tuple[str, ...] = (module,)
        if match.group(2) == "from":
            names = re.match(r"^\s*from\s+[\w.]+\s+import\s+(.+)$", line)
            if names:
                parts = [p.strip().split(" ")[0] for p in names.group(1).strip("() \\").split(",")]
                cands += tuple(f"{module}.{p}" for p in parts if re.fullmatch(r"\w+", p))
        out.append(
            ImportRef(
                module=module,
                line=number,
                kind=match.group(2),
                top_level=match.group(1) == "",
                candidates=cands,
            )
        )
    return out


def parse_python_imports(text: str) -> list[ImportRef]:
    """Imported modules with line numbers. Falls back to a line regex on a syntax error."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return _fallback_imports(text)
    out: list[ImportRef] = []
    _walk_imports(tree.body, True, False, out)
    return out


# ----------------------------------------------------------------------- dependencies


def parse_requirements(text: str) -> list[Dependency]:
    out: list[Dependency] = []
    continuation = False
    for number, line in enumerate(text.splitlines(), start=1):
        is_continuation, continuation = continuation, line.rstrip().endswith("\\")
        if is_continuation:
            continue
        name = requirement_name(line)
        if name:
            out.append(Dependency(name=name, line=number, raw=line.strip()))
    return out


def _find_line(lines: list[str], needle: str, used: set[tuple[int, str]]) -> int | None:
    """The first line containing ``needle`` that this same needle has not claimed yet."""
    for number, line in enumerate(lines, start=1):
        if (number, needle) not in used and needle in line:
            used.add((number, needle))
            return number
    return None


def parse_pyproject(text: str) -> list[Dependency]:
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return []
    project = data.get("project")
    if not isinstance(project, dict):
        return []
    specs: list[Any] = []
    deps = project.get("dependencies")
    if isinstance(deps, list):
        specs.extend(deps)
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        for group in optional.values():
            if isinstance(group, list):
                specs.extend(group)
    lines = text.splitlines()
    used: set[tuple[int, str]] = set()
    out: list[Dependency] = []
    for spec in specs:
        if not isinstance(spec, str):
            continue
        name = requirement_name(spec)
        if name:
            out.append(Dependency(name=name, line=_find_line(lines, spec, used), raw=spec))
    return out


def parse_environment_yml(text: str) -> list[Dependency]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict) or not isinstance(data.get("dependencies"), list):
        return []
    specs: list[str] = []
    for item in data["dependencies"]:
        if isinstance(item, str):
            specs.append(item)
        elif isinstance(item, dict) and isinstance(item.get("pip"), list):
            specs.extend(s for s in item["pip"] if isinstance(s, str))
    lines = text.splitlines()
    used: set[tuple[int, str]] = set()
    out: list[Dependency] = []
    for spec in specs:
        name = requirement_name(spec)
        if name:
            out.append(Dependency(name=name, line=_find_line(lines, spec, used), raw=spec))
    return out


def parse_dependencies(filename: str, text: str) -> list[Dependency]:
    """Dependencies of any supported dependency file; empty for other files."""
    if is_requirements_file(filename):
        return parse_requirements(text)
    if is_pyproject(filename):
        return parse_pyproject(text)
    if is_environment_file(filename):
        return parse_environment_yml(text)
    return []


# ------------------------------------------------------------------------- Dockerfile


def parse_dockerfile(text: str) -> list[DockerFrom]:
    out: list[DockerFrom] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _FROM_RE.match(line)
        if match:
            out.append(DockerFrom(line=number, image=match.group(2), raw=line.strip()))
    return out


def replace_from_image(line: str, new_image: str) -> str:
    """Swap the image reference on one FROM line, keeping flags and any ``AS`` suffix."""
    match = _FROM_RE.match(line)
    if not match:
        return line
    start, end = match.span(2)
    return line[:start] + new_image + line[end:]
