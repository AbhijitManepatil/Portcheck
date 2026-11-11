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
_NEWLINE_RE = re.compile(r"(\r\n|\n|\r)")
_HEREDOC_RE = re.compile(r"<<-?\s*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\1")
_ARG_RE = re.compile(r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)", re.IGNORECASE)
_VAR_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)(?::?[-+][^}]*)?\}|([A-Za-z_][A-Za-z0-9_]*))")


def split_lines(text: str, keepends: bool = False) -> list[str]:
    """Split on ``\\n``, ``\\r\\n`` and ``\\r`` only.

    ``str.splitlines`` also splits on form feeds and other separators that Python's
    parser, pip and Docker do not treat as line ends, which shifts every line number
    after one. All line handling in portcheck goes through this function instead.
    """
    parts = _NEWLINE_RE.split(text)
    lines = [parts[i] + (parts[i + 1] if keepends else "") for i in range(0, len(parts) - 1, 2)]
    if parts[-1] != "":
        lines.append(parts[-1])
    return lines


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
    image: str  # as written on the FROM line
    raw: str
    resolved: str = ""  # image with ARG defaults substituted
    uses_arg: bool = False  # the image reference contains a build argument


# ----------------------------------------------------------------------------- names


def normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


_URL_SPEC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://|^git\+|^[./\\]|^[A-Za-z]:[\\/]")


def requirement_name(spec: str) -> str | None:
    """The normalised package name at the start of a requirement string, if any."""
    text = spec.strip()
    if not text or text.startswith(("#", "-")):
        return None  # blank, comment, or a pip option such as -r / --index-url
    if _URL_SPEC_RE.match(text):
        return None  # a bare URL, VCS reference or local path has no name to read
    if "::" in text.split("#")[0]:  # conda channel prefix, e.g. nvidia::cudatoolkit
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
    # Lines touched by more than one statement (`import a; b = 1`, also when the
    # other statement only ends on that line).
    shared: set[int] = set()
    if top_level:
        seen: set[int] = set()
        for stmt in nodes:
            span = set(range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1))
            shared |= span & seen
            seen |= span
    for node in nodes:
        if isinstance(node, ast.Import):
            plain = (
                top_level
                and not guarded
                and len(node.names) == 1
                and node.names[0].asname is None
                and node.col_offset == 0
                and not shared & set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
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
    for number, line in enumerate(split_lines(text), start=1):
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
        out: list[ImportRef] = []
        _walk_imports(tree.body, True, False, out)
        return out
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        # Not parseable, or too deeply nested to walk: use the line regex instead.
        return _fallback_imports(text)


# ----------------------------------------------------------------------- dependencies


def continues_requirement(line: str) -> bool:
    """True if the next line continues this one. A comment line never does (as in pip)."""
    return line.rstrip().endswith("\\") and not line.lstrip().startswith("#")


def parse_requirements(text: str) -> list[Dependency]:
    out: list[Dependency] = []
    continuation = False
    for number, line in enumerate(split_lines(text), start=1):
        is_continuation, continuation = continuation, continues_requirement(line)
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
    except (tomllib.TOMLDecodeError, ValueError, RecursionError):
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
    lines = split_lines(text)
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
    except (yaml.YAMLError, RecursionError, ValueError):
        return []
    if not isinstance(data, dict) or not isinstance(data.get("dependencies"), list):
        return []
    specs: list[str] = []
    for item in data["dependencies"]:
        if isinstance(item, str):
            specs.append(item)
        elif isinstance(item, dict) and isinstance(item.get("pip"), list):
            specs.extend(s for s in item["pip"] if isinstance(s, str))
    lines = split_lines(text)
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


def _substitute_args(image: str, args: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return args.get(name, match.group(0))

    return _VAR_RE.sub(repl, image)


def parse_dockerfile(text: str) -> list[DockerFrom]:
    """FROM instructions, skipping heredoc bodies, continuation lines and comments."""
    out: list[DockerFrom] = []
    args: dict[str, str] = {}
    heredoc_end: str | None = None
    continuation = False
    for number, line in enumerate(split_lines(text), start=1):
        stripped = line.strip()
        if heredoc_end is not None:
            if stripped == heredoc_end:
                heredoc_end = None
            continue
        if stripped.startswith("#"):
            continue  # comments do not end a continued instruction
        is_continuation, continuation = continuation, stripped.endswith(("\\", "`"))
        heredoc = _HEREDOC_RE.search(line)
        if heredoc:
            heredoc_end = heredoc.group(2)
        if is_continuation:
            continue
        arg = _ARG_RE.match(line)
        if arg:
            args[arg.group(1)] = arg.group(2).strip("\"'")
            continue
        match = _FROM_RE.match(line)
        if match:
            image = match.group(2)
            out.append(
                DockerFrom(
                    line=number,
                    image=image,
                    raw=stripped,
                    resolved=_substitute_args(image, args),
                    uses_arg="$" in image,
                )
            )
    return out


def replace_from_image(line: str, new_image: str) -> str:
    """Swap the image reference on one FROM line, keeping flags and any ``AS`` suffix."""
    match = _FROM_RE.match(line)
    if not match:
        return line
    start, end = match.span(2)
    return line[:start] + new_image + line[end:]
