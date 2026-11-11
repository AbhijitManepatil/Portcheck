"""Rule engine and ``scan_repo`` (spec sections 8.1 and 8.2)."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from portcheck.audit import parsers
from portcheck.config import Config, load_config
from portcheck.envelope import InvalidInputError, NotFoundError
from portcheck.paths import get_root, rel_to_root, resolve_in_root
from portcheck.schemas import SEVERITY_ORDER, Finding, ScanResult

RULES_PATH = Path(__file__).with_name("rules.yaml")
_EVIDENCE_MAX = 200
_MATCHER_KINDS = {"file_glob", "import", "dependency", "docker_from", "regex"}


@dataclass(frozen=True)
class Matcher:
    kind: str
    patterns: tuple[str, ...] = ()
    regex: re.Pattern[str] | None = None
    files: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    fix: str | None = None
    ignore_guarded: bool = False
    auto_requires_plain: bool = False
    auto_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    severity: str
    fix: str
    action: str
    plan_title: str
    max_per_file: int | None
    matchers: tuple[Matcher, ...] = field(default=())


def _build_matcher(rule_id: str, raw: dict[str, Any]) -> Matcher:
    kind = raw.get("kind")
    if kind not in _MATCHER_KINDS:
        raise InvalidInputError(f"Rule {rule_id}: unknown matcher kind '{kind}'.")
    regex = None
    if kind in ("regex", "docker_from"):
        flags = re.IGNORECASE if raw.get("ignore_case") else 0
        regex = re.compile(raw["pattern"], flags)
    patterns = raw.get("modules") if kind == "import" else raw.get("patterns")
    return Matcher(
        kind=kind,
        patterns=tuple(patterns or ()),
        regex=regex,
        files=tuple(raw.get("files") or ()),
        exclude=tuple(raw.get("exclude") or ()),
        fix=raw.get("fix"),
        ignore_guarded=bool(raw.get("ignore_guarded", False)),
        auto_requires_plain=bool(raw.get("auto_requires_plain", False)),
        auto_files=tuple(raw.get("auto_files") or ()),
    )


@lru_cache(maxsize=1)
def load_rules() -> tuple[Rule, ...]:
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rules = []
    for raw in data["rules"]:
        rules.append(
            Rule(
                id=raw["id"],
                title=raw["title"],
                severity=raw["severity"],
                fix=raw["fix"],
                action=" ".join(str(raw["action"]).split()),
                plan_title=raw.get("plan_title", raw["title"] + " in {file}"),
                max_per_file=raw.get("max_per_file"),
                matchers=tuple(_build_matcher(raw["id"], m) for m in raw["matchers"]),
            )
        )
    return tuple(rules)


def get_rule(rule_id: str) -> Rule:
    for rule in load_rules():
        if rule.id == rule_id:
            return rule
    raise NotFoundError(f"Unknown rule '{rule_id}'.")


# ------------------------------------------------------------------------------ globs


@lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append(r"(?:.*/)?")
            i += 3
        elif pattern[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(rel_path: str, pattern: str) -> bool:
    """Match a forward-slash relative path against one rule glob."""
    name = rel_path.rsplit("/", 1)[-1]
    if pattern == "@requirements":
        return parsers.is_requirements_file(name)
    if pattern == "@dockerfile":
        return parsers.is_dockerfile(name)
    if "/" not in pattern:
        return fnmatch.fnmatchcase(name, pattern)
    return bool(_glob_regex(pattern).match(rel_path))


def _any_glob(rel_path: str, patterns: tuple[str, ...]) -> bool:
    return any(glob_match(rel_path, p) for p in patterns)


def module_matches(candidate: str, prefix: str) -> bool:
    return candidate == prefix or candidate.startswith(prefix + ".")


# ----------------------------------------------------------------------------- engine


class _FileView:
    """Lazily parsed views of one file."""

    def __init__(self, rel_path: str, text: str | None) -> None:
        self.rel_path = rel_path
        self.name = rel_path.rsplit("/", 1)[-1]
        self.text = text
        self._lines: list[str] | None = None
        self._imports: list[parsers.ImportRef] | None = None
        self._deps: list[parsers.Dependency] | None = None
        self._froms: list[parsers.DockerFrom] | None = None

    @property
    def lines(self) -> list[str]:
        if self._lines is None:
            self._lines = (self.text or "").splitlines()
        return self._lines

    def line_text(self, line: int | None) -> str:
        if line is None or not 1 <= line <= len(self.lines):
            return ""
        return self.lines[line - 1].strip()

    @property
    def imports(self) -> list[parsers.ImportRef]:
        if self._imports is None:
            is_py = self.name.endswith(".py") and self.text is not None
            self._imports = parsers.parse_python_imports(self.text or "") if is_py else []
        return self._imports

    @property
    def deps(self) -> list[parsers.Dependency]:
        if self._deps is None:
            self._deps = (
                parsers.parse_dependencies(self.name, self.text) if self.text is not None else []
            )
        return self._deps

    @property
    def froms(self) -> list[parsers.DockerFrom]:
        if self._froms is None:
            is_docker = parsers.is_dockerfile(self.name) and self.text is not None
            self._froms = parsers.parse_dockerfile(self.text or "") if is_docker else []
        return self._froms


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _EVIDENCE_MAX else text[: _EVIDENCE_MAX - 3] + "..."


def _match(rule: Rule, matcher: Matcher, view: _FileView) -> list[tuple[int | None, str, str]]:
    """Findings of one matcher on one file as ``(line, evidence, fix)``."""
    base_fix = matcher.fix or rule.fix
    if (
        base_fix == "auto"
        and matcher.auto_files
        and not _any_glob(view.rel_path, matcher.auto_files)
    ):
        base_fix = "manual"
    hits: list[tuple[int | None, str, str]] = []

    if matcher.kind == "file_glob":
        if _any_glob(view.rel_path, matcher.patterns):
            hits.append((None, view.rel_path, base_fix))
        return hits
    if view.text is None:
        return hits

    if matcher.kind == "import":
        for imp in view.imports:
            if matcher.ignore_guarded and imp.guarded:
                continue
            if not any(module_matches(c, p) for c in imp.candidates for p in matcher.patterns):
                continue
            fix = base_fix
            if fix == "auto" and matcher.auto_requires_plain and not imp.plain:
                fix = "manual"
            hits.append((imp.line, view.line_text(imp.line) or imp.module, fix))
    elif matcher.kind == "dependency":
        for dep in view.deps:
            if not any(fnmatch.fnmatchcase(dep.name, p) for p in matcher.patterns):
                continue
            if any(fnmatch.fnmatchcase(dep.name, p) for p in matcher.exclude):
                continue
            hits.append((dep.line, view.line_text(dep.line) or dep.raw, base_fix))
    elif matcher.kind == "docker_from":
        assert matcher.regex is not None
        for frm in view.froms:
            if matcher.regex.search(frm.image):
                hits.append((frm.line, frm.raw, base_fix))
    elif matcher.kind == "regex":
        assert matcher.regex is not None
        if _any_glob(view.rel_path, matcher.files):
            for number, line in enumerate(view.lines, start=1):
                if matcher.regex.search(line):
                    hits.append((number, line, base_fix))
    return hits


def scan_file(
    rel_path: str, text: str | None, rules: tuple[Rule, ...] | None = None
) -> list[Finding]:
    """Apply every rule to one file. ``text`` is None when the content is unreadable."""
    view = _FileView(rel_path, text)
    findings: list[Finding] = []
    for rule in rules or load_rules():
        seen: dict[int | None, Finding] = {}
        for matcher in rule.matchers:
            for line, evidence, fix in _match(rule, matcher, view):
                if line in seen:
                    continue  # dedupe on (rule_id, file, line); the first matcher wins
                seen[line] = Finding(
                    rule_id=rule.id,
                    severity=rule.severity,  # type: ignore[arg-type]
                    title=rule.title,
                    file=rel_path,
                    line=line,
                    evidence=_clip(evidence),
                    action=rule.action,
                    fix=fix,  # type: ignore[arg-type]
                )
        ordered = sorted(seen.values(), key=lambda f: (f.line is not None, f.line or 0))
        if rule.max_per_file is not None:
            ordered = ordered[: rule.max_per_file]
        findings.extend(ordered)
    return findings


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (
            SEVERITY_ORDER[f.severity],
            f.file,
            f.line if f.line is not None else 0,
            f.rule_id,
        ),
    )


def _read_text(path: Path, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        # utf-8-sig drops a leading byte-order mark, which would otherwise hide line 1.
        return path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None


def scan_repo(path: str = ".", config: Config | None = None) -> ScanResult:
    """Scan the directory at ``path`` (relative to the root) with every rule."""
    root = get_root()
    target = resolve_in_root(path, root)
    if not target.exists():
        raise NotFoundError(f"Path '{path}' does not exist under the root directory.")
    if not target.is_dir():
        raise InvalidInputError(f"Path '{path}' is not a directory.")
    config = config or load_config(root)
    exclude = set(config.scan.exclude)
    rules = load_rules()

    findings: list[Finding] = []
    files_scanned = 0
    for dirpath, dirnames, filenames in os.walk(target):
        # Skip excluded names and any virtualenv, whatever it is called. (A virtualenv
        # given as the scan path itself is still scanned: that was asked for.)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in exclude and not (Path(dirpath) / d / "pyvenv.cfg").is_file()
        )
        for filename in sorted(filenames):
            if filename in exclude:
                continue
            full = Path(dirpath) / filename
            if not full.is_file():
                continue
            if full.is_symlink():
                # Never read through a link that leaves the root.
                resolved = full.resolve()
                if resolved != root and root not in resolved.parents:
                    continue
            files_scanned += 1
            rel_path = full.relative_to(target).as_posix()
            findings.extend(
                scan_file(rel_path, _read_text(full, config.scan.max_file_bytes), rules)
            )

    findings = sort_findings(findings)
    counts = {"red": 0, "yellow": 0, "green": 0}
    for finding in findings:
        counts[finding.severity] += 1
    grade = next((s for s in ("red", "yellow", "green") if counts[s]), "green")
    return ScanResult(
        root=rel_to_root(target, root),
        files_scanned=files_scanned,
        grade=grade,  # type: ignore[arg-type]
        counts=counts,
        findings=findings,
    )
