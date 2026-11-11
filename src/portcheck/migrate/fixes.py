"""Auto-fixes. Each is a pure function from file text to new file text (section 8.4)."""

from __future__ import annotations

import ast
import fnmatch
import re
from collections.abc import Callable

from portcheck.audit import parsers
from portcheck.audit.scanner import Rule, get_rule
from portcheck.config import Config
from portcheck.envelope import ConfigMissingError

_CUDA_INDEX_RE = re.compile(r"(?:https?://)?download\.pytorch\.org/whl/cu\d+/?")
PYNVML_CALL_RE = re.compile(r"\bpynvml\.")

_GUARDED_IMPORT = (
    "try:\n"
    "    import pynvml\n"
    "except ImportError:  # not available on non-NVIDIA machines\n"
    "    pynvml = None\n"
)


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def fix_pc007(text: str, filename: str, config: Config) -> str:
    """Swap every NVIDIA base image on a FROM line for ``target.rocm_image``."""
    rule = get_rule("PC007")
    regexes = [m.regex for m in rule.matchers if m.kind == "docker_from" and m.regex]
    targets = {
        frm.line
        for frm in parsers.parse_dockerfile(text)
        if any(r.search(frm.image) for r in regexes)
    }
    if not targets:
        return text
    image = config.target.rocm_image.strip()
    if not image:
        raise ConfigMissingError(
            "target.rocm_image is empty. Set it in portcheck.yaml to the ROCm PyTorch image "
            "you have verified, then run the step again."
        )
    lines = text.splitlines(keepends=True)
    for number in targets:
        lines[number - 1] = parsers.replace_from_image(lines[number - 1], image)
    return "".join(lines)


def _pc008_name_matches(name: str, rule: Rule) -> bool:
    for matcher in rule.matchers:
        if matcher.kind != "dependency":
            continue
        if any(fnmatch.fnmatchcase(name, p) for p in matcher.patterns) and not any(
            fnmatch.fnmatchcase(name, p) for p in matcher.exclude
        ):
            return True
    return False


def fix_pc008(text: str, filename: str, config: Config) -> str:
    """Drop ``nvidia-*`` requirement lines and repoint CUDA wheel index URLs."""
    rule = get_rule("PC008")
    lines = text.splitlines(keepends=True)
    if parsers.is_requirements_file(filename):
        kept: list[str] = []
        dropping = False  # inside a backslash continuation of a deleted requirement
        continuation = False
        for line in lines:
            is_continuation, continuation = continuation, line.rstrip().endswith("\\")
            if is_continuation:
                if not dropping:
                    kept.append(line)
                continue
            name = parsers.requirement_name(line)
            dropping = bool(name) and _pc008_name_matches(name, rule)
            if not dropping:
                kept.append(line)
        lines = kept
    new_text = "".join(lines)
    if _CUDA_INDEX_RE.search(new_text):
        url = config.target.torch_index_url.strip()
        if not url:
            raise ConfigMissingError(
                "target.torch_index_url is empty. Set it in portcheck.yaml to the PyTorch ROCm "
                "wheel index you have verified, then run the step again."
            )
        new_text = _CUDA_INDEX_RE.sub(lambda _m: url, new_text)
    return new_text


def fix_pc009(text: str, filename: str, config: Config) -> str:
    """Wrap each top-level ``import pynvml`` in a try/except ImportError guard."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return text
    spans = [
        (node.lineno, node.end_lineno or node.lineno)
        for node in tree.body
        if isinstance(node, ast.Import)
        and len(node.names) == 1
        and node.names[0].name == "pynvml"
        and node.names[0].asname is None
    ]
    if not spans:
        return text
    newline = _newline(text)
    lines = text.splitlines(keepends=True)
    for start, end in sorted(spans, reverse=True):
        if lines[start - 1].lstrip() != lines[start - 1] or ";" in "".join(lines[start - 1 : end]):
            continue  # shares a line with another statement; leave for a person
        lines[start - 1 : end] = [_GUARDED_IMPORT.replace("\n", newline)]
    return "".join(lines)


def pynvml_call_lines(text: str) -> list[int]:
    """1-based lines that call ``pynvml.`` (for the step detail of PC009)."""
    out = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("#", "import ", "from ")):
            continue
        if PYNVML_CALL_RE.search(line):
            out.append(number)
    return out


FIXES: dict[str, Callable[[str, str, Config], str]] = {
    "PC007": fix_pc007,
    "PC008": fix_pc008,
    "PC009": fix_pc009,
}
