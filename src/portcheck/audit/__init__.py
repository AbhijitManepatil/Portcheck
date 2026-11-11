"""Static audit of a repo for NVIDIA-only dependencies."""

from __future__ import annotations

from portcheck.audit.scanner import load_rules, scan_repo

__all__ = ["load_rules", "scan_repo"]
