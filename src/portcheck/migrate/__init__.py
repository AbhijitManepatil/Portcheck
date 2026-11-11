"""Safe, reviewable edits that make a CUDA repo ready for ROCm."""

from __future__ import annotations

from portcheck.migrate.apply import apply_step
from portcheck.migrate.planner import plan_migration

__all__ = ["apply_step", "plan_migration"]
