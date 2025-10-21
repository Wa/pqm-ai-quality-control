"""Workflow surface definitions shared across tabs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class WorkflowPaths:
    """Concrete filesystem layout for a tab workflow."""

    standards_dir: str
    examined_dir: str
    output_root: str
    standards_txt_dir: str
    examined_txt_dir: str
    initial_results_dir: str
    final_results_dir: str
    checkpoint_dir: str


@dataclass(frozen=True)
class WorkflowSurface:
    """Declarative description of a tab's workflow footprint."""

    slug: str
    output_subdir: str
    chunk_prompt_prefix: str
    standards_dir_key: str
    examined_dir_key: str
    warmup_prompt: Optional[str] = None
    initial_results_name: str = "initial_results"
    final_results_name: str = "final_results"
    standards_txt_name: str = "standards_txt"
    examined_txt_name: str = "examined_txt"
    checkpoint_name: str = "checkpoint"

    def knowledge_base_name(self, session_id: str) -> str:
        """Return the knowledge-base slug for a session."""

        return f"{session_id}_{self.slug}".strip("_")

    def build_chunk_prompt(self, chunk: str) -> str:
        """Construct the comparison prompt for a document chunk."""

        return f"{self.chunk_prompt_prefix}{chunk}" if chunk else self.chunk_prompt_prefix

    def prepare_paths(self, session_dirs: Mapping[str, str]) -> WorkflowPaths:
        """Ensure the workflow directories exist and return their paths."""

        generated_dir = session_dirs.get("generated") or session_dirs.get("generated_files")
        if not generated_dir:
            raise KeyError("Session directories missing generated output root")

        output_root = os.path.join(generated_dir, self.output_subdir)
        standards_txt_dir = os.path.join(output_root, self.standards_txt_name)
        examined_txt_dir = os.path.join(output_root, self.examined_txt_name)
        initial_results_dir = os.path.join(output_root, self.initial_results_name)
        final_results_dir = os.path.join(output_root, self.final_results_name)
        checkpoint_dir = os.path.join(output_root, self.checkpoint_name)

        for path in (output_root, standards_txt_dir, examined_txt_dir, initial_results_dir, final_results_dir, checkpoint_dir):
            os.makedirs(path, exist_ok=True)

        standards_dir = session_dirs.get(self.standards_dir_key, "")
        if standards_dir:
            os.makedirs(standards_dir, exist_ok=True)
        examined_dir = session_dirs.get(self.examined_dir_key, "")
        if examined_dir:
            os.makedirs(examined_dir, exist_ok=True)

        return WorkflowPaths(
            standards_dir=standards_dir,
            examined_dir=examined_dir,
            output_root=output_root,
            standards_txt_dir=standards_txt_dir,
            examined_txt_dir=examined_txt_dir,
            initial_results_dir=initial_results_dir,
            final_results_dir=final_results_dir,
            checkpoint_dir=checkpoint_dir,
        )


__all__ = ["WorkflowPaths", "WorkflowSurface"]
