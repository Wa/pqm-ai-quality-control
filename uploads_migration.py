"""Utilities for migrating legacy upload folders into the unified uploads/ tree."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import CONFIG


# Sentinel file to ensure the migration only runs once per environment
_MIGRATION_SENTINEL = Path(CONFIG["directories"]["uploads_root"]) / ".legacy_migration_done.json"
_MIGRATION_COMPLETED = False


def _legacy_pairs() -> Iterable[Tuple[Path, Path]]:
    """Yield (source, destination) directory pairs that require migration."""

    project_root = Path(CONFIG["directories"]["project_root"])
    uploads = CONFIG["uploads"]

    parameters = uploads["parameters"]
    enterprise = uploads["enterprise"]
    special = uploads["special_symbols"]
    completeness = uploads["file_completeness"]
    history = uploads["history_issues"]

    # Legacy parameter folders
    yield project_root / "CP_files", Path(parameters["reference"])
    yield project_root / "reference_files", Path(parameters["reference"])
    yield project_root / "target_files", Path(parameters["target"])
    yield project_root / "graph_files", Path(parameters["graph"])

    # Legacy enterprise folders
    yield project_root / "enterprise_standard_files" / "standards", Path(enterprise["standards"])
    yield project_root / "enterprise_standard_files" / "examined_files", Path(enterprise["examined"])

    # Legacy special symbols folders
    yield project_root / "special_symbols_files" / "reference", Path(special["reference"])
    yield project_root / "special_symbols_files" / "inspected", Path(special["inspected"])

    # Legacy APQP stage folders
    yield project_root / "APQP_files" / "Stage_Initial", Path(completeness["stage_initial"])
    yield project_root / "APQP_files" / "Stage_A", Path(completeness["stage_a"])
    yield project_root / "APQP_files" / "Stage_B", Path(completeness["stage_b"])
    yield project_root / "APQP_files" / "Stage_C", Path(completeness["stage_c"])

    # Legacy history issues folders
    yield project_root / "history_issues_avoidance_files" / "issue_lists", Path(history["issue_lists"])
    yield project_root / "history_issues_avoidance_files" / "target_files", Path(history["target"])


def _move_entry(src: Path, dest_root: Path, log: List[str]) -> None:
    """Move a single file or directory into the destination root."""

    dest_root.mkdir(parents=True, exist_ok=True)
    destination = dest_root / src.name

    if destination.exists():
        if src.is_dir() and destination.is_dir():
            for child in src.iterdir():
                _move_entry(child, destination, log)
            try:
                src.rmdir()
            except OSError:
                pass
            log.append(f"merged directory {src} -> {destination}")
        else:
            log.append(f"skipped {src} (destination exists)")
        return

    shutil.move(str(src), str(destination))
    log.append(f"moved {src} -> {destination}")


def _migrate_pair(source: Path, destination: Path, log: List[str]) -> None:
    """Migrate the contents of a single legacy folder."""

    if not source.exists():
        return

    for item in list(source.iterdir()):
        _move_entry(item, destination, log)

    try:
        source.rmdir()
    except OSError:
        # Leave non-empty folders (e.g., hidden files) in place
        pass


def maybe_run_legacy_uploads_migration() -> List[str]:
    """Execute the migration from legacy directories into uploads/ once."""

    global _MIGRATION_COMPLETED

    if _MIGRATION_COMPLETED:
        return []

    uploads_root = Path(CONFIG["directories"]["uploads_root"])
    uploads_root.mkdir(parents=True, exist_ok=True)

    if _MIGRATION_SENTINEL.exists():
        _MIGRATION_COMPLETED = True
        return []

    log: List[str] = []
    for legacy_src, new_dest in _legacy_pairs():
        _migrate_pair(Path(legacy_src), Path(new_dest), log)

    sentinel_payload: Dict[str, object] = {
        "completed_at": datetime.utcnow().isoformat(timespec="seconds"),
        "actions": log,
    }

    try:
        with _MIGRATION_SENTINEL.open("w", encoding="utf-8") as handle:
            json.dump(sentinel_payload, handle, ensure_ascii=False, indent=2)
    except Exception:
        # Writing the sentinel is best-effort; if it fails we still mark the run to avoid loops
        pass

    _MIGRATION_COMPLETED = True
    return log


__all__ = ["maybe_run_legacy_uploads_migration"]


if __name__ == "__main__":
    moves = maybe_run_legacy_uploads_migration()
    if moves:
        print("Legacy upload directories migrated:")
        for entry in moves:
            print(f" - {entry}")
    else:
        print("No legacy upload directories required migration.")
