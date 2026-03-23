from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PHASE_DIRS = {
    "phase1": "phase1_brain_teacher",
    "embeddings": "embeddings",
    "graph": "phase2_graph",
    "graphs": "graphs",
    "phase3": "phase3_imputer",
    "phase4": "phase4_confidence",
    "phase5": "phase5_fundus",
    "phase_kd": "phase_kd_benchmark",
    "phase_kd_embed": "phase_kd_embed_benchmark",
    "phase_fitnets": "phase_fitnets_benchmark",
    "phase_rkd": "phase_rkd_benchmark",
    "phase_dkd": "phase_dkd_benchmark",
    "vis": "visualizations",
}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def is_unified_config(cfg: dict) -> bool:
    if not isinstance(cfg, dict):
        return False
    if "run" not in cfg:
        return False
    for key in ("phase1", "phase3", "phase4", "phase5", "graph"):
        if key in cfg:
            return True
    return False


def get_run_root(cfg: dict) -> Path:
    run_cfg = cfg.get("run", {})
    name = run_cfg.get("name")
    if not name:
        raise ValueError("run.name is required for unified config")
    output_root = run_cfg.get("output_root", "logs/runs")
    return Path(output_root) / str(name)


def phase_dir(run_root: Path, phase_key: str) -> Path:
    subdir = PHASE_DIRS.get(phase_key, phase_key)
    return run_root / subdir


def resolve_path(value: Any, default: Path | str) -> Any:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "auto"}:
        return default
    return value


def write_run_config(run_root: Path, config_path: str | Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    target = run_root / "config.yaml"
    target.write_text(Path(config_path).read_text())
    return target
