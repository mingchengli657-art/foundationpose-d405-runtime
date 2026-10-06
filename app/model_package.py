"""Resolve and validate portable object-model packages.

The modeling project owns model generation.  This small module is the only
contract FoundationPose needs: a package directory contains a metadata YAML
file and a mesh path relative to that directory.  Legacy metadata emitted by
the earlier scripts is accepted so existing 001 models remain usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """A resolved model package and its metadata."""

    model_dir: Path
    mesh_path: Path
    model_id: str
    model_frame: str
    units: str
    metadata_path: Path | None
    metadata: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"model metadata must be a mapping: {path}")
    return value


def _metadata_candidates(model_dir: Path) -> list[Path]:
    # New object_modeling output first, then the two legacy formats.
    return [
        model_dir / "model.yaml",
        model_dir / "object_fp_cleaned.yaml",
        model_dir / "model_summary.yaml",
    ]


def resolve_model(model: str | Path) -> ModelSpec:
    """Resolve a model directory or a direct OBJ path.

    Relative mesh references are resolved relative to the metadata file, never
    relative to the caller's current working directory.  This is what makes a
    model directory copyable between the modeling, FoundationPose and robot
    computers.
    """
    supplied = Path(model).expanduser().resolve()
    if supplied.is_file():
        if supplied.suffix.lower() != ".obj":
            raise ValueError(f"model file must be OBJ: {supplied}")
        model_dir = supplied.parent
        metadata_path = None
        metadata: dict[str, Any] = {}
        mesh_path = supplied
    elif supplied.is_dir():
        model_dir = supplied
        metadata_path = next((p for p in _metadata_candidates(model_dir) if p.is_file()), None)
        metadata = _read_yaml(metadata_path) if metadata_path else {}
        mesh_ref = metadata.get("mesh") or metadata.get("foundationpose_mesh")
        if mesh_ref:
            mesh_path = Path(str(mesh_ref))
            if not mesh_path.is_absolute():
                mesh_path = model_dir / mesh_path
            mesh_path = mesh_path.resolve()
        else:
            fallback_names = (
                "object_foundationpose.obj",
                "object_fp_cleaned_centered.obj",
                "object_convex_hull.obj",
            )
            mesh_path = next(
                (model_dir / name for name in fallback_names if (model_dir / name).is_file()),
                model_dir / fallback_names[0],
            ).resolve()
    else:
        raise FileNotFoundError(f"model path does not exist: {supplied}")

    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh referenced by model package does not exist: {mesh_path}")
    units = str(metadata.get("units", "meters")).lower()
    if units not in {"m", "meter", "meters"}:
        raise ValueError(f"FoundationPose requires meter units; package says {units!r}")
    model_id = str(metadata.get("model_id") or model_dir.name)
    model_frame = str(metadata.get("model_frame", "object_model_center"))
    return ModelSpec(
        model_dir=model_dir,
        mesh_path=mesh_path,
        model_id=model_id,
        model_frame=model_frame,
        units="meters",
        metadata_path=metadata_path,
        metadata=metadata,
    )


def inspect_mesh(mesh_path: str | Path) -> dict[str, Any]:
    """Return FoundationPose-relevant mesh diagnostics using Trimesh."""
    import numpy as np
    import trimesh

    path = Path(mesh_path).expanduser().resolve()
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"mesh has no triangle faces: {path}")
    center = mesh.bounds.mean(axis=0)
    radial = mesh.triangles_center - center
    dots = np.einsum("ij,ij->i", mesh.face_normals, radial)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents_m": [float(x) for x in mesh.extents],
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "volume_m3": float(mesh.volume),
        "outward_face_ratio": float(np.mean(dots > 1e-12)),
        "inward_face_ratio": float(np.mean(dots < -1e-12)),
    }


def require_foundationpose_mesh(
    mesh_path: str | Path, *, convex_hull: bool = False
) -> dict[str, Any]:
    """Validate topology/normals before loading a model into FoundationPose.

    ``outward_face_ratio`` is a useful extra check for the convex-hull meshes
    emitted by object_modeling.  It is intentionally not required for a
    general concave CAD mesh: correctly oriented inner surfaces of a cup or
    bowl can point toward the global mesh center by design.
    """
    report = inspect_mesh(mesh_path)
    problems = []
    if not report["watertight"]:
        problems.append("not watertight")
    if not report["winding_consistent"]:
        problems.append("inconsistent face winding")
    if not report["is_volume"] or report["volume_m3"] <= 0.0:
        problems.append("invalid/non-positive volume")
    if convex_hull and report["outward_face_ratio"] < 0.99:
        problems.append("inward-facing normals remain")
    if problems:
        raise ValueError(
            f"invalid FoundationPose mesh {Path(mesh_path).resolve()}: "
            + ", ".join(problems)
        )
    return report
