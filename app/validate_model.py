#!/usr/bin/env python3
"""Validate an object_modeling model package before FoundationPose startup."""

from __future__ import annotations

import argparse
import json

from model_package import require_foundationpose_mesh, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model package directory or OBJ file")
    args = parser.parse_args()
    spec = resolve_model(args.model)
    convex_hull = spec.metadata.get("mesh_quality", {}).get("topology") == "convex_hull"
    report = require_foundationpose_mesh(spec.mesh_path, convex_hull=convex_hull)
    print(json.dumps({
        "ok": True,
        "model_id": spec.model_id,
        "model_dir": str(spec.model_dir),
        "mesh": str(spec.mesh_path),
        "model_frame": spec.model_frame,
        "units": spec.units,
        "mesh_quality": report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
