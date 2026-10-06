#!/usr/bin/env python3
"""Depth-based pose validation independent of FoundationPose's neural score."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthConsistencyResult:
    passed: bool
    reason: str
    rendered_pixels: int
    valid_pixels: int
    coverage: float
    median_residual_m: float
    p90_residual_m: float
    inlier_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_depth_consistency(
    observed_depth_m: np.ndarray,
    rendered_depth_m: np.ndarray,
    *,
    inlier_threshold_m: float = 0.025,
    min_inlier_ratio: float = 0.45,
    max_median_residual_m: float = 0.035,
    min_coverage: float = 0.70,
    erosion_px: int = 2,
    min_rendered_pixels: int = 250,
) -> DepthConsistencyResult:
    """Compare rendered model depth with aligned measured depth.

    Only pixels safely inside the rendered silhouette are used.  Eroding the
    silhouette prevents ordinary rasterization/calibration errors at object
    boundaries from dominating the score.
    """
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    rendered = np.asarray(rendered_depth_m, dtype=np.float32)
    if observed.shape != rendered.shape or observed.ndim != 2:
        raise ValueError(
            f"depth shape mismatch: observed={observed.shape}, rendered={rendered.shape}"
        )

    rendered_mask = np.isfinite(rendered) & (rendered > 0.001)
    if erosion_px > 0 and rendered_mask.any():
        kernel_size = 2 * int(erosion_px) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        rendered_mask = cv2.erode(
            rendered_mask.astype(np.uint8), kernel, iterations=1
        ).astype(bool)

    rendered_pixels = int(rendered_mask.sum())
    if rendered_pixels < int(min_rendered_pixels):
        return DepthConsistencyResult(
            False,
            f"rendered object too small ({rendered_pixels} px)",
            rendered_pixels,
            0,
            0.0,
            float("inf"),
            float("inf"),
            0.0,
        )

    valid_observed = np.isfinite(observed) & (observed > 0.001)
    comparison_mask = rendered_mask & valid_observed
    valid_pixels = int(comparison_mask.sum())
    coverage = valid_pixels / rendered_pixels
    if valid_pixels == 0:
        return DepthConsistencyResult(
            False,
            "no valid D405 depth inside rendered object",
            rendered_pixels,
            0,
            coverage,
            float("inf"),
            float("inf"),
            0.0,
        )

    residual = np.abs(observed[comparison_mask] - rendered[comparison_mask])
    median_residual_m = float(np.median(residual))
    p90_residual_m = float(np.percentile(residual, 90))
    inlier_ratio = float(np.mean(residual <= float(inlier_threshold_m)))

    failures: list[str] = []
    if coverage < min_coverage:
        failures.append(f"coverage {coverage:.2f} < {min_coverage:.2f}")
    if inlier_ratio < min_inlier_ratio:
        failures.append(f"inliers {inlier_ratio:.2f} < {min_inlier_ratio:.2f}")
    if median_residual_m > max_median_residual_m:
        failures.append(
            f"median {median_residual_m * 1000.0:.1f}mm > "
            f"{max_median_residual_m * 1000.0:.1f}mm"
        )

    return DepthConsistencyResult(
        not failures,
        "; ".join(failures),
        rendered_pixels,
        valid_pixels,
        coverage,
        median_residual_m,
        p90_residual_m,
        inlier_ratio,
    )

