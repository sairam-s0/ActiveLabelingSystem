"""Optional Segment Anything support for ALS detections."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_SAM_MODEL = "sam_b.pt"


def sam_available() -> bool:
    try:
        from ultralytics import SAM  # noqa: F401

        return True
    except Exception:
        return False


def _resolve_sam_path(model_path: str = DEFAULT_SAM_MODEL) -> str:
    """Search for the SAM model file in common locations."""
    p = Path(model_path)
    if p.exists():
        return str(p)

    # Check relative to this script's directory (src/)
    script_dir = Path(__file__).resolve().parent.parent
    candidate = script_dir / model_path
    if candidate.exists():
        return str(candidate)

    # Check relative to project root (two levels up from core/)
    project_root = script_dir.parent
    candidate = project_root / model_path
    if candidate.exists():
        return str(candidate)

    # Return original path and let ultralytics handle/download it
    return model_path


@lru_cache(maxsize=1)
def _load_sam(model_path: str = DEFAULT_SAM_MODEL) -> Any:
    from ultralytics import SAM

    resolved = _resolve_sam_path(model_path)
    print(f"[SAM] Loading model from: {resolved}")
    return SAM(resolved)


def _simplify_polygon(points: list[list[float]], max_points: int = 80) -> list[list[float]]:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def segment_boxes(
    image_path: str,
    detections: list[dict],
    model_path: str = DEFAULT_SAM_MODEL,
) -> list[dict]:
    """Attach SAM polygon metadata to YOLO detections when possible.

    The function is deliberately best-effort. Missing SAM weights, unsupported
    environments, and occasional mask failures should not block box labeling.
    """

    if not detections:
        return detections

    try:
        sam_model = _load_sam(model_path)
        boxes = [det.get("bbox", []) for det in detections]
        if not all(len(box) == 4 for box in boxes):
            return detections

        results = sam_model(str(Path(image_path)), bboxes=boxes, verbose=False)
        if not results:
            return detections

        masks = getattr(results[0], "masks", None)
        if masks is None or getattr(masks, "xy", None) is None:
            return detections

        mask_data = getattr(masks, "data", None)
        for idx, (det, polygon) in enumerate(zip(detections, masks.xy)):
            points = [[float(x), float(y)] for x, y in polygon.tolist()]
            if len(points) < 3:
                continue
            det["segmentation"] = [_simplify_polygon(points)]
            det["mask_area"] = float(mask_data[idx].sum().item()) if mask_data is not None and idx < len(mask_data) else 0.0
            det["mask_source"] = "sam"
    except Exception as exc:
        print(f"[SAM] Segmentation skipped: {exc}")

    return detections


def segment_point(
    image_path: str,
    x: float,
    y: float,
    model_path: str = DEFAULT_SAM_MODEL,
) -> dict | None:
    """Return one SAM mask from a positive point prompt when possible.

    Tries all returned masks and picks the best one (smallest area that
    still contains the click point). Returns tight bounding box with
    segmentation polygon covering all edges.
    """

    try:
        sam_model = _load_sam(model_path)
        print(f"[SAM] Point segmentation at ({x:.0f}, {y:.0f}) on {Path(image_path).name}")
        results = sam_model(
            str(Path(image_path)),
            points=[[float(x), float(y)]],
            labels=[1],
            verbose=False,
        )
        if not results:
            print("[SAM] No results returned")
            return None

        masks = getattr(results[0], "masks", None)
        if masks is None:
            print("[SAM] No masks in result")
            return None

        xy_data = getattr(masks, "xy", None)
        if xy_data is None or len(xy_data) == 0:
            print("[SAM] No xy polygon data in masks")
            return None

        mask_data = getattr(masks, "data", None)

        # Try all masks - pick the best one (smallest area containing click point)
        best_result = None
        best_area = float("inf")

        for idx, polygon in enumerate(xy_data):
            points = [[float(px), float(py)] for px, py in polygon.tolist()]
            if len(points) < 3:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            area = (x2 - x1) * (y2 - y1)

            if area <= 0:
                continue

            # Check if click point is roughly within the bbox (with margin)
            margin = max(20.0, min(x2 - x1, y2 - y1) * 0.1)
            in_bbox = (x1 - margin <= x <= x2 + margin) and (y1 - margin <= y <= y2 + margin)

            if in_bbox and area < best_area:
                best_area = area
                mask_area = 0.0
                if mask_data is not None and idx < len(mask_data):
                    mask_area = float(mask_data[idx].sum().item())
                best_result = {
                    "bbox": [x1, y1, x2, y2],
                    "segmentation": [_simplify_polygon(points)],
                    "mask_area": mask_area,
                    "mask_source": "sam_point",
                }

        # Fallback: if no mask contains click point, use the first valid mask
        if best_result is None:
            for idx, polygon in enumerate(xy_data):
                points = [[float(px), float(py)] for px, py in polygon.tolist()]
                if len(points) < 3:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                mask_area = 0.0
                if mask_data is not None and idx < len(mask_data):
                    mask_area = float(mask_data[idx].sum().item())
                best_result = {
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "segmentation": [_simplify_polygon(points)],
                    "mask_area": mask_area,
                    "mask_source": "sam_point",
                }
                break

        if best_result:
            bbox = best_result["bbox"]
            print(f"[SAM] Point segment result: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}], "
                  f"area={best_result['mask_area']:.0f}")
        else:
            print("[SAM] No valid mask found")

        return best_result
    except Exception as exc:
        import traceback
        print(f"[SAM] Point segmentation error: {exc}")
        traceback.print_exc()
        return None

