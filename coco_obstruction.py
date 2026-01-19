# coco_obstruction.py
import os
import uuid
import numpy as np
import cv2


def r2(x):
    return float(round(float(x), 2))


def _component_confidence(prob_map: np.ndarray, comp_bool: np.ndarray, cls: int) -> float:
    # prob_map: (H,W,C)
    # comp_bool: (H,W) boolean mask for component
    if prob_map is None or prob_map.ndim != 3:
        return 0.0
    if cls < 0 or cls >= prob_map.shape[-1]:
        return 0.0
    n = int(np.sum(comp_bool))
    if n <= 0:
        return 0.0
    return float(np.mean(prob_map[:, :, cls][comp_bool]))


def _contour_area_pixels(cnt) -> float:
    return float(abs(cv2.contourArea(cnt)))


def _perimeter(cnt) -> float:
    return float(cv2.arcLength(cnt, True))


def _is_circle_like(cnt, circularity_thresh: float = 0.72) -> bool:
    # circularity = 4*pi*A / P^2; 1.0 is perfect circle
    A = _contour_area_pixels(cnt)
    P = _perimeter(cnt)
    if P <= 1e-6:
        return False
    circ = (4.0 * np.pi * A) / (P * P)
    return float(circ) >= float(circularity_thresh)


def _circle_from_contour(cnt):
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return float(cx), float(cy), float(r)


def _rect_from_contour(cnt):
    # Oriented rectangle
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)    # 4x2 float
    return box.astype(np.float32)


def build_obstruction_coco_from_mask(
    original_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    save_debug: bool = False,
    debug_dir: str = "debug_obstruction",
    base_name: str = "image",
    background_class_id: int = 0,
    min_component_pixels: int = 50,
    circle_circularity_thresh: float = 0.72,
):
    """
    Returns COCO-like dict with mixed shapes:
      - circle: center_x, center_y, radius
      - rectangle: points (4 corners)
    """
    h_img, w_img = original_bgr.shape[:2]
    h_pred, w_pred = mask_uint8.shape[:2]

    sx = float(w_img) / float(w_pred)
    sy = float(h_img) / float(h_pred)

    coco = {
        "info": {"description": "obstacle AI model"},
        "images": [{"file_name": f"{base_name}.png", "width": int(w_img), "height": int(h_img)}],
        "annotations": [],
    }

    # Debug visualization (optional)
    debug_vis = None
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        debug_vis = original_bgr.copy()

    classes = [c for c in range(int(num_classes)) if c != int(background_class_id)]

    for cls in classes:
        bin_mask = (mask_uint8 == int(cls)).astype(np.uint8) * 255
        if int(cv2.countNonZero(bin_mask)) < int(min_component_pixels):
            continue

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            A = _contour_area_pixels(cnt)
            if A < float(min_component_pixels):
                continue

            comp_mask = np.zeros((h_pred, w_pred), dtype=np.uint8)
            cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
            comp_bool = comp_mask.astype(bool)

            conf = _component_confidence(y_prob, comp_bool, int(cls))

            # Choose shape
            if _is_circle_like(cnt, circularity_thresh=circle_circularity_thresh):
                cx0, cy0, r0 = _circle_from_contour(cnt)

                # scale to original image coords
                cx = cx0 * sx
                cy = cy0 * sy
                r = r0 * (sx + sy) * 0.5

                coco["annotations"].append(
                    {
                        "id": uuid.uuid4().hex,
                        "area": r2(float(A) * sx * sy),
                        "confidence": r2(conf),
                        "shape": "circle",
                        "circle": {"center_x": r2(cx), "center_y": r2(cy), "radius": r2(r)},
                        "class_id": int(cls),
                    }
                )

                if save_debug and debug_vis is not None:
                    cv2.circle(debug_vis, (int(round(cx)), int(round(cy))), max(1, int(round(r))), (0, 255, 0), 2)

            else:
                box = _rect_from_contour(cnt)  # 4x2 in pred coords

                # scale box points to original image coords
                pts = []
                for (x0, y0) in box:
                    pts.append([r2(x0 * sx), r2(y0 * sy)])

                coco["annotations"].append(
                    {
                        "id": uuid.uuid4().hex,
                        "area": r2(float(A) * sx * sy),
                        "confidence": r2(conf),
                        "shape": "rectangle",
                        "rectangle": {"points": pts},
                        "class_id": int(cls),
                    }
                )

                if save_debug and debug_vis is not None:
                    poly = np.array([[p[0], p[1]] for p in pts], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(debug_vis, [poly], True, (255, 0, 0), 2)

    if save_debug and debug_vis is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_geometry.png"), debug_vis)

    return coco
