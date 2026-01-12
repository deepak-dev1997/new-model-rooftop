# coco_tree.py
import os
import uuid
import numpy as np
import cv2


def r2(x):
    return float(round(float(x), 2))


def color_for_class_bgr(cls):
    b = (37 * int(cls)) % 256
    g = (17 * int(cls)) % 256
    r = (97 * int(cls)) % 256
    return (int(b), int(g), int(r))


def colorize_pred_mask_bgr(mask_cls, num_classes):
    h, w = mask_cls.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(int(num_classes)):
        out[mask_cls == cls] = color_for_class_bgr(cls)
    return out


def make_circle_mask(h, w, cx, cy, r):
    m = np.zeros((int(h), int(w)), dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)
    return m


def circle_outside_frac(component_mask, cx, cy, r):
    h, w = component_mask.shape
    c = make_circle_mask(h, w, cx, cy, r)
    circle_area = int(cv2.countNonZero(c))
    if circle_area <= 0:
        return 1.0, 0
    inside = int(cv2.countNonZero(cv2.bitwise_and(c, component_mask)))
    outside = float(circle_area - inside) / float(circle_area)
    return float(outside), int(inside)


def max_radius_with_limit(component_mask, cx, cy, r0, max_outside=0.30, max_scale=2.5, iters=12):
    low = float(max(1.0, r0))
    high = float(max(1.0, r0) * float(max_scale))
    best = low

    for _ in range(int(iters)):
        mid = (low + high) * 0.5
        outside, _ = circle_outside_frac(component_mask, cx, cy, mid)
        if outside <= float(max_outside):
            best = mid
            low = mid
        else:
            high = mid

    return float(best)


def circle_intersection_area(r1, r2, d):
    r1 = float(r1)
    r2 = float(r2)
    d = float(d)

    if r1 <= 0.0 or r2 <= 0.0:
        return 0.0
    if d >= (r1 + r2):
        return 0.0
    if d <= abs(r1 - r2):
        r = min(r1, r2)
        return float(np.pi * r * r)

    a = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)
    b = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)
    a = float(np.clip(a, -1.0, 1.0))
    b = float(np.clip(b, -1.0, 1.0))

    part1 = r1 * r1 * float(np.arccos(a))
    part2 = r2 * r2 * float(np.arccos(b))
    part3 = 0.5 * float(
        np.sqrt(
            max(
                0.0,
                (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2),
            )
        )
    )

    return float(part1 + part2 - part3)


def overlap_frac_smaller(cx1, cy1, r1, cx2, cy2, r2):
    rmin = min(float(r1), float(r2))
    if rmin <= 0.0:
        return 0.0
    d = float(np.hypot(float(cx1) - float(cx2), float(cy1) - float(cy2)))
    inter = circle_intersection_area(r1, r2, d)
    amin = float(np.pi * rmin * rmin)
    return float(inter / amin) if amin > 0.0 else 0.0


def ok_circle(component_mask, cx, cy, r, accepted, max_outside, max_overlap):
    outside, _ = circle_outside_frac(component_mask, cx, cy, r)
    if outside > float(max_outside):
        return False
    for ax, ay, ar in accepted:
        if overlap_frac_smaller(cx, cy, r, ax, ay, ar) > float(max_overlap):
            return False
    return True


def refine_radius(component_mask, cx, cy, r_init, accepted, min_r, max_outside, max_overlap, iters=12):
    r_init = float(r_init)
    min_r = float(min_r)

    if r_init < min_r:
        return None

    if ok_circle(component_mask, cx, cy, r_init, accepted, max_outside, max_overlap):
        return r_init

    lo = min_r
    hi = r_init
    best = None

    for _ in range(int(iters)):
        mid = (lo + hi) * 0.5
        if ok_circle(component_mask, cx, cy, mid, accepted, max_outside, max_overlap):
            best = mid
            lo = mid
        else:
            hi = mid

    return best


def extract_circles_from_component(
    component_mask,
    min_r=2.0,
    max_outside=0.30,
    max_scale=2.5,
    suppress_frac=0.6,
    max_circles=200,
    max_overlap=0.30,
):
    component_mask = (component_mask > 0).astype(np.uint8) * 255
    if int(cv2.countNonZero(component_mask)) == 0:
        return []

    remaining = component_mask.copy()
    accepted = []

    for _ in range(int(max_circles)):
        dt = cv2.distanceTransform(remaining, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dt)

        r0 = float(max_val)
        if r0 < float(min_r):
            break

        cx = float(max_loc[0])
        cy = float(max_loc[1])

        r_candidate = max_radius_with_limit(
            component_mask,
            cx,
            cy,
            r0,
            max_outside=max_outside,
            max_scale=max_scale,
            iters=12,
        )

        r_final = refine_radius(
            component_mask,
            cx,
            cy,
            r_candidate,
            accepted,
            min_r=min_r,
            max_outside=max_outside,
            max_overlap=max_overlap,
            iters=12,
        )

        if r_final is None:
            sup_r = max(1, int(round(r0 * float(suppress_frac))))
            cv2.circle(remaining, (int(round(cx)), int(round(cy))), sup_r, 0, -1)
            continue

        accepted.append((cx, cy, float(r_final)))

        sup_r = max(1, int(round(r0 * float(suppress_frac))))
        cv2.circle(remaining, (int(round(cx)), int(round(cy))), sup_r, 0, -1)

    return accepted


def build_tree_coco_from_mask(
    original_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    save_debug: bool = False,
    debug_dir: str = "debug_tree",
    base_name: str = "image",
):
    h_img, w_img = original_bgr.shape[:2]
    h_pred, w_pred = mask_uint8.shape[:2]

    sx = float(w_img) / float(w_pred)
    sy = float(h_img) / float(h_pred)

    BACKGROUND_CLASS_ID = 0
    MIN_COMPONENT_PIXELS = 500

    # debug-only
    ALPHA_MASK = 0.5
    GEOM_THICKNESS = 2
    MIN_DRAW_RADIUS = 3

    # circle extraction
    MAX_OUTSIDE_FRAC = 0.30
    MAX_RADIUS_SCALE = 2.5
    MIN_DT_RADIUS = 2.0
    SUPPRESS_FRAC = 0.6
    MAX_CIRCLES_PER_COMPONENT = 200
    MAX_OVERLAP_FRAC = 0.30

    coco = {
        "info": {"description": "tree AI model"},
        "images": [{"file_name": f"{base_name}.png", "width": int(w_img), "height": int(h_img)}],
        "annotations": [],
    }

    classes = [c for c in range(int(num_classes)) if c != int(BACKGROUND_CLASS_ID)]

    geom_on_rgb = None
    overlay_mask = None
    pred_color_big = None

    if save_debug:
        pred_color_small = colorize_pred_mask_bgr(mask_uint8.astype(np.int32), num_classes)
        pred_color_big = cv2.resize(pred_color_small, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        overlay_mask = cv2.addWeighted(original_bgr, 1.0 - float(ALPHA_MASK), pred_color_big, float(ALPHA_MASK), 0.0)
        geom_on_rgb = original_bgr.copy()

    for cls in classes:
        bin_mask = (mask_uint8 == int(cls)).astype(np.uint8) * 255
        if int(np.sum(bin_mask > 0)) < int(MIN_COMPONENT_PIXELS):
            continue

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        draw_color = color_for_class_bgr(cls)
        prob_map = y_prob[:, :, int(cls)]

        for cnt in contours:
            component_mask = np.zeros((h_pred, w_pred), dtype=np.uint8)
            cv2.drawContours(component_mask, [cnt], -1, 255, -1)

            comp_pix = int(cv2.countNonZero(component_mask))
            if comp_pix < int(MIN_COMPONENT_PIXELS):
                continue

            circles = extract_circles_from_component(
                component_mask,
                min_r=MIN_DT_RADIUS,
                max_outside=MAX_OUTSIDE_FRAC,
                max_scale=MAX_RADIUS_SCALE,
                suppress_frac=SUPPRESS_FRAC,
                max_circles=MAX_CIRCLES_PER_COMPONENT,
                max_overlap=MAX_OVERLAP_FRAC,
            )

            for cx0, cy0, r_pred in circles:
                outside, inside_pix = circle_outside_frac(component_mask, cx0, cy0, r_pred)
                if outside > float(MAX_OUTSIDE_FRAC):
                    continue
                if inside_pix < int(MIN_COMPONENT_PIXELS):
                    continue

                c_mask = make_circle_mask(h_pred, w_pred, cx0, cy0, r_pred)
                inside_mask = cv2.bitwise_and(c_mask, component_mask)
                inside_bool = inside_mask.astype(bool)

                confidence = float(np.mean(prob_map[inside_bool])) if int(np.sum(inside_bool)) > 0 else 0.0

                cx = float(cx0) * sx
                cy = float(cy0) * sy
                r = float(r_pred) * (sx + sy) * 0.5

                if save_debug and geom_on_rgb is not None:
                    draw_r = max(int(MIN_DRAW_RADIUS), int(round(r)))
                    cv2.circle(
                        geom_on_rgb,
                        (int(round(cx)), int(round(cy))),
                        max(1, draw_r),
                        draw_color,
                        int(GEOM_THICKNESS),
                        lineType=cv2.LINE_AA,
                    )

                coco["annotations"].append(
                    {
                        "id": uuid.uuid4().hex,
                        "area": r2(float(inside_pix) * sx * sy),
                        "confidence": r2(confidence),
                        "shape": "circle",
                        "circle": {"center_x": r2(cx), "center_y": r2(cy), "radius": r2(r)},
                        "class_id": int(cls),
                    }
                )

    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        if pred_color_big is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_pred_mask.png"), pred_color_big)
        if overlay_mask is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_overlay.png"), overlay_mask)
        if geom_on_rgb is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_geometry.png"), geom_on_rgb)

    return coco
