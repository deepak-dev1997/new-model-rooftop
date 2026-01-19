# app.py
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
import os
import requests
from predictor import from_env, mask_to_base64_raw, mask_to_pixel_matrix, mask_to_pixel_flat, overlay_mask_on_image
from coco_tree import build_tree_coco_from_mask
from coco_obstruction import build_obstruction_coco_from_mask



app = Flask(__name__)

# Load models once at startup
PRED = from_env()

CROPPER_URL =  "http://localhost:5013/crop"
CROPPER_TIMEOUT_SEC = 60

def _decode_image_bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    npbuf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(npbuf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unable to decode image bytes")

    # If cropper returns RGBA/ BGRA PNG, drop alpha (or composite if you want)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # BGRA -> BGR by dropping alpha
    return img

CROPPER_URL = os.environ.get("CROPPER_URL", "http://localhost:5013/crop")
CROPPER_TIMEOUT_SEC = float(os.environ.get("CROPPER_TIMEOUT_SEC", "60"))


def _decode_image_bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    npbuf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(npbuf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unable to decode image bytes")

    # If cropper returns RGBA/ BGRA PNG, drop alpha (or composite if you want)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # BGRA -> BGR by dropping alpha
    return img


def call_crop_service(original_image_bytes: bytes) -> np.ndarray:
    """
    Calls crop API and returns cropped image as BGR numpy array.
    Expects cropper to return image/png (RGBA is fine).
    """
    resp = requests.post(
        CROPPER_URL,
        files={"image": ("image.png", original_image_bytes, "application/octet-stream")},
        timeout=CROPPER_TIMEOUT_SEC,
    )

    if resp.status_code != 200:
        # Cropper sometimes returns JSON error; include it for debugging
        raise RuntimeError(f"cropper failed: HTTP {resp.status_code} - {resp.text[:300]}")

    return _decode_image_bytes_to_bgr(resp.content)


def mask_to_png_bytes(mask_uint8: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise RuntimeError("Failed to encode mask as PNG")
    return buf.tobytes()


@app.get("/health")
def health():
    return jsonify(PRED.health_payload())


@app.post("/predict")
def predict():
    """
    Query params:
      - model: tree | obstruction | obstacle (default: tree)
      - response: image | json | pixel | cocoJson (default: image)
          * image: returns PNG (mask or overlay)
          * json : returns compact base64 raw bytes of mask
          * pixel: returns pixel-level JSON (matrix or flat)
          * cocoJson: returns coco JSON (tree-only for now)

      - pixel_format: matrix | flat (default: matrix) [only when response=pixel]
      - debug: 1/0 (default 0)                        [only when response=cocoJson]
      - name: string (default "image")                [only when response=cocoJson]

    Form-data:
      - file: image
      - output: mask | overlay (default: mask) [only when response=image]
      - alpha: float (default: 0.45)           [only when output=overlay]
    """
    model_key = request.args.get("model", "tree").lower().strip()
    
    response_mode = request.args.get("response", "image").lower().strip()
    if response_mode not in ("image", "json", "pixel", "cocojson"):
        return jsonify({"error": "response must be 'image' or 'json' or 'pixel' or 'cocoJson'"}), 400

    if "file" not in request.files:
        return jsonify({"error": "missing file field"}), 400

    data = request.files["file"].read()
    if not data:
        return jsonify({"error": "empty file"}), 400
        # --- cropMode handling (obstruction only) ---
    crop_mode = request.args.get("cropMode", "false").strip().lower() in ("1", "true", "yes")

    # Use original upload bytes for crop service (so it sees the true image)
    if crop_mode and model_key in ("obstruction", "obstacle"):
        try:
            img_bgr = call_crop_service(data)  # now img_bgr becomes the cropped image
        except Exception as e:
            return jsonify({"error": f"cropMode failed: {str(e)}"}), 500

    npbuf = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "unable to decode image"}), 400

    try:
        mask, prob_map, resized_bgr = PRED.infer_from_bgr(img_bgr, model_key=model_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"inference failed: {str(e)}"}), 500

    input_h, input_w = mask.shape[:2]

    # -------- JSON mode (compact base64 bytes) --------
    if response_mode == "json":
        return jsonify(
            {
                "model": model_key,
                "input_size": {"h": int(input_h), "w": int(input_w)},
                "mask": {
                    "encoding": "raw_u8_base64",
                    "shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "data": mask_to_base64_raw(mask),
                    "dtype": "uint8",
                    "order": "C",
                },
            }
        )

    # -------- PIXEL mode (pixel-level JSON) --------
    if response_mode == "pixel":
        pixel_format = request.args.get("pixel_format", "matrix").lower().strip()
        if pixel_format not in ("matrix", "flat"):
            return jsonify({"error": "pixel_format must be 'matrix' or 'flat'"}), 400

        pixel_data = mask_to_pixel_matrix(mask) if pixel_format == "matrix" else mask_to_pixel_flat(mask)

        return jsonify(
            {
                "model": model_key,
                "input_size": {"h": int(input_h), "w": int(input_w)},
                "mask": {
                    "encoding": "pixel_json",
                    "shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "pixel_format": pixel_format,
                    "data": pixel_data,
                    "dtype": "uint8",
                    "order": "C",
                },
            }
        )

    # -------- COCO JSON mode (tree-only for now) --------
    # -------- COCO JSON mode --------
    if response_mode == "cocojson":
        save_debug = request.args.get("debug", "0").strip().lower() in ("1", "true", "yes")
        base_name = request.args.get("name", "image").strip() or "image"

        if model_key == "tree":
            coco = build_tree_coco_from_mask(
                original_bgr=img_bgr,
                mask_uint8=mask,
                y_prob=prob_map,
                num_classes=int(prob_map.shape[-1]),
                save_debug=save_debug,
                debug_dir="debug_tree",
                base_name=base_name,
            )
            return jsonify({"model": model_key, "input_size": {"h": int(input_h), "w": int(input_w)}, "coco": coco})

        if model_key in ("obstruction", "obstacle"):
            
            coco = build_obstruction_coco_from_mask(
                original_bgr=img_bgr,
                mask_uint8=mask,
                y_prob=prob_map,
                num_classes=int(prob_map.shape[-1]),
                save_debug=save_debug,
                debug_dir="debug_obstruction",
                base_name=base_name,
                background_class_id=0,
                min_component_pixels=50,          # tune as needed
                circle_circularity_thresh=0.72,   # tune as needed
            )
            return jsonify({"model": model_key, "input_size": {"h": int(input_h), "w": int(input_w)}, "coco": coco})

        return jsonify({"error": "cocoJson supported for model=tree,obstruction,obstacle"}), 400


    # -------- IMAGE mode (PNG) --------
    output = request.form.get("output", "mask").lower().strip()
    if output not in ("mask", "overlay"):
        return jsonify({"error": "output must be 'mask' or 'overlay'"}), 400

    if output == "mask":
        return send_file(
            io.BytesIO(mask_to_png_bytes(mask)),
            mimetype="image/png",
            as_attachment=False,
            download_name=f"{model_key}_mask.png",
        )

    try:
        alpha = float(request.form.get("alpha", "0.45"))
    except ValueError:
        return jsonify({"error": "alpha must be a float"}), 400

    over, _ = overlay_mask_on_image(resized_bgr, mask, alpha=alpha)
    ok, buf = cv2.imencode(".png", over)
    if not ok:
        return jsonify({"error": "failed to encode overlay"}), 500

    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{model_key}_overlay.png",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5012, debug=False)
