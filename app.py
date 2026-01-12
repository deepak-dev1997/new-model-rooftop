# app.py
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file

from predictor import from_env, mask_to_base64_raw, mask_to_pixel_matrix, mask_to_pixel_flat, overlay_mask_on_image
from coco_tree import build_tree_coco_from_mask


app = Flask(__name__)

# Load models once at startup
PRED = from_env()


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
    if response_mode == "cocojson":
        if model_key != "tree":
            return jsonify(
                {
                    "error": "cocoJson mode is currently supported only for model=tree. Obstruction flow will be added later."
                }
            ), 400

        save_debug = request.args.get("debug", "0").strip().lower() in ("1", "true", "yes")
        base_name = request.args.get("name", "image").strip() or "image"

        coco = build_tree_coco_from_mask(
            original_bgr=img_bgr,  # original image size
            mask_uint8=mask,       # model input size mask
            y_prob=prob_map,
            num_classes=int(prob_map.shape[-1]),
            save_debug=save_debug,
            debug_dir="debug_tree",
            base_name=base_name,
        )

        return jsonify({"model": model_key, "input_size": {"h": int(input_h), "w": int(input_w)}, "coco": coco})

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
