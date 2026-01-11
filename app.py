import os
import io
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_file
import tensorflow as tf

# -------------------------
# Custom functions (must match training)
# -------------------------
def dice_coef_multiclass(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = (1, 2)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)

def edge_loss_term(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    e_true = tf.image.sobel_edges(y_true)
    e_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(e_true - e_pred))

def roof_loss(lambda_dice=0.5, lambda_edge=0.1):
    ce_obj = tf.keras.losses.CategoricalCrossentropy()
    def loss(y_true, y_pred):
        ce = ce_obj(y_true, y_pred)
        dice_loss = 1.0 - dice_coef_multiclass(y_true, y_pred)
        e_loss = edge_loss_term(y_true, y_pred)
        return ce + lambda_dice * dice_loss + lambda_edge * e_loss
    return loss

# -------------------------
# Preprocess: build 6-channel features (same as DataGenerator)
# -------------------------
def build_feature_channels(image_bgr, target_size=(256, 256)):
    img_bgr = cv2.resize(image_bgr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_float = image_rgb.astype(np.float32) / 255.0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    mag_max = np.max(mag)
    if mag_max > 0:
        mag = mag / mag_max
    else:
        mag = np.zeros_like(mag, dtype=np.float32)
    mag = mag.astype(np.float32)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0].astype(np.float32) / 179.0
    S = hsv[..., 1].astype(np.float32) / 255.0

    feat = np.stack(
        [rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2], mag, H, S],
        axis=-1,
    )
    return feat, img_bgr

# -------------------------
# Output helpers
# -------------------------
def mask_to_png_bytes(mask_uint8: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise RuntimeError("Failed to encode mask as PNG")
    return buf.tobytes()

def mask_to_base64_raw(mask_uint8: np.ndarray) -> str:
    # raw H*W uint8 bytes -> base64
    return base64.b64encode(mask_uint8.tobytes(order="C")).decode("utf-8")

def mask_to_pixel_matrix(mask_uint8: np.ndarray):
    # 2D list (H x W) of class ids
    return mask_uint8.tolist()

def mask_to_pixel_flat(mask_uint8: np.ndarray):
    # flat list (H*W) of class ids in row-major order
    return mask_uint8.flatten(order="C").astype(int).tolist()

def overlay_mask_on_image(image_bgr: np.ndarray, mask_uint8: np.ndarray, alpha=0.45):
    palette = np.array([
        [0, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [128, 128, 255],
        [255, 128, 128],
        [128, 255, 128],
    ], dtype=np.uint8)

    color_mask = palette[np.clip(mask_uint8, 0, len(palette) - 1)]
    blended = cv2.addWeighted(image_bgr, 1.0, color_mask, float(alpha), 0)
    return blended, palette

# -------------------------
# Flask setup
# -------------------------
app = Flask(__name__)

INPUT_H = int(os.environ.get("INPUT_H", "256"))
INPUT_W = int(os.environ.get("INPUT_W", "256"))

TREE_MODEL_PATH = os.environ.get("TREE_MODEL_PATH", "saved_models2/tree_unet_segmentation.h5")
OBST_MODEL_PATH = os.environ.get("OBST_MODEL_PATH", "saved_models2/obstacle_unet_segmentation.h5")

custom_objects = {
    "loss": roof_loss(lambda_dice=0.5, lambda_edge=0.1),
    "dice_coef_multiclass": dice_coef_multiclass,
}

# Load models once
tree_model = tf.keras.models.load_model(TREE_MODEL_PATH, custom_objects=custom_objects, compile=False)
obst_model = tf.keras.models.load_model(OBST_MODEL_PATH, custom_objects=custom_objects, compile=False)

MODELS = {
    "tree": tree_model,
    "obstruction": obst_model,
    "obstacle": obst_model,  # alias
}

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "tree": TREE_MODEL_PATH,
            "obstruction": OBST_MODEL_PATH,
        }
    })

@app.post("/predict")
def predict():
    """
    Query params:
      - model: tree | obstruction | obstacle (default: tree)
      - response: image | json | pixel (default: image)
          * image: returns PNG (mask or overlay)
          * json : returns compact base64 raw bytes of mask
          * pixel: returns pixel-level JSON (matrix or flat)
      - pixel_format: matrix | flat (default: matrix)   [only when response=pixel]

    Form-data:
      - file: image
      - output: mask | overlay (default: mask)   [only when response=image]
      - alpha: float (default: 0.45)             [only when output=overlay]
    """
    model_key = request.args.get("model", "tree").lower().strip()
    model = MODELS.get(model_key)
    if model is None:
        return jsonify({"error": f"invalid model '{model_key}'. Use one of: {sorted(MODELS.keys())}"}), 400

    response_mode = request.args.get("response", "image").lower().strip()
    if response_mode not in ("image", "json", "pixel"):
        return jsonify({"error": "response must be 'image' or 'json' or 'pixel'"}), 400

    if "file" not in request.files:
        return jsonify({"error": "missing file field"}), 400

    data = request.files["file"].read()
    if not data:
        return jsonify({"error": "empty file"}), 400

    npbuf = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "unable to decode image"}), 400

    feat, resized_bgr = build_feature_channels(img_bgr, target_size=(INPUT_H, INPUT_W))
    x = np.expand_dims(feat.astype(np.float32), axis=0)  # (1,H,W,6)

    y = model.predict(x, verbose=0)  # (1,H,W,C)
    mask = np.argmax(y[0], axis=-1).astype(np.uint8)  # (H,W)

    # -------- JSON mode (compact base64 bytes) --------
    if response_mode == "json":
        return jsonify({
            "model": model_key,
            "input_size": {"h": int(INPUT_H), "w": int(INPUT_W)},
            "mask": {
                "encoding": "raw_u8_base64",
                "shape": [int(mask.shape[0]), int(mask.shape[1])],
                "data": mask_to_base64_raw(mask),
                "dtype": "uint8",
                "order": "C",
            }
        })

    # -------- PIXEL mode (pixel-level JSON) --------
    if response_mode == "pixel":
        pixel_format = request.args.get("pixel_format", "matrix").lower().strip()
        if pixel_format not in ("matrix", "flat"):
            return jsonify({"error": "pixel_format must be 'matrix' or 'flat'"}), 400

        if pixel_format == "matrix":
            pixel_data = mask_to_pixel_matrix(mask)
        else:
            pixel_data = mask_to_pixel_flat(mask)

        return jsonify({
            "model": model_key,
            "input_size": {"h": int(INPUT_H), "w": int(INPUT_W)},
            "mask": {
                "encoding": "pixel_json",
                "shape": [int(mask.shape[0]), int(mask.shape[1])],
                "pixel_format": pixel_format,
                "data": pixel_data,
                "dtype": "uint8",
                "order": "C",
            }
        })

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

    over, _palette = overlay_mask_on_image(resized_bgr, mask, alpha=alpha)
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
