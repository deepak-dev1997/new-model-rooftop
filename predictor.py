# predictor.py
import os
import cv2
import numpy as np
import base64
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
# Encoding helpers (used by app.py)
# -------------------------
def mask_to_base64_raw(mask_uint8: np.ndarray) -> str:
    return base64.b64encode(mask_uint8.tobytes(order="C")).decode("utf-8")


def mask_to_pixel_matrix(mask_uint8: np.ndarray):
    return mask_uint8.tolist()


def mask_to_pixel_flat(mask_uint8: np.ndarray):
    return mask_uint8.flatten(order="C").astype(int).tolist()


def overlay_mask_on_image(image_bgr: np.ndarray, mask_uint8: np.ndarray, alpha=0.45):
    palette = np.array(
        [
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
        ],
        dtype=np.uint8,
    )

    color_mask = palette[np.clip(mask_uint8, 0, len(palette) - 1)]
    blended = cv2.addWeighted(image_bgr, 1.0, color_mask, float(alpha), 0)
    return blended, palette


# -------------------------
# Predictor class: load once, infer many
# -------------------------
class SegmentationPredictor:
    def __init__(
        self,
        input_h: int,
        input_w: int,
        tree_model_path: str,
        obst_model_path: str,
    ):
        self.input_h = int(input_h)
        self.input_w = int(input_w)

        custom_objects = {
            "loss": roof_loss(lambda_dice=0.5, lambda_edge=0.1),
            "dice_coef_multiclass": dice_coef_multiclass,
        }

        self.tree_model_path = tree_model_path
        self.obst_model_path = obst_model_path

        # Load models once
        self.tree_model = tf.keras.models.load_model(self.tree_model_path, custom_objects=custom_objects, compile=False)
        self.obst_model = tf.keras.models.load_model(self.obst_model_path, custom_objects=custom_objects, compile=False)

        self.models = {
            "tree": self.tree_model,
            "obstruction": self.obst_model,
            "obstacle": self.obst_model,  # alias
        }

    def get_model(self, model_key: str):
        return self.models.get((model_key or "tree").lower().strip())

    def infer_from_bgr(self, img_bgr: np.ndarray, model_key: str):
        """
        Returns:
          mask_uint8: (H,W) class ids (uint8)
          prob_map:  (H,W,C) float probs
          resized_bgr: (H,W,3) resized to model input
        """
        model_key = (model_key or "tree").lower().strip()
        model = self.get_model(model_key)
        if model is None:
            raise ValueError(f"invalid model '{model_key}'. Use one of: {sorted(self.models.keys())}")

        feat, resized_bgr = build_feature_channels(img_bgr, target_size=(self.input_h, self.input_w))
        x = np.expand_dims(feat.astype(np.float32), axis=0)  # (1,H,W,6)

        y = model.predict(x, verbose=0)  # (1,H,W,C)
        prob_map = y[0]
        mask = np.argmax(prob_map, axis=-1).astype(np.uint8)

        return mask, prob_map, resized_bgr

    def health_payload(self):
        return {
            "status": "ok",
            "models": {"tree": self.tree_model_path, "obstruction": self.obst_model_path},
            "input_size": {"h": int(self.input_h), "w": int(self.input_w)},
            "available_model_keys": sorted(self.models.keys()),
        }


def from_env():
    input_h = int(os.environ.get("INPUT_H", "256"))
    input_w = int(os.environ.get("INPUT_W", "256"))
    tree_path = os.environ.get("TREE_MODEL_PATH", "Roof-Segmentation-model-Development-main/saved_models2/tree_unet_segmentation.h5")
    obst_path = os.environ.get("OBST_MODEL_PATH", "Roof-Segmentation-model-Development-main/saved_models2/obstacle_unet_segmentation.h5")

    return SegmentationPredictor(
        input_h=input_h,
        input_w=input_w,
        tree_model_path=tree_path,
        obst_model_path=obst_path,
    )
