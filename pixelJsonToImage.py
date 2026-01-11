import json
import cv2
import numpy as np

# -----------------------
# Config
# -----------------------
JSON_PATH = "response2.json"
IMAGE_PATH = "withTree.png"

OUT_MASK = "pixel_reconstructed_mask_tree.png"
OUT_OVERLAY = "pixel_reconstructed_overlay_tree.png"
ALPHA = 0.45

PALETTE = np.array([
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

# -----------------------
# Load JSON
# -----------------------
with open(JSON_PATH, "r") as f:
    resp = json.load(f)

mask_info = resp["mask"]
H, W = mask_info["shape"]
pixel_format = mask_info.get("pixel_format", "matrix").lower().strip()
data = mask_info["data"]

# -----------------------
# Reconstruct mask
# -----------------------
if pixel_format == "matrix":
    # data is a 2D list: H rows, each W cols
    mask = np.array(data, dtype=np.uint8)
    if mask.shape != (H, W):
        raise RuntimeError(f"Matrix shape mismatch. Expected {(H, W)}, got {mask.shape}")

elif pixel_format == "flat":
    # data is a flat list of length H*W, row-major
    flat = np.array(data, dtype=np.uint8)
    if flat.size != H * W:
        raise RuntimeError(f"Flat size mismatch. Expected {H*W}, got {flat.size}")
    mask = flat.reshape((H, W))

else:
    raise RuntimeError(f"Unsupported pixel_format: {pixel_format}")

# Save mask image
cv2.imwrite(OUT_MASK, mask)

# -----------------------
# Load original image
# -----------------------
img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise RuntimeError("Failed to load image. Check IMAGE_PATH.")

# Resize image to match mask size (because model output is at INPUT_H x INPUT_W)
if img_bgr.shape[0] != H or img_bgr.shape[1] != W:
    img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

# -----------------------
# Overlay
# -----------------------
color_mask = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
overlay = cv2.addWeighted(img_bgr, 1.0, color_mask, ALPHA, 0)

cv2.imwrite(OUT_OVERLAY, overlay)

print("✅ Done")
print(f"Mask saved:    {OUT_MASK}")
print(f"Overlay saved: {OUT_OVERLAY}")
