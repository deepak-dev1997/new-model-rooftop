import json
import base64
import cv2
import numpy as np

# -----------------------
# Config
# -----------------------
JSON_PATH = "response.json"
IMAGE_PATH = "withTree.png"
OUT_MASK = "reconstructed_mask_tree.png"
OUT_OVERLAY = "reconstructed_overlay_tree.png"
ALPHA = 0.45

# Same palette used by inference service
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
    data = json.load(f)

mask_info = data["mask"]
H, W = mask_info["shape"]

# -----------------------
# Decode mask
# -----------------------
raw_bytes = base64.b64decode(mask_info["data"])
mask = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((H, W))

# Save raw mask (for debugging)
cv2.imwrite(OUT_MASK, mask)

# -----------------------
# Load original image
# -----------------------
img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise RuntimeError("Failed to load input image")

# Resize to match mask if needed
if img_bgr.shape[0] != H or img_bgr.shape[1] != W:
    img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

# -----------------------
# Apply overlay
# -----------------------
color_mask = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
overlay = cv2.addWeighted(img_bgr, 1.0, color_mask, ALPHA, 0)

cv2.imwrite(OUT_OVERLAY, overlay)

print("✅ Mask and overlay reconstructed successfully")
print(f"- Mask saved to     : {OUT_MASK}")
print(f"- Overlay saved to  : {OUT_OVERLAY}")
