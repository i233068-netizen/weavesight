"""Stand-in predict(): fake heatmap, meaningless verdicts. Replaced by app/predict.py."""
import numpy as np
from PIL import Image, ImageFilter


def predict(image, model, category):
    gray = image.convert("L")
    g = np.asarray(gray, dtype=np.float32)
    blur = np.asarray(gray.filter(ImageFilter.GaussianBlur(12)), dtype=np.float32)
    diff = np.clip(np.abs(g - blur), 0, 255).astype(np.uint8)
    smooth = Image.fromarray(diff).filter(ImageFilter.GaussianBlur(4))
    heatmap = np.asarray(smooth, dtype=np.float32) / 255.0
    k = max(1, int(heatmap.size * 0.01))
    score = float(np.sort(heatmap.ravel())[-k:].mean())
    threshold = 0.10
    mask = heatmap > 0.08
    return {
        "heatmap": heatmap,
        "mask": mask,
        "score": score,
        "threshold": threshold,
        "verdict": "FAIL" if score > threshold else "PASS",
    }
