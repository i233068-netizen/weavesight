"""predict_cae(image, category): same dict contract as app/predict.py. Uses src/cae_common.py."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from cae_common import CAE, SIZE, PATCH, STRIDE, SIGMA, anomaly_map, image_score

HOME = os.path.expanduser("~")
DEVICE = os.environ.get("CAE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_state = {"thr": None, "cat": None, "model": None}


def _thresholds():
    if _state["thr"] is None:
        _state["thr"] = json.load(open(os.path.join(HERE, "thresholds_cae.json")))
    return _state["thr"]


def _load(category):
    if _state["cat"] == category:
        return _state["model"]
    ck = torch.load(f"{HOME}/results/cae_{category}/cae_{category}.pt", map_location=DEVICE)
    assert ck["size"] == SIZE and ck["patch"] == PATCH and ck["stride"] == STRIDE and ck["sigma"] == SIGMA
    model = CAE().to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    _state.update(cat=category, model=model)
    return model


def predict_cae(image, category):
    thr = _thresholds().get(f"cae/{category}")
    if thr is None:
        raise KeyError(f"no thresholds for cae/{category}")
    model = _load(category)
    image = image.convert("RGB")
    W, H = image.size
    if (W, H) != (SIZE, SIZE):  # same resize as load_img in training and evaluation
        image = image.resize((SIZE, SIZE), Image.BILINEAR)
    am = anomaly_map(model, np.asarray(image, dtype=np.uint8).copy(), DEVICE)
    score = image_score(am)
    heat = F.interpolate(torch.from_numpy(am)[None, None], size=(H, W), mode="bilinear",
                         align_corners=False)[0, 0].numpy()
    return {"heatmap": heat, "mask": heat > thr["pixel_threshold"], "score": score,
            "threshold": thr["image_threshold"], "pixel_threshold": thr["pixel_threshold"],
            "verdict": "FAIL" if score > thr["image_threshold"] else "PASS"}
