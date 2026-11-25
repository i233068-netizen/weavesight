"""Real predict(): Dinomaly / PatchCore. Contract: see app/app.py."""
import glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_state = {"thr": None, "key": None, "model": None}


def _thresholds():
    if _state["thr"] is None:
        _state["thr"] = json.load(open(os.path.join(HERE, "thresholds.json")))
    return _state["thr"]


def _load(name, category):
    key = f"{name}/{category}"
    if _state["key"] == key:
        return _state["model"]
    _state.update(key=None, model=None)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    from anomalib.models import Dinomaly, Patchcore
    cls = {"dinomaly": Dinomaly, "patchcore": Patchcore}[name]
    pattern = f"{HOME}/results/{name}_split_{category}*/**/weights/lightning/model.ckpt"
    ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
    if len(ckpts) != 1:
        raise FileNotFoundError(f"expected one checkpoint for {key}, found {ckpts}")
    model = cls.load_from_checkpoint(ckpts[0])
    model.post_processor = None  # raw scores, same as our calibration
    model.eval().to(DEVICE)
    _state.update(key=key, model=model)
    return model


def _hw(s):
    return (s, s) if isinstance(s, int) else (s[0], s[-1])


def predict(image, model, category):
    name = model.lower()
    if name not in ("dinomaly", "patchcore"):
        raise NotImplementedError(f"{model} is not connected yet")
    thr = _thresholds().get(f"{name}/{category}")
    if thr is None:
        raise KeyError(f"no thresholds for {name}/{category}")
    m = _load(name, category)

    image = image.convert("RGB")
    W, H = image.size
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    resize = None
    for t in m.pre_processor.transform.transforms:
        if isinstance(t, v2.Resize):
            resize = _hw(t.size)
    with torch.no_grad():
        x = m.pre_processor.transform(x)  # same transform as training and testing
        out = m.model(x.unsqueeze(0).to(DEVICE))
    am = out.anomaly_map.float().cpu()
    if am.dim() == 4:
        am = am[:, 0]
    am = am[0]
    score = float(out.pred_score.flatten()[0])

    ah, aw = am.shape
    rh, rw = resize if resize else (ah, aw)
    if (ah, aw) == (rh, rw):
        canvas = am
    else:  # the model only sees the centre crop; the border stays at a low fill value
        fill = min(float(am.min()), 0.5 * thr["pixel_threshold"])
        canvas = torch.full((rh, rw), fill)
        top, left = (rh - ah) // 2, (rw - aw) // 2
        canvas[top:top + ah, left:left + aw] = am
    heat = F.interpolate(canvas[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()

    return {
        "heatmap": heat,
        "mask": heat > thr["pixel_threshold"],
        "score": score,
        "threshold": thr["image_threshold"],
        "pixel_threshold": thr["pixel_threshold"],
        "verdict": "FAIL" if score > thr["image_threshold"] else "PASS",
    }
