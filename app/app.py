import os
import sys
import time

import numpy as np
import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.path.exists(os.path.join(HERE, "predict.py")):
    from predict import predict  # real models; errors here should show, not be hidden
    STUB = False
else:
    from predict_stub import predict
    STUB = True

MODELS = ["Dinomaly", "PatchCore", "CAE"]
CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"] + [f"aitex_{i:02d}" for i in range(7)]
SHOW_W = 480


def fit(img):
    w, h = img.size
    return img.resize((SHOW_W, max(1, int(h * SHOW_W / w))))


def heat_overlay(img, heat):
    lo, hi = float(heat.min()), float(heat.max())
    norm = (heat - lo) / (hi - lo + 1e-8)
    r = np.clip(norm * 2, 0, 1)
    g = np.clip(norm * 2 - 1, 0, 1)
    color = np.stack([r, g, np.zeros_like(r)], axis=-1) * 255
    alpha = (0.65 * norm)[..., None]
    base = np.asarray(img, dtype=np.float32)
    return Image.fromarray(((1 - alpha) * base + alpha * color).astype(np.uint8))


def mask_overlay(img, mask):
    base = np.asarray(img, dtype=np.float32).copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    base[mask] = 0.55 * base[mask] + 0.45 * red
    return Image.fromarray(base.astype(np.uint8))


st.set_page_config(page_title="WeaveSight", layout="wide")
st.title("WeaveSight")
st.caption("Annotation-free fabric defect localization")
if STUB:
    st.warning("STUB MODE: the heatmap is fake and the verdicts mean nothing. The real models are not connected yet.")

model = st.sidebar.selectbox("Model", MODELS)
category = st.sidebar.selectbox("Fabric / texture", CATEGORIES)
uploaded = st.file_uploader("Upload a fabric image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    t0 = time.perf_counter()
    res = predict(image, model, category)
    ms = (time.perf_counter() - t0) * 1000

    if res["verdict"] == "FAIL":
        st.error("FAIL: possible defect found. Please check the highlighted region.")
    else:
        st.success("PASS: no defect found.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{res['score']:.4f}")
    c2.metric("Threshold", f"{res['threshold']:.4f}" if res.get("threshold") is not None else "n/a")
    c3.metric("Time", f"{ms:.0f} ms")

    a, b, c = st.columns(3)
    a.image(fit(image), caption="Original")
    b.image(fit(heat_overlay(image, res["heatmap"])), caption="Heatmap overlay")
    c.image(fit(mask_overlay(image, res["mask"])), caption="Predicted defect region")

    st.caption(
        "The verdict depends on a threshold set from a small number of defect-free images. "
        "The system can raise false alarms or miss defects, so an inspector should check the heatmap."
    )
else:
    st.info("Upload an image to start.")
