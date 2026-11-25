import os

from anomalib.engine import Engine
from anomalib.models import Dinomaly

home = os.path.expanduser("~")
ckpt = f"{home}/results/dinomaly_split_carpet_1000/Dinomaly/carpet/v0/weights/lightning/model.ckpt"
data = f"{home}/datasets/MVTec_split/carpet/val_good"

engine = Engine(default_root_dir=f"{home}/results/peek")
preds = engine.predict(model=Dinomaly(), data_path=data, ckpt_path=ckpt)

print("RESULT type:", type(preds), "len:", len(preds))
b = preds[0]
print("BATCH type:", type(b))
print("BATCH fields:", [a for a in dir(b) if not a.startswith("_")])
print("pred_score:", getattr(b, "pred_score", "missing"))
am = getattr(b, "anomaly_map", None)
print("anomaly_map shape:", getattr(am, "shape", "missing"))
