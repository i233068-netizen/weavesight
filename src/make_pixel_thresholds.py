import glob, json, os
import numpy as np
home = os.path.expanduser("~")
from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

classes = {"dinomaly": Dinomaly, "patchcore": Patchcore}
cats = ["carpet", "grid", "leather", "tile", "wood"] + [f"aitex_{i:02d}" for i in range(7)]
out_path = f"{home}/weavesight/app/thresholds.json"
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
engine = Engine(default_root_dir=f"{home}/results/threshold_runs")
for name, cls in classes.items():
    for cat in cats:
        pattern = f"{home}/results/{name}_split_{cat}*/**/weights/lightning/model.ckpt"
        ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
        if len(ckpts) != 1:
            print("SKIP", name, cat, ckpts)
            continue
        model = cls.load_from_checkpoint(ckpts[0])
        model.post_processor = None  # raw values only
        preds = engine.predict(model=model, data_path=f"{home}/datasets/MVTec_split/{cat}/val_good")
        scores, pix = [], []
        for b in preds:
            scores.extend(b.pred_score.flatten().tolist())
            pix.append(b.anomaly_map.float().cpu().numpy().ravel())
        pix = np.concatenate(pix)
        res[f"{name}/{cat}"] = {
            "image_threshold": float(np.percentile(scores, 99)),
            "pixel_threshold": float(np.percentile(pix, 99.9)),
            "n_val_images": len(scores),
            "image_rule": "99th percentile of raw val_good image scores",
            "pixel_rule": "99.9th percentile of all raw val_good pixel values",
        }
        json.dump(res, open(out_path, "w"), indent=2)
        print("DONE", name, cat, res[f"{name}/{cat}"]["image_threshold"], res[f"{name}/{cat}"]["pixel_threshold"])
