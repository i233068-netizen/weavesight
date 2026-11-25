import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
parser.add_argument("--n", type=int, default=4)
args = parser.parse_args()

from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

classes = {"dinomaly": Dinomaly, "patchcore": Patchcore}
home = os.path.expanduser("~")
pattern = f"{home}/results/{args.model}_split_{args.category}*/**/weights/lightning/model.ckpt"
ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
assert len(ckpts) == 1, f"expected one checkpoint, found {ckpts}"
model = classes[args.model].load_from_checkpoint(ckpts[0])
resize_size, crop_size = None, None
for t in model.pre_processor.transform.transforms:
    if isinstance(t, v2.Resize):
        resize_size = t.size
    if isinstance(t, v2.CenterCrop):
        crop_size = t.size
model.post_processor = None
engine = Engine(default_root_dir=f"{home}/results/threshold_runs")
root = f"{home}/datasets/MVTec_split/{args.category}"


def as_hw(s):
    return (s, s) if isinstance(s, int) else (s[0], s[-1])


def prep(im, resample, hw):
    if resize_size is not None:
        h, w = as_hw(resize_size)
        im = im.resize((w, h), resample)
    if crop_size is not None:
        ch, cw = as_hw(crop_size)
        im = v2.functional.center_crop(im, [ch, cw])
    if (im.size[1], im.size[0]) != tuple(hw):
        im = im.resize((hw[1], hw[0]), resample)
    return im


def get(folder):
    items = []
    for b in engine.predict(model=model, data_path=folder):
        am = b.anomaly_map
        if am.dim() == 4:
            am = am[:, 0]
        for i in range(am.shape[0]):
            items.append({"path": b.image_path[i], "map": am[i].float().cpu().numpy(), "score": float(b.pred_score[i])})
    return items


val = get(f"{root}/val_good")
thr = float(np.percentile([x["score"] for x in val], 99))
good = get(f"{root}/test_good")
bad = get(f"{root}/test_defect")
fps_all = sorted([x for x in good if x["score"] > thr], key=lambda x: -x["score"])
miss_all = sorted([x for x in bad if x["score"] <= thr], key=lambda x: x["score"])
print(f"RESULT {args.model} {args.category}: threshold {thr:.4f} | false alarms {len(fps_all)}/{len(good)} | missed defects {len(miss_all)}/{len(bad)}")
rows = [("FALSE ALARM", x) for x in fps_all[:args.n]] + [("MISSED DEFECT", x) for x in miss_all[:args.n]]
if not rows:
    print("NO FAILURES TO SHOW")
    raise SystemExit
vmin = min(x["map"].min() for _, x in rows)
vmax = max(x["map"].max() for _, x in rows)
fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
for r, (kind, x) in enumerate(rows):
    name = os.path.basename(x["path"])
    hw = x["map"].shape
    img = prep(Image.open(x["path"]).convert("RGB"), Image.BILINEAR, hw)
    axes[r][0].imshow(img)
    axes[r][0].set_title(f"{kind}\n{name}", fontsize=8)
    axes[r][1].imshow(img)
    axes[r][1].imshow(x["map"], cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    axes[r][1].set_title(f"score {x['score']:.4f} (threshold {thr:.4f})", fontsize=8)
    axes[r][2].imshow(img)
    if kind == "MISSED DEFECT":
        m = prep(Image.open(f"{root}/test_mask/{name}").convert("L"), Image.NEAREST, hw)
        mm = np.array(m) > 127
        if mm.any():
            axes[r][2].contour(mm.astype(float), levels=[0.5], colors="lime", linewidths=1)
        axes[r][2].set_title("annotated defect (green outline)", fontsize=8)
    else:
        axes[r][2].set_title("no defect annotated", fontsize=8)
    for a in axes[r]:
        a.axis("off")
plt.tight_layout()
os.makedirs(f"{home}/weavesight/results/failures", exist_ok=True)
out = f"{home}/weavesight/results/failures/{args.model}_{args.category}.png"
plt.savefig(out, dpi=110)
print("SAVED", out)
