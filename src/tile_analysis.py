import argparse, glob, json, os, re
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
args = parser.parse_args()

from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

classes = {"dinomaly": Dinomaly, "patchcore": Patchcore}
home = os.path.expanduser("~")
pattern = f"{home}/results/{args.model}_split_{args.category}*/**/weights/lightning/model.ckpt"
ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
assert len(ckpts) == 1, f"expected one checkpoint, found {ckpts}"
model = classes[args.model].load_from_checkpoint(ckpts[0])
model.post_processor = None
engine = Engine(default_root_dir=f"{home}/results/threshold_runs")


def score(folder):
    out = []
    for b in engine.predict(model=model, data_path=folder):
        paths = [str(p) for p in b.image_path]
        out.extend(zip(paths, b.pred_score.flatten().tolist()))
    return out


root = f"{home}/datasets/MVTec_split/{args.category}"
val = np.array([s for _, s in score(f"{root}/val_good")])
thr = float(np.percentile(val, 99))
good = score(f"{root}/test_good")
defect = score(f"{root}/test_defect")


def tile_no(path):
    m = re.search(r"_t(\d+)\.png$", os.path.basename(path))
    return int(m.group(1)) if m else None


def source(path):
    parts = os.path.basename(path).split("_")
    return "defect-free strip" if len(parts) > 1 and parts[1] == "000" else "defective strip"


def rate(flags):
    return f"{sum(flags)}/{len(flags)}" if flags else "0/0"


fa_pos = {"t00": [], "t01-t14": [], "t15": [], "unknown": []}
fa_src = {"defect-free strip": [], "defective strip": []}
for p, s in good:
    n = tile_no(p)
    key = "unknown" if n is None else ("t00" if n == 0 else ("t15" if n == 15 else "t01-t14"))
    fa_pos[key].append(s > thr)
    fa_src[source(p)].append(s > thr)

bins = {"<0.1% of pixels": [], "0.1-1%": [], ">=1%": []}
areas = []
for p, s in defect:
    mpath = p.replace("/test_defect/", "/test_mask/")
    area = float((np.array(Image.open(mpath).convert("L")) > 127).mean())
    areas.append(area)
    key = "<0.1% of pixels" if area < 0.001 else ("0.1-1%" if area < 0.01 else ">=1%")
    bins[key].append(s > thr)

print(f"[{args.model} {args.category}] threshold {thr:.4f}, val tiles {len(val)}")
print("  false alarms by tile position:", {k: rate(v) for k, v in fa_pos.items() if v})
print("  false alarms by strip type:   ", {k: rate(v) for k, v in fa_src.items()})
print("  defects caught by defect area:", {k: rate(v) for k, v in bins.items()}, "| median area %.4f%%" % (100 * float(np.median(areas))))
os.makedirs(f"{home}/weavesight/results", exist_ok=True)
json.dump({"model": args.model, "category": args.category, "threshold": thr,
           "false_alarms_by_position": {k: [int(sum(v)), len(v)] for k, v in fa_pos.items()},
           "false_alarms_by_strip_type": {k: [int(sum(v)), len(v)] for k, v in fa_src.items()},
           "defects_caught_by_area": {k: [int(sum(v)), len(v)] for k, v in bins.items()}},
          open(f"{home}/weavesight/results/ayesha_tilepos_{args.model}_{args.category}.json", "w"), indent=2)
