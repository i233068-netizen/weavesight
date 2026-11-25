import argparse, glob, os, re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
parser.add_argument("--maps", action="store_true")
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
        out.extend(zip([str(p) for p in b.image_path], b.pred_score.flatten().tolist()))
    return out


root = f"{home}/datasets/MVTec_split/{args.category}"
thr = float(np.percentile([s for _, s in score(f"{root}/val_good")], 99))
status = {}


def add(path, ch):
    m = re.match(r"(.+)_t(\d+)\.png$", os.path.basename(path))
    if m:
        status.setdefault(m.group(1), {})[int(m.group(2))] = ch


for p, s in score(f"{root}/test_good"):
    add(p, "F" if s > thr else ".")
for p, s in score(f"{root}/test_defect"):
    add(p, "D" if s > thr else "m")

strips = sorted(k for k, v in status.items() if any(c in "Dm" for c in v.values()))
tot = adj = 0
for k in strips:
    v = status[k]
    if args.maps:
        print(k, "".join(v.get(i, "-") for i in range(16)))
    for i, c in v.items():
        if c == "F":
            tot += 1
            adj += any(v.get(j) in ("D", "m") for j in (i - 1, i + 1))
print(f"[{args.model} {args.category}] threshold {thr:.4f}; strips with a defect: {len(strips)}")
print(f"  false alarms in defective strips: {tot}; next to a defect tile (+-1): {adj}")
