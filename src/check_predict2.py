import argparse, glob, os, sys
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
args = parser.parse_args()

home = os.path.expanduser("~")
sys.path.insert(0, f"{home}/weavesight/app")
from predict import predict, _thresholds
from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

cls = {"dinomaly": Dinomaly, "patchcore": Patchcore}[args.model]
pattern = f"{home}/results/{args.model}_split_{args.category}*/**/weights/lightning/model.ckpt"
ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
assert len(ckpts) == 1, ckpts
model = cls.load_from_checkpoint(ckpts[0])
model.post_processor = None
engine = Engine(default_root_dir=f"{home}/results/threshold_runs")
root = f"{home}/datasets/MVTec_split/{args.category}"
thr = _thresholds()[f"{args.model}/{args.category}"]["image_threshold"]
print("app threshold %.5f" % thr)


def score(folder):
    out = {}
    for b in engine.predict(model=model, data_path=folder):
        for p, s in zip(b.image_path, b.pred_score.flatten().tolist()):
            out[str(p)] = s
    return out


del_model = model
todo = []
for folder, k in (("test_good", 6), ("test_defect", 4)):
    sc = score(f"{root}/{folder}")
    keys = sorted(sc)
    idx = np.linspace(0, len(keys) - 1, k).astype(int)
    todo += [(folder, keys[i], sc[keys[i]]) for i in idx]
worst = 0.0
print("%-12s %-26s %10s %10s %10s %s" % ("folder", "file", "predict()", "engine", "abs diff", "verdicts"))
for folder, p, ref in todo:
    out = predict(Image.open(p), args.model, args.category)
    diff = abs(out["score"] - ref)
    worst = max(worst, diff)
    print("%-12s %-26s %10.5f %10.5f %10.6f %s vs %s" % (folder, os.path.basename(p), out["score"], ref, diff, out["verdict"], "FAIL" if ref > thr else "PASS"))
print("worst abs difference: %.6f" % worst)
