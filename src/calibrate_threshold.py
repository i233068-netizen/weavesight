import argparse
import glob
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
args = parser.parse_args()

from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

models = {"dinomaly": Dinomaly, "patchcore": Patchcore}
home = os.path.expanduser("~")
pattern = f"{home}/results/{args.model}_split_{args.category}*/**/weights/lightning/model.ckpt"
ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
assert len(ckpts) == 1, f"expected one checkpoint, found {ckpts}"
ckpt = ckpts[0]

engine = Engine(default_root_dir=f"{home}/results/threshold_runs")


def score(folder):
    preds = engine.predict(model=models[args.model](), data_path=folder, ckpt_path=ckpt)
    out = []
    for b in preds:
        out.extend(b.pred_score.flatten().tolist())
    return out


root = f"{home}/datasets/MVTec_split/{args.category}"
val = score(f"{root}/val_good")
test_good = score(f"{root}/test_good")
test_defect = score(f"{root}/test_defect")

thr = float(np.percentile(val, 99))
fpr = float(np.mean(np.array(test_good) > thr))
recall = float(np.mean(np.array(test_defect) > thr))
print(f"RESULT {args.model} {args.category} thr={thr:.4f} fpr={fpr:.3f} recall={recall:.3f} "
      f"(n_val={len(val)} n_good={len(test_good)} n_defect={len(test_defect)})")

os.makedirs(f"{home}/weavesight/results", exist_ok=True)
with open(f"{home}/weavesight/results/ayesha_threshold_{args.model}_{args.category}.json", "w") as f:
    json.dump(
        {
            "model": args.model,
            "category": args.category,
            "threshold_rule": "99th percentile of val_good scores",
            "threshold": thr,
            "fpr_test_good": fpr,
            "recall_test_defect": recall,
            "val_scores": val,
            "test_good_scores": test_good,
            "test_defect_scores": test_defect,
        },
        f,
        indent=2,
    )
