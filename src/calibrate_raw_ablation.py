import argparse, glob, json, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
args = parser.parse_args()

from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

classes = {"dinomaly": Dinomaly, "patchcore": Patchcore}
home = os.path.expanduser("~")
pattern = f"{home}/results/ablation_{args.model}_{args.category}_5000/**/weights/lightning/model.ckpt"
ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
assert len(ckpts) == 1, f"expected one checkpoint, found {ckpts}"
ckpt = ckpts[0]

model = classes[args.model].load_from_checkpoint(ckpt)
model.post_processor = None  # raw scores: no rescaling, no label-fitted threshold
engine = Engine(default_root_dir=f"{home}/results/threshold_runs")


def score(folder):
    preds = engine.predict(model=model, data_path=folder)
    out = []
    for b in preds:
        out.extend(b.pred_score.flatten().tolist())
    return out


root = f"{home}/datasets/MVTec_split/{args.category}"
val = np.array(score(f"{root}/val_good"))
tg = np.array(score(f"{root}/test_good"))
td = np.array(score(f"{root}/test_defect"))

print(f"RAW RANGE val {val.min():.4f}-{val.max():.4f} | test_good {tg.min():.4f}-{tg.max():.4f} | test_defect {td.min():.4f}-{td.max():.4f}")

try:
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score([0] * len(tg) + [1] * len(td), np.concatenate([tg, td])))
    print(f"IMAGE AUROC from raw scores: {auroc:.4f}")
except Exception as e:
    auroc = None
    print("AUROC check skipped:", e)

# Rules are fixed in advance and use val_good only. P99 is the main rule.
rules = {
    "P99_main": float(np.percentile(val, 99)),
    "val_max_sensitivity": float(val.max()),
    "mean_plus_3std_sensitivity": float(val.mean() + 3 * val.std()),
}
results = {}
for name, thr in rules.items():
    results[name] = {"threshold": thr, "fpr_test_good": float(np.mean(tg > thr)), "recall_test_defect": float(np.mean(td > thr))}
    print(f"{name}: thr={thr:.4f} fpr={results[name]['fpr_test_good']:.3f} recall={results[name]['recall_test_defect']:.3f}")

os.makedirs(f"{home}/weavesight/results", exist_ok=True)
with open(f"{home}/weavesight/results/ayesha_ablation_rawthreshold_{args.model}_{args.category}_5000.json", "w") as f:
    json.dump({"model": args.model, "category": args.category, "scores": "raw (post_processor=None)",
               "image_auroc_raw": auroc, "rules": results,
               "val_scores": val.tolist(), "test_good_scores": tg.tolist(), "test_defect_scores": td.tolist()}, f, indent=2)
