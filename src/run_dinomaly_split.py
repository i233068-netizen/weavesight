import argparse
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--category", required=True)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

from lightning.pytorch import seed_everything

seed_everything(args.seed, workers=True)

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Dinomaly

home = os.path.expanduser("~")
datamodule = Folder(
    name=args.category,
    root=f"{home}/datasets/MVTec_split/{args.category}",
    normal_dir="train_good",
    abnormal_dir="test_defect",
    normal_test_dir="test_good",
    mask_dir="test_mask",
    train_batch_size=8,
    eval_batch_size=8,
    val_split_mode="same_as_test",
    seed=args.seed,
)
model = Dinomaly()
engine = Engine(
    default_root_dir=f"{home}/results/dinomaly_split_{args.category}_{args.max_steps}",
    max_steps=args.max_steps,
    max_epochs=-1,
)

t0 = time.time()
engine.fit(model=model, datamodule=datamodule)
fit_s = round(time.time() - t0)
print("fit seconds:", fit_s)

t1 = time.time()
results = engine.test(model=model, datamodule=datamodule)
test_s = round(time.time() - t1)
print("test seconds:", test_s)
print(results)

out = {
    "model": "dinomaly",
    "category": args.category,
    "seed": args.seed,
    "steps": args.max_steps,
    "split": "mvtec_seed42.csv",
    "fit_seconds": fit_s,
    "test_seconds": test_s,
    "metrics": results[0],
}
os.makedirs(f"{home}/weavesight/results", exist_ok=True)
path = f"{home}/weavesight/results/ayesha_dinomaly_{args.category}.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("saved", path)
