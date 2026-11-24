import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--category", required=True)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

from lightning.pytorch import seed_everything

seed_everything(args.seed, workers=True)

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Dinomaly

home = os.path.expanduser("~")
datamodule = MVTecAD(
    root=f"{home}/datasets/MVTecAD",
    category=args.category,
    train_batch_size=8,
    eval_batch_size=8,
)
model = Dinomaly()
engine = Engine(
    default_root_dir=f"{home}/results/dinomaly_{args.category}_{args.max_steps}",
    max_steps=args.max_steps,
    max_epochs=-1,
)

t0 = time.time()
engine.fit(model=model, datamodule=datamodule)
print("fit seconds:", round(time.time() - t0))

t1 = time.time()
results = engine.test(model=model, datamodule=datamodule)
print("test seconds:", round(time.time() - t1))
print(results)
