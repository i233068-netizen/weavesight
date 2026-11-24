import os
import time

from lightning.pytorch import seed_everything

seed_everything(42, workers=True)

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Dinomaly

home = os.path.expanduser("~")
datamodule = MVTecAD(
    root=f"{home}/datasets/MVTecAD",
    category="carpet",
    train_batch_size=8,
    eval_batch_size=8,
)
model = Dinomaly()
engine = Engine(default_root_dir=f"{home}/results/dinomaly_carpet_1000_test")

t0 = time.time()
results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=f"{home}/checkpoints/dinomaly_carpet_1000/model.ckpt",
)
print("test seconds:", round(time.time() - t0))
print(results)
