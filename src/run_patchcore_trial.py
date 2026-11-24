import os
import time

from lightning.pytorch import seed_everything

seed_everything(42, workers=True)

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

home = os.path.expanduser("~")
datamodule = MVTecAD(
    root=f"{home}/datasets/MVTecAD",
    category="carpet",
    train_batch_size=8,
    eval_batch_size=8,
)
model = Patchcore()
engine = Engine(default_root_dir=f"{home}/results/patchcore_carpet")

t0 = time.time()
engine.fit(model=model, datamodule=datamodule)
print("fit seconds:", round(time.time() - t0))

t1 = time.time()
results = engine.test(model=model, datamodule=datamodule)
print("test seconds:", round(time.time() - t1))
print(results)
