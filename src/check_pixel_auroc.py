import glob, os
home = os.path.expanduser("~")
c = sorted({os.path.realpath(p) for p in glob.glob(f"{home}/results/dinomaly_split_carpet*/**/weights/lightning/model.ckpt", recursive=True)})
assert len(c) == 1, c

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Dinomaly

dm = Folder(name="carpet", root=f"{home}/datasets/MVTec_split/carpet", normal_dir="train_good",
            abnormal_dir="test_defect", normal_test_dir="test_good", mask_dir="test_mask",
            train_batch_size=8, eval_batch_size=8, val_split_mode="same_as_test", seed=42)

for label, drop in [("WITH post-processor", False), ("WITHOUT post-processor", True)]:
    model = Dinomaly.load_from_checkpoint(c[0])
    if drop:
        model.post_processor = None
    try:
        r = Engine(default_root_dir=f"{home}/results/threshold_runs").test(model=model, datamodule=dm)
        print("CHECK", label, {k: round(v, 4) for k, v in r[0].items()})
    except Exception as e:
        print("CHECK", label, "FAILED:", repr(e)[:300])
