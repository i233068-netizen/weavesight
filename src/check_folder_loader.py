import os

from anomalib.data import Folder
from anomalib.data.utils.split import TestSplitMode, ValSplitMode

print("ValSplitMode options:", [m.value for m in ValSplitMode])
print("TestSplitMode options:", [m.value for m in TestSplitMode])

home = os.path.expanduser("~")
dm = Folder(
    name="carpet",
    root=f"{home}/datasets/MVTec_split/carpet",
    normal_dir="train_good",
    abnormal_dir="test_defect",
    normal_test_dir="test_good",
    mask_dir="test_mask",
    train_batch_size=8,
    eval_batch_size=8,
    val_split_mode="same_as_test",
    seed=42,
)
dm.setup()
print("train images:", len(dm.train_data))
print("test images:", len(dm.test_data))
