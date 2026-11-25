import glob, os, re
import numpy as np
from PIL import Image

home = os.path.expanduser("~")
root = f"{home}/datasets/MVTec_split/aitex_02"
whole = {"0010_006_02", "0011_006_02", "0012_006_02", "0013_006_02", "0014_006_02",
         "0015_006_02", "0030_019_02", "0068_023_02", "0069_023_02", "0070_023_02"}


def stats(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=float)
    return a.mean(), a.std()


def strip_id(p):
    return re.sub(r"_t\d+\.png$", "", os.path.basename(p))


train = [stats(p) for d in ("train_good", "val_good") for p in glob.glob(f"{root}/{d}/*.png")]
tm = np.array([m for m, _ in train])
ts = np.array([s for _, s in train])
print("train+val tiles (n=%d): brightness min/median/max %.1f / %.1f / %.1f | contrast min/median/max %.1f / %.1f / %.1f"
      % (len(tm), tm.min(), np.median(tm), tm.max(), ts.min(), np.median(ts), ts.max()))

defective = {strip_id(p) for p in glob.glob(f"{root}/test_defect/*.png")}
per = {}
for d in ("test_good", "test_defect"):
    for p in glob.glob(f"{root}/{d}/*.png"):
        k = strip_id(p)
        if k in defective:
            per.setdefault(k, []).append(stats(p))
rows = {k: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for k, v in per.items()}

for label, keys in (("10 whole-strip flagged", sorted(k for k in rows if k in whole)),
                    ("other defective strips (n=%d)" % len([k for k in rows if k not in whole]),
                     sorted(k for k in rows if k not in whole))):
    b = np.array([rows[k][0] for k in keys])
    c = np.array([rows[k][1] for k in keys])
    print("%s: brightness min/median/max %.1f / %.1f / %.1f | contrast min/median/max %.1f / %.1f / %.1f"
          % (label, b.min(), np.median(b), b.max(), c.min(), np.median(c), c.max()))
print("flagged strips:")
for k in sorted(k for k in rows if k in whole):
    print("  ", k, "brightness %.1f contrast %.1f" % rows[k])
