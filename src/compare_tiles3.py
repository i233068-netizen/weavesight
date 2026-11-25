import glob, os, random, re
import numpy as np
from PIL import Image, ImageDraw

home = os.path.expanduser("~")
root = f"{home}/datasets/MVTec_split/aitex_02"
whole = ["0010_006_02", "0011_006_02", "0012_006_02", "0013_006_02", "0014_006_02",
         "0015_006_02", "0030_019_02", "0068_023_02", "0069_023_02", "0070_023_02"]
rng = random.Random(42)


def left_diff(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=float)
    return a[:, :64].mean() - a[:, 64:].mean()


rows = []
for pos in ("t00", "t05"):
    train = sorted(glob.glob(f"{root}/train_good/*_{pos}.png") + glob.glob(f"{root}/val_good/*_{pos}.png"))
    flagged = [f"{root}/test_good/{s}_{pos}.png" for s in whole if os.path.exists(f"{root}/test_good/{s}_{pos}.png")]
    d_tr = [left_diff(p) for p in train]
    d_fl = [left_diff(p) for p in flagged]
    print(f"{pos}: training/val tiles {len(train)}, dark left edge (diff < -8): {sum(d < -8 for d in d_tr)}/{len(train)} | "
          f"flagged-strip tiles {len(flagged)}, dark left edge: {sum(d < -8 for d in d_fl)}/{len(flagged)}")
    rows.append(("train " + pos, rng.sample(train, min(4, len(train)))))
    rows.append(("flagged " + pos, rng.sample(flagged, min(4, len(flagged)))))

W = 256
canvas = Image.new("RGB", (W * 4, len(rows) * (W + 20)), "white")
draw = ImageDraw.Draw(canvas)
for r, (label, paths) in enumerate(rows):
    for c, p in enumerate(paths):
        x, y = c * W, r * (W + 20)
        draw.text((x + 4, y + 4), f"{label}: {os.path.basename(p)}", fill="black")
        canvas.paste(Image.open(p).convert("RGB"), (x, y + 20))
os.makedirs(f"{home}/to_upload", exist_ok=True)
canvas.save(f"{home}/to_upload/compare_tiles3.png")
print("saved", f"{home}/to_upload/compare_tiles3.png")
