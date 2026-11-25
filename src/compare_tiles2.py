import glob, os, re
import numpy as np
from PIL import Image, ImageDraw

home = os.path.expanduser("~")
root = f"{home}/datasets/MVTec_split/aitex_02"


def strip_id(p):
    return re.sub(r"_t\d+\.png$", "", os.path.basename(p))


picks = {}
for p in sorted(glob.glob(f"{root}/train_good/*.png")):
    a = np.asarray(Image.open(p).convert("L"), dtype=float)
    if 115 <= a.mean() <= 125 and 15 <= a.std() <= 25:
        picks.setdefault(strip_id(p), p)
train3 = [("train " + k, p) for k, p in list(picks.items())[:3]]

def t(strip):
    path = f"{root}/test_good/{strip}_t05.png"
    return path if os.path.exists(path) else None

items = train3 + [("0022_019 NOT flagged", t("0022_019_02"))]
items += [(s[:8] + " flagged", t(s)) for s in ("0010_006_02", "0011_006_02", "0030_019_02", "0068_023_02")]

W = 256
canvas = Image.new("RGB", (W * 4, 2 * (W + 20)), "white")
draw = ImageDraw.Draw(canvas)
for i, (label, path) in enumerate(items):
    print(label, "->", path)
    x, y = (i % 4) * W, (i // 4) * (W + 20)
    draw.text((x + 4, y + 4), label, fill="black")
    if path:
        canvas.paste(Image.open(path).convert("RGB"), (x, y + 20))
os.makedirs(f"{home}/to_upload", exist_ok=True)
canvas.save(f"{home}/to_upload/compare_tiles2.png")
print("saved", f"{home}/to_upload/compare_tiles2.png")
