import glob, os
from PIL import Image, ImageDraw

home = os.path.expanduser("~")
root = f"{home}/datasets/MVTec_split/aitex_02"


def pick(pattern):
    files = sorted(glob.glob(pattern))
    return files[len(files) // 2] if files else None


items = [
    ("training tile", pick(f"{root}/train_good/*.png")),
    ("0010_006 flagged strip", pick(f"{root}/test_good/0010_006_02_t0[4-9].png")),
    ("0030_019 flagged strip", pick(f"{root}/test_good/0030_019_02_t0[4-9].png")),
    ("0068_023 flagged strip", pick(f"{root}/test_good/0068_023_02_t0[4-9].png")),
    ("0022_019 NOT flagged", pick(f"{root}/test_good/0022_019_02_t0[4-9].png")),
]
W = 256
canvas = Image.new("RGB", (W * len(items), W + 20), "white")
for i, (label, path) in enumerate(items):
    print(label, "->", path)
    if path:
        canvas.paste(Image.open(path).convert("RGB"), (i * W, 20))
    ImageDraw.Draw(canvas).text((i * W + 4, 4), label, fill="black")
os.makedirs(f"{home}/to_upload", exist_ok=True)
canvas.save(f"{home}/to_upload/compare_tiles.png")
print("saved", f"{home}/to_upload/compare_tiles.png")
