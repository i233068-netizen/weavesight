import argparse, csv, os
import numpy as np
from collections import Counter
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--tile", type=int, default=256)
parser.add_argument("--root", default=os.path.expanduser("~/datasets/AITEX"))
parser.add_argument("--out", default=os.path.expanduser("~/datasets/AITEX_tiles"))
parser.add_argument("--csv", default=os.path.expanduser("~/weavesight/splits/aitex_seed42.csv"))
args = parser.parse_args()
root, out, t = Path(args.root), Path(args.out), args.tile
mask_dir = root / "Mask_images" / "Mask_images"
assert mask_dir.exists(), mask_dir

def resolve(rel):
    p = root / rel
    if p.exists():
        return p
    parts = Path(rel).parts
    p2 = root / parts[0] / parts[0] / Path(*parts[1:])  # extra folder level on this PC
    assert p2.exists(), f"cannot find {rel}"
    return p2

shown = [False]
def load_mask(stem):
    files = sorted(mask_dir.glob(f"{stem}_mask*.png"))
    assert files, f"no mask for {stem}"
    if not shown[0]:
        print("MASK VALUES (example):", np.unique(np.array(Image.open(files[0]))))
        shown[0] = True
    m = None
    for f in files:
        a = np.array(Image.open(f).convert("L"))
        m = a if m is None else np.maximum(m, a)  # union of double masks, grey levels kept
    return m, len(files)

FOLDER = {"train": "train_good", "val": "val_good", "test": "test_good"}
counts, alpha_warn, merged, skipped = Counter(), 0, 0, 0
rows = list(csv.DictReader(open(args.csv)))
for r in rows:
    fab, split, label = r["fabric"], r["split"], r["label"]
    src = resolve(r["image_path"])
    stem = src.stem
    im = Image.open(src)
    if im.mode == "LA" and np.array(im.split()[1]).min() < 255:
        alpha_warn += 1
    img = im.convert("RGB")
    W, H = img.size
    base = out / f"fabric_{fab}"
    mask = None
    if label != "good":
        mask, n = load_mask(stem)
        assert mask.shape == (H, W), f"mask size mismatch for {stem}"
        merged += (n > 1)
    idx = 0
    for y in range(0, H - t + 1, t):
        for x in range(0, W - t + 1, t):
            tile = img.crop((x, y, x + t, y + t))
            name = f"{stem}_t{idx:02d}.png"
            idx += 1
            if label == "good":
                folder = FOLDER[split]
            elif (mask[y:y + t, x:x + t] > 127).any():
                folder = "test_defect"
                (base / "test_mask").mkdir(parents=True, exist_ok=True)
                Image.fromarray(((mask[y:y + t, x:x + t] > 127) * 255).astype(np.uint8)).save(base / "test_mask" / name)
            elif (mask[y:y + t, x:x + t] > 0).any():
                skipped += 1  # only faint edge pixels: ambiguous, excluded
                continue
            else:
                folder = "test_good"
            (base / folder).mkdir(parents=True, exist_ok=True)
            tile.save(base / folder / name)
            counts[(fab, folder)] += 1

print("STRIPS", len(rows), "| strips with merged double masks:", merged, "| strips with alpha<255:", alpha_warn, "| ambiguous tiles skipped:", skipped)
for fab in sorted({k[0] for k in counts}):
    print("FABRIC", fab, {k[1]: v for k, v in sorted(counts.items()) if k[0] == fab})
print("TOTAL TILES", sum(counts.values()))
