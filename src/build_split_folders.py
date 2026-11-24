"""Build split-aware folders of symlinks from splits/mvtec_seed42.csv.

<dst>/<category>/train_good, val_good, test_good   (defect-free images)
<dst>/<category>/test_defect, test_mask            (defective images and
masks, same file names, prefixed with the defect type to avoid collisions)
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="splits/mvtec_seed42.csv")
parser.add_argument("--src", default=str(Path.home() / "datasets/MVTecAD"))
parser.add_argument("--dst", default=str(Path.home() / "datasets/MVTec_split"))
args = parser.parse_args()

src, dst = Path(args.src), Path(args.dst)
folders = {"train": "train_good", "val": "val_good", "test": "test_good"}
counts = Counter()

with open(args.csv) as f:
    for r in csv.DictReader(f):
        cat, split, label = r["category"], r["split"], r["label"]
        img = src / r["image_path"]
        if label == "good":
            kind = folders[split]
            links = [(dst / cat / kind / img.name, img)]
        else:
            kind = "test_defect"
            name = f"{label}_{img.name}"
            links = [
                (dst / cat / "test_defect" / name, img),
                (dst / cat / "test_mask" / name, src / r["mask_path"]),
            ]
        for link, target in links:
            if not target.exists():
                raise FileNotFoundError(target)
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        counts[(cat, kind)] += 1

for cat in ["carpet", "grid", "leather", "tile", "wood"]:
    print(cat, {k: counts[(cat, k)] for k in ["train_good", "val_good", "test_good", "test_defect"]})
