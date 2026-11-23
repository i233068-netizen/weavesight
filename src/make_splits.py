"""Create image-level train/val/test splits for the MVTec texture classes.

Training images are defect-free. A fixed fraction of them is held out as a
defect-free validation set (used only to calibrate the threshold). The
original MVTec test images stay as the test set. Splitting happens per image,
before any patches are cut, so patches of one image never cross splits.
"""
import argparse
import csv
import random
from pathlib import Path

CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]


def make_splits(root, out_csv, seed=42, val_fraction=0.1):
    root = Path(root)
    rows = []
    for cat in CATEGORIES:
        rng = random.Random(f"{seed}-{cat}")
        train_imgs = sorted((root / cat / "train" / "good").glob("*.png"))
        rng.shuffle(train_imgs)
        n_val = max(1, round(len(train_imgs) * val_fraction))
        val_set = set(train_imgs[:n_val])
        for p in sorted(train_imgs):
            split = "val" if p in val_set else "train"
            rows.append([cat, split, "good", p.relative_to(root).as_posix(), ""])
        for p in sorted((root / cat / "test").glob("*/*.png")):
            label = p.parent.name
            mask = ""
            if label != "good":
                m = root / cat / "ground_truth" / label / f"{p.stem}_mask.png"
                mask = m.relative_to(root).as_posix()
            rows.append([cat, "test", label, p.relative_to(root).as_posix(), mask])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "split", "label", "image_path", "mask_path"])
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/content/datasets/MVTecAD")
    parser.add_argument("--out", default="splits/mvtec_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    rows = make_splits(args.root, args.out, args.seed, args.val_fraction)
    for cat in CATEGORIES:
        counts = {}
        for r in rows:
            if r[0] == cat:
                counts[r[1]] = counts.get(r[1], 0) + 1
        print(cat, counts)
