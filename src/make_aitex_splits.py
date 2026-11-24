"""Create strip-level splits for AITEX (before any tiling).

Per fabric, the defect-free strips are shuffled and split into
val / test / train. Defective strips all go to test, with their masks
(a few strips have two masks, which are merged later). Strips without a
mask, or from a fabric with no defect-free folder, are skipped and listed.
"""
import argparse
import csv
import random
from pathlib import Path


def fabric_code(path):
    # '0001_000_05.png' -> '05' (the last number is the fabric)
    return Path(path).stem.split("_")[2]


def make_splits(root, out_csv, seed=42, n_val=4, n_test=4):
    root = Path(root)

    def rel(p):
        return p.relative_to(root).as_posix()

    rows = []
    normal_fabrics = set()
    for folder in sorted((root / "NODefect_images").iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.png"))
        fab = fabric_code(files[0])
        normal_fabrics.add(fab)
        random.Random(f"{seed}-{fab}").shuffle(files)
        groups = {
            "val": files[:n_val],
            "test": files[n_val:n_val + n_test],
            "train": files[n_val + n_test:],
        }
        for split, group in groups.items():
            for p in sorted(group):
                rows.append([fab, split, "good", rel(p), ""])

    skipped = []
    for p in sorted((root / "Defect_images").glob("*.png")):
        fab = fabric_code(p)
        masks = sorted((root / "Mask_images").glob(f"{p.stem}_mask*.png"))
        if fab not in normal_fabrics or not masks:
            skipped.append(p.name)
            continue
        rows.append([fab, "test", "defect", rel(p), ";".join(rel(m) for m in masks)])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fabric", "split", "label", "image_path", "mask_path"])
        writer.writerows(rows)
    return rows, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/content/datasets/AITEX")
    parser.add_argument("--out", default="splits/aitex_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_val", type=int, default=4)
    parser.add_argument("--n_test", type=int, default=4)
    args = parser.parse_args()

    rows, skipped = make_splits(args.root, args.out, args.seed, args.n_val, args.n_test)
    for fab in sorted({r[0] for r in rows}):
        counts = {}
        for r in rows:
            if r[0] == fab:
                key = f"{r[1]}-{r[2]}"
                counts[key] = counts.get(key, 0) + 1
        print("fabric", fab, counts)
    print("skipped:", skipped)
