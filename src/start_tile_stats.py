import glob, json, os, re
import numpy as np
from PIL import Image

home = os.path.expanduser("~")
root = f"{home}/datasets/MVTec_split/aitex_02"
d = json.load(open(f"{home}/weavesight/results/ayesha_tilescores_dinomaly_aitex_02.json"))
thr = d["threshold_P99"]
score = {n: s for n, s in d["test_good"]}
whole = {"0010_006_02", "0011_006_02", "0012_006_02", "0013_006_02", "0014_006_02",
         "0015_006_02", "0030_019_02", "0068_023_02", "0069_023_02", "0070_023_02"}


def sid(n):
    return re.match(r"(.+)_t\d+\.png$", n).group(1)


def pos(n):
    return int(re.search(r"_t(\d+)\.png$", n).group(1))


def stats(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=float)
    return a.mean(), a.std(), (a > 240).mean()


defective = {sid(n) for n, _ in d["test_defect"]}
rows = {}


def add(key, p):
    rows.setdefault(key, []).append(stats(p))


for folder in ("train_good", "val_good"):
    for p in glob.glob(f"{root}/{folder}/*.png"):
        if pos(os.path.basename(p)) in (0, 1):
            add(("train+val", pos(os.path.basename(p))), p)
for p in glob.glob(f"{root}/test_good/*.png"):
    n = os.path.basename(p)
    if pos(n) not in (0, 1):
        continue
    k = sid(n)
    if k in whole:
        g = "test: the 10 flagged strips"
    elif k in defective:
        g = "test: defective, flagged" if score[n] > thr else "test: defective, not flagged"
    else:
        g = "test: defect-free strips"
    add((g, pos(n)), p)

print("%-30s %-4s %4s | brightness med | contrast med | white-fraction p10/med/p90" % ("group", "pos", "n"))
for key in sorted(rows, key=lambda k: (k[1], k[0])):
    a = np.array(rows[key])
    wf = np.percentile(a[:, 2], [10, 50, 90])
    print("%-30s t%02d  %4d | %14.1f | %12.1f | %.2f / %.2f / %.2f"
          % (key[0], key[1], len(a), np.median(a[:, 0]), np.median(a[:, 1]), *wf))
