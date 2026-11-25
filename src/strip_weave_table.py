import argparse, glob, json, os, re
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
args = parser.parse_args()

from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

classes = {"dinomaly": Dinomaly, "patchcore": Patchcore}
home = os.path.expanduser("~")
pattern = f"{home}/results/{args.model}_split_{args.category}*/**/weights/lightning/model.ckpt"
ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
assert len(ckpts) == 1, f"expected one checkpoint, found {ckpts}"
model = classes[args.model].load_from_checkpoint(ckpts[0])
model.post_processor = None
engine = Engine(default_root_dir=f"{home}/results/threshold_runs")
root = f"{home}/datasets/MVTec_split/{args.category}"


def score(folder):
    out = []
    for b in engine.predict(model=model, data_path=folder):
        out.extend(zip([os.path.basename(str(p)) for p in b.image_path], b.pred_score.flatten().tolist()))
    return out


val = score(f"{root}/val_good")
good = score(f"{root}/test_good")
dfc = score(f"{root}/test_defect")
thr = float(np.percentile([s for _, s in val], 99))
out = f"{home}/weavesight/results/ayesha_tilescores_{args.model}_{args.category}.json"
json.dump({"model": args.model, "category": args.category, "threshold_P99": thr,
           "val_good": val, "test_good": good, "test_defect": dfc}, open(out, "w"))
print("saved", out)
print("P99 threshold recomputed: %.5f" % thr)

N = 256
win = np.outer(np.hanning(N), np.hanning(N))
f = np.fft.fftshift(np.fft.fftfreq(N)) * N
fy, fx = np.meshgrid(f, f, indexing="ij")
r = np.hypot(fx, fy)
band = (r >= 3) & (r <= 100)


def peak(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=float)
    a = (a - a.mean()) * win
    p = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    pb = np.where(band, p, 0.0)
    iy, ix = np.unravel_index(np.argmax(pb), pb.shape)
    return r[iy, ix], np.degrees(np.arctan2(fy[iy, ix], fx[iy, ix])) % 180


def sid(n):
    return re.match(r"(.+)_t\d+\.png$", n).group(1)


def pos(n):
    return int(re.search(r"_t(\d+)\.png$", n).group(1))


whole = {"0010_006_02", "0011_006_02", "0012_006_02", "0013_006_02", "0014_006_02",
         "0015_006_02", "0030_019_02", "0068_023_02", "0069_023_02", "0070_023_02"}
defective = {sid(n) for n, _ in dfc}
rows = {}
for n, s in good:
    if 3 <= pos(n) <= 12:
        rows.setdefault(sid(n), []).append((n, s > thr))
print("\nstrip         group        tiles t03-t12 flagged | weave freq  angle  diff-from-144")
for k in sorted(rows):
    g = "FLAGGED10" if k in whole else ("other_def" if k in defective else "defect_free")
    ps = [peak(f"{root}/test_good/{n}") for n, _ in rows[k]]
    fr = np.median([x[0] for x in ps])
    z = np.mean(np.exp(1j * np.radians(2 * np.array([x[1] for x in ps]))))
    ang = (np.degrees(np.angle(z)) / 2) % 180
    d = abs(ang - 144)
    d = min(d, 180 - d)
    print("%s  %-11s  %2d  %2d | %5.1f  %5.0f  %5.0f" % (k, g, len(rows[k]), sum(x[1] for x in rows[k]), fr, ang, d))
