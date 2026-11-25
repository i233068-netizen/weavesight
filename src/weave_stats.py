import glob, os, re
import numpy as np
from PIL import Image

home = os.path.expanduser("~")
root = f"{home}/datasets/MVTec_split/aitex_02"
whole = {"0010_006_02", "0011_006_02", "0012_006_02", "0013_006_02", "0014_006_02",
         "0015_006_02", "0030_019_02", "0068_023_02", "0069_023_02", "0070_023_02"}
POS = range(3, 13)
N = 256
win = np.outer(np.hanning(N), np.hanning(N))
f = np.fft.fftshift(np.fft.fftfreq(N)) * N
fy, fx = np.meshgrid(f, f, indexing="ij")
r = np.hypot(fx, fy)
band = (r >= 3) & (r <= 100)


def strip_id(p):
    return re.sub(r"_t\d+\.png$", "", os.path.basename(p))


def pos(p):
    return int(re.search(r"_t(\d+)\.png$", p).group(1))


def peak(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=float)
    assert a.shape == (N, N), (path, a.shape)
    a = (a - a.mean()) * win
    p = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    pb = np.where(band, p, 0.0)
    iy, ix = np.unravel_index(np.argmax(pb), pb.shape)
    return r[iy, ix], np.degrees(np.arctan2(fy[iy, ix], fx[iy, ix])) % 180, pb[iy, ix] / pb.sum()


def circ(angles):
    z = np.mean(np.exp(1j * np.radians(2 * np.array(angles))))
    return (np.degrees(np.angle(z)) / 2) % 180, abs(z)


def summary(label, res):
    fr = np.array([x[0] for x in res])
    an = [x[1] for x in res]
    sh = np.array([x[2] for x in res])
    m, R = circ(an)
    print("%-34s n=%3d | freq cycles/tile p10/med/p90 %.1f/%.1f/%.1f | period px med %.1f | angle mean %.0f deg (R=%.2f) | peak share med %.3f"
          % (label, len(res), *np.percentile(fr, [10, 50, 90]), N / np.median(fr), m, R, np.median(sh)))


train = [peak(p) for d in ("train_good", "val_good") for p in glob.glob(f"{root}/{d}/*.png") if pos(p) in POS]
defective = {strip_id(p) for p in glob.glob(f"{root}/test_defect/*.png")}
groups = {"flagged": {}, "other_def": {}, "defect_free": {}}
for p in glob.glob(f"{root}/test_good/*.png"):
    if pos(p) not in POS:
        continue
    k = strip_id(p)
    if k in whole:
        g = "flagged"
    elif k in defective:
        g = "other_def"
    else:
        g = "defect_free"
    groups[g].setdefault(k, []).append(peak(p))

print("Positions t03-t12, test_good tiles only for test groups. Window: Hann; band 3-100 cycles/tile.")
summary("train+val", train)
for g, name in (("defect_free", "defect-free test strips"), ("flagged", "10 flagged strips"), ("other_def", "other defective strips")):
    allres = [x for v in groups[g].values() for x in v]
    summary(name, allres)
print()
for g in ("flagged", "other_def"):
    print("per strip:", g)
    for k in sorted(groups[g]):
        res = groups[g][k]
        m, R = circ([x[1] for x in res])
        print("  %s n=%2d freq med %.1f angle %.0f R=%.2f" % (k, len(res), np.median([x[0] for x in res]), m, R))
