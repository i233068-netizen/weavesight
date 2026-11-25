import json, re
import numpy as np

d = json.load(open("results/ayesha_tilescores_dinomaly_aitex_02.json"))
thr = d["threshold_P99"]
whole = {"0010_006_02", "0011_006_02", "0012_006_02", "0013_006_02", "0014_006_02",
         "0015_006_02", "0030_019_02", "0068_023_02", "0069_023_02", "0070_023_02"}


def sid(n):
    return re.match(r"(.+)_t\d+\.png$", n).group(1)


def auroc(neg, pos):
    neg = np.array(neg)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


good, dfc = d["test_good"], d["test_defect"]
print("threshold %.5f (fixed, P99 of val_good)" % thr)
for label, sel in (("all tiles", lambda k: True),
                   ("inside the 10 strips", lambda k: k in whole),
                   ("outside the 10 strips", lambda k: k not in whole)):
    g = [s for n, s in good if sel(sid(n))]
    f = [s for n, s in dfc if sel(sid(n))]
    fa = sum(s > thr for s in g)
    cu = sum(s > thr for s in f)
    line = "%-22s normal flagged %3d/%3d | defects caught %2d/%2d" % (label, fa, len(g), cu, len(f))
    if g and f:
        line += " | image AUROC %.4f" % auroc(g, f)
    print(line)
