import json, os, sys
import numpy as np
from PIL import Image
home = os.path.expanduser("~")

def kind(n): return "defect-free strip" if n.split("_")[1] == "000" else "defective strip"
def pos(n):
    t = int(n[:-4].split("_")[3][1:])
    return "t00" if t == 0 else ("t15" if t == 15 else "t01-t14")
def rate(items, thr, above=True):
    k = sum((s > thr) for _, s in items)
    return f"{k}/{len(items)} ({100*k/len(items):.1f}%)" if items else "0/0"

for c in sys.argv[1:]:
    ts = json.load(open(f"{home}/weavesight/results/manahil_tilescores_cae_aitex_{c}.json"))
    thr, good, bad = ts["threshold_P99"], ts["test_good"], ts["test_defect"]
    print(f"\n===== CAE aitex_{c} (POST-HOC) P99 threshold {thr:.4f} =====")
    print("-- normal test tiles flagged, by strip type")
    for k in ["defect-free strip", "defective strip"]:
        print(f"  {k}: {rate([x for x in good if kind(x[0]) == k], thr)}")
    print("-- by tile position: normal flagged | defects caught")
    for p in ["t00", "t01-t14", "t15"]:
        print(f"  {p}: {rate([x for x in good if pos(x[0]) == p], thr)} | {rate([x for x in bad if pos(x[0]) == p], thr)}")
    strips = {}
    for n, s in good:
        d = strips.setdefault(n.split("_")[0], [0, 0]); d[1] += 1; d[0] += s > thr
    top = sorted(strips.items(), key=lambda kv: -kv[1][0])[:5]
    print("-- strips with most false alarms:", ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in top if v[0]))
    if len(bad) < 6:
        print("-- too few defect tiles for size bins; per tile:")
    areas = []
    for n, s in bad:
        m = np.array(Image.open(f"{home}/datasets/AITEX_tiles/fabric_{c}/test_mask/{n}").convert("L")) > 127
        areas.append(int(m.sum()))
        if len(bad) < 6: print(f"  {n}: mask pixels {areas[-1]}, score {s:.4f}, caught {s > thr}")
    if len(bad) >= 6:
        a = np.array(areas); e = np.percentile(a, [33.3, 66.7])
        grp = np.where(a <= e[0], 0, np.where(a <= e[1], 1, 2))
        print("-- defects caught by size (terciles of mask pixels)")
        for g, name in enumerate(["small", "medium", "large"]):
            sel = [bad[i] for i in range(len(bad)) if grp[i] == g]
            print(f"  {name} ({a[grp == g].min()}-{a[grp == g].max()} px): {rate(sel, thr)}")
