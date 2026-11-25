import json, re

d = json.load(open("results/ayesha_tilescores_dinomaly_aitex_02.json"))
thr = d["threshold_P99"]
whole = {"0010_006_02", "0011_006_02", "0012_006_02", "0013_006_02", "0014_006_02",
         "0015_006_02", "0030_019_02", "0068_023_02", "0069_023_02", "0070_023_02"}
between = {"0016_006_02"} | {"00%d_019_02" % i for i in range(22, 30)} | {"00%d_030_02" % i for i in range(85, 89)}


def sid(n):
    return re.match(r"(.+)_t\d+\.png$", n).group(1)


def pos(n):
    return int(re.search(r"_t(\d+)\.png$", n).group(1))


def pbin(p):
    return "t00" if p == 0 else "t01" if p == 1 else "t15" if p == 15 else "t02-t14"


defective = {sid(n) for n, _ in d["test_defect"]}


def grp(k):
    if k in whole:
        return None
    if k in between:
        return "defective, in-between weave"
    if k in defective:
        return "defective, matching weave"
    return "defect-free strip"


print("normal test tiles outside the 10 strips, flagged/total at fixed P99 threshold %.5f" % thr)
cnt = {}
for n, s in d["test_good"]:
    g = grp(sid(n))
    if g is None:
        continue
    c = cnt.setdefault((g, pbin(pos(n))), [0, 0])
    c[0] += s > thr
    c[1] += 1
groups = ["defect-free strip", "defective, matching weave", "defective, in-between weave"]
bins = ["t00", "t01", "t02-t14", "t15"]
print("%-30s" % "" + "".join("%-12s" % b for b in bins) + "total")
for g in groups:
    row = "%-30s" % g
    tf = tt = 0
    for b in bins:
        f, t = cnt.get((g, b), [0, 0])
        tf += f
        tt += t
        row += "%-12s" % ("%d/%d" % (f, t))
    print(row + "%d/%d" % (tf, tt))
