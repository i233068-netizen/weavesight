import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/weavesight/app"))
from predict_cae import predict_cae
cat = sys.argv[1]
home = os.path.expanduser("~")
ts = json.load(open(f"{home}/weavesight/results/manahil_tilescores_cae_{cat}.json"))
thr = ts["threshold_P99"]
rows = [("test_good", x) for x in ts["test_good"][:6]] + [("test_defect", x) for x in ts["test_defect"][:6]]
worst, bad = 0.0, 0
for folder, (n, s) in rows:
    r = predict_cae(Image.open(f"{home}/datasets/MVTec_split/{cat}/{folder}/{n}"), cat)
    worst = max(worst, abs(r["score"] - s))
    bad += (r["verdict"] == "FAIL") != (s > thr)
    assert r["heatmap"].shape == r["mask"].shape
print(f"{len(rows)} tiles: max score difference {worst:.6f}, verdict mismatches {bad}")
