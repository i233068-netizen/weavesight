import json, os, sys, time
from PIL import Image

home = os.path.expanduser("~")
sys.path.insert(0, f"{home}/weavesight/app")
from predict import predict

d = json.load(open(f"{home}/weavesight/results/ayesha_tilescores_dinomaly_aitex_02.json"))
thr = d["threshold_P99"]
root = f"{home}/datasets/MVTec_split/aitex_02"
todo = [("test_good", n, s) for n, s in d["test_good"][:8]] + [("test_defect", n, s) for n, s in d["test_defect"][:4]]
print("saved P99 threshold %.5f" % thr)
print("%-12s %-22s %10s %10s %10s %s" % ("folder", "tile", "predict()", "saved", "abs diff", "verdict vs saved flag"))
worst = 0.0
for folder, n, saved in todo:
    t0 = time.time()
    out = predict(Image.open(f"{root}/{folder}/{n}"), "dinomaly", "aitex_02")
    dt = time.time() - t0
    diff = abs(out["score"] - saved)
    worst = max(worst, diff)
    saved_flag = "FAIL" if saved > thr else "PASS"
    print("%-12s %-22s %10.5f %10.5f %10.6f %s vs %s  (%.2fs)" % (folder, n, out["score"], saved, diff, out["verdict"], saved_flag, dt))
print("worst abs difference: %.6f" % worst)
