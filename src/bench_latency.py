import argparse, json, os, sys, time
import numpy as np, torch
from PIL import Image
home = os.path.expanduser("~")
sys.path.insert(0, f"{home}/weavesight/app")
p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["cae", "dinomaly", "patchcore"])
p.add_argument("--category", required=True)
p.add_argument("--data", default="MVTec_split")
a = p.parse_args()
if a.model == "cae":
    from predict_cae import predict_cae
    fn = lambda im: predict_cae(im, a.category)
else:
    from predict import predict
    fn = lambda im: predict(im, a.model, a.category)
folder = f"{home}/datasets/{a.data}/{a.category}/test_good"
names = sorted(os.listdir(folder))[:20]
imgs = [Image.open(f"{folder}/{n}").convert("RGB") for n in names]
sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
sync(); t = time.perf_counter(); fn(imgs[0]); sync()
first = (time.perf_counter() - t) * 1000
for i in range(5): fn(imgs[i % len(imgs)])
ts = []
for i in range(50):
    sync(); t = time.perf_counter(); fn(imgs[i % len(imgs)]); sync()
    ts.append((time.perf_counter() - t) * 1000)
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
res = {"model": a.model, "category": a.category, "device": dev, "n_images": len(imgs), "warmup": 5,
       "timed_runs": 50, "first_call_ms": round(first, 1), "median_ms": round(float(np.median(ts)), 1),
       "p95_ms": round(float(np.percentile(ts, 95)), 1)}
print(json.dumps(res))
os.makedirs(f"{home}/weavesight/results", exist_ok=True)
json.dump(res, open(f"{home}/weavesight/results/manahil_latency_{a.model}_{a.category}.json", "w"), indent=2)
