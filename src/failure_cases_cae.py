import argparse, json, os
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from cae_common import *

p = argparse.ArgumentParser()
p.add_argument("--category", required=True)
p.add_argument("--data", default="MVTec_split")
p.add_argument("--n", type=int, default=4)
p.add_argument("--device", default="cpu")
a = p.parse_args()
home = os.path.expanduser("~")
root = f"{home}/datasets/{a.data}/{a.category}"
ts = json.load(open(f"{home}/weavesight/results/manahil_tilescores_cae_{a.category}.json"))
thr = ts["threshold_P99"]
good, bad = ts["test_good"], ts["test_defect"]
fps = sorted([x for x in good if x[1] > thr], key=lambda x: -x[1])
miss = sorted([x for x in bad if x[1] <= thr], key=lambda x: x[1])
print(f"RESULT cae {a.category}: threshold {thr:.4f} | false alarms {len(fps)}/{len(good)} | missed defects {len(miss)}/{len(bad)}")
rows = [("FALSE ALARM (worst case)", "test_good", n, s) for n, s in fps[:a.n]] + \
       [("MISSED DEFECT (worst case)", "test_defect", n, s) for n, s in miss[:a.n]]
if not rows:
    print("NO FAILURES TO SHOW"); raise SystemExit
ck = torch.load(f"{home}/results/cae_{a.category}/cae_{a.category}.pt", map_location=a.device)
model = CAE().to(a.device); model.load_state_dict(ck["state_dict"]); model.eval()
items = []
for kind, folder, n, s in rows:
    img = load_img(f"{root}/{folder}/{n}")
    am = anomaly_map(model, img.copy(), a.device)
    print(f"{n}: saved score {s:.4f}, recomputed {image_score(am):.4f}")
    items.append((kind, folder, n, s, img, am))
vmin = min(x[5].min() for x in items); vmax = max(x[5].max() for x in items)
fig, axes = plt.subplots(len(items), 3, figsize=(9, 3 * len(items)), squeeze=False)
for r, (kind, folder, n, s, img, am) in enumerate(items):
    axes[r][0].imshow(img); axes[r][0].set_title(f"{kind}\n{n}", fontsize=8)
    axes[r][1].imshow(img); axes[r][1].imshow(am, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    axes[r][1].set_title(f"score {s:.4f} (threshold {thr:.4f})", fontsize=8)
    axes[r][2].imshow(img)
    if folder == "test_defect":
        m = Image.open(f"{root}/test_mask/{n}").convert("L").resize((SIZE, SIZE), Image.NEAREST)
        mm = np.array(m) > 127
        if mm.any():
            axes[r][2].contour(mm.astype(float), levels=[0.5], colors="lime", linewidths=1)
        axes[r][2].set_title("annotated defect (green outline)", fontsize=8)
    else:
        axes[r][2].set_title("no defect annotated", fontsize=8)
    for ax in axes[r]: ax.axis("off")
plt.tight_layout()
os.makedirs(f"{home}/weavesight/results/failures", exist_ok=True)
out = f"{home}/weavesight/results/failures/cae_{a.category}.png"
plt.savefig(out, dpi=110)
print("SAVED", out)
