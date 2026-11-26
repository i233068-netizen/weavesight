import argparse, json, os
import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from cae_common import *

p = argparse.ArgumentParser()
p.add_argument("--category", required=True)
p.add_argument("--data", default="MVTec_split")
p.add_argument("--folder", default=None)
a = p.parse_args()

dev = "cuda"
home = os.path.expanduser("~")
root = f"{home}/datasets/{a.data}/{a.folder or a.category}"
ck = torch.load(f"{home}/results/cae_{a.category}/cae_{a.category}.pt", map_location=dev)
assert ck["size"] == SIZE and ck["patch"] == PATCH and ck["stride"] == STRIDE and ck["sigma"] == SIGMA
print("checkpoint steps:", ck["steps"], flush=True)
model = CAE().to(dev)
model.load_state_dict(ck["state_dict"])
model.eval()


def run(folder):
    names = list_images(folder)
    maps, scores = [], []
    for n in names:
        im = load_img(f"{folder}/{n}").copy()
        am = anomaly_map(model, im, dev)
        maps.append(am)
        scores.append(image_score(am))
    return names, maps, scores


vn, vm, vs = run(f"{root}/val_good")
gn, gm, gs = run(f"{root}/test_good")
dn, dm, ds = run(f"{root}/test_defect")
print("counts val/good/defect:", len(vn), len(gn), len(dn), flush=True)

val, tg, td = np.array(vs), np.array(gs), np.array(ds)
auroc = float(roc_auc_score([0] * len(tg) + [1] * len(td), np.concatenate([tg, td])))
print(f"IMAGE AUROC (raw, max of map): {auroc:.4f}")

rules = {"P99_main": float(np.percentile(val, 99)),
         "val_max_sensitivity": float(val.max()),
         "mean_plus_3std_sensitivity": float(val.mean() + 3 * val.std())}
res = {}
for k, thr in rules.items():
    res[k] = {"threshold": thr, "fpr_test_good": float(np.mean(tg > thr)), "recall_test_defect": float(np.mean(td > thr))}
    print(f"{k}: thr={thr:.4f} fpr={res[k]['fpr_test_good']:.3f} recall={res[k]['recall_test_defect']:.3f}")


def prep_mask(path):
    m = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return (np.array(m) > 127).astype(np.float32)


masks = [np.zeros((SIZE, SIZE), dtype=np.float32) for _ in gm]
for n in dn:
    pth = f"{root}/test_mask/{n}"
    assert os.path.exists(pth), pth
    masks.append(prep_mask(pth))
preds = np.stack(gm + dm).astype(np.float32)
target = np.stack(masks)
print("MASK FRACTION:", float(target.mean()), "map size", preds.shape[1:])
pix = float(roc_auc_score(target.ravel().astype(int), preds.ravel()))
print(f"PIXEL AUROC (raw): {pix:.4f}")
aupro = None
try:
    from anomalib.metrics.aupro import _AUPRO
    mt = _AUPRO(fpr_limit=0.3)
    mt.update(torch.from_numpy(preds), torch.from_numpy(target))
    aupro = float(mt.compute())
    print(f"AUPRO (fpr 0.3): {aupro:.4f}")
except Exception as e:
    print("AUPRO FAILED:", repr(e))
pix_thr = float(np.percentile(np.concatenate([m.ravel() for m in vm]), 99.9))

os.makedirs(f"{home}/weavesight/results", exist_ok=True)
c = a.category
with open(f"{home}/weavesight/results/manahil_rawthreshold_cae_{c}.json", "w") as f:
    json.dump({"model": "cae", "category": c, "scores": "raw (max of smoothed L1 residual map)",
               "image_auroc_raw": auroc, "rules": res, "val_scores": vs,
               "test_good_scores": gs, "test_defect_scores": ds}, f, indent=2)
with open(f"{home}/weavesight/results/manahil_pixel_cae_{c}.json", "w") as f:
    json.dump({"model": "cae", "category": c, "map_size": [SIZE, SIZE], "image_auroc": auroc,
               "pixel_auroc_raw": pix, "aupro_fpr0.3": aupro, "pixel_threshold_p99.9_val": pix_thr,
               "mask_alignment": "masks resized to 256x256 (nearest), same size as images; no crop"}, f, indent=2)
with open(f"{home}/weavesight/results/manahil_tilescores_cae_{c}.json", "w") as f:
    json.dump({"model": "cae", "category": c, "threshold_P99": rules["P99_main"],
               "val_good": [[n, s] for n, s in zip(vn, vs)],
               "test_good": [[n, s] for n, s in zip(gn, gs)],
               "test_defect": [[n, s] for n, s in zip(dn, ds)]}, f, indent=2)
print("saved 3 result files for", c)
