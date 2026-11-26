import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["dinomaly", "patchcore"])
parser.add_argument("--category", required=True)
args = parser.parse_args()

from anomalib.engine import Engine
from anomalib.models import Dinomaly, Patchcore

classes = {"dinomaly": Dinomaly, "patchcore": Patchcore}
home = os.path.expanduser("~")
pattern = f"{home}/results/ablation_{args.model}_{args.category}_5000/**/weights/lightning/model.ckpt"
ckpts = sorted({os.path.realpath(c) for c in glob.glob(pattern, recursive=True)})
assert len(ckpts) == 1, f"expected one checkpoint, found {ckpts}"

model = classes[args.model].load_from_checkpoint(ckpts[0])
resize_size, crop_size = None, None
for t in model.pre_processor.transform.transforms:
    if isinstance(t, v2.Resize):
        resize_size = t.size
    if isinstance(t, v2.CenterCrop):
        crop_size = t.size
print("PREPROC resize:", resize_size, "crop:", crop_size)
model.post_processor = None  # raw maps, no rescaling
engine = Engine(default_root_dir=f"{home}/results/threshold_runs")
root = f"{home}/datasets/MVTec_split/{args.category}"


def get_maps(folder):
    preds = engine.predict(model=model, data_path=folder)
    names, maps, scores = [], [], []
    for b in preds:
        am = b.anomaly_map
        if am.dim() == 4:
            am = am[:, 0]
        for i in range(am.shape[0]):
            maps.append(am[i].float().cpu().numpy())
            scores.append(float(b.pred_score[i]))
            names.append(os.path.basename(b.image_path[i]))
    return names, maps, scores


def as_hw(s):
    return (s, s) if isinstance(s, int) else (s[0], s[-1]) if len(s) > 1 else (s[0], s[0])


def prep_mask(path, map_hw):
    m = Image.open(path).convert("L")
    if resize_size is not None:
        h, w = as_hw(resize_size)
        m = m.resize((w, h), Image.NEAREST)
    if crop_size is not None:
        ch, cw = as_hw(crop_size)
        m = v2.functional.center_crop(m, [ch, cw])
    if (m.size[1], m.size[0]) != tuple(map_hw):
        print("WARNING: mask", m.size[::-1], "!= map", map_hw, "-> nearest resize")
        m = m.resize((map_hw[1], map_hw[0]), Image.NEAREST)
    return (np.array(m) > 127).astype(np.float32)


gn, gm, gs = get_maps(f"{root}/test_good")
dn, dm, ds = get_maps(f"{root}/test_defect")
H, W = gm[0].shape
assert all(m.shape == (H, W) for m in gm + dm), "maps have different sizes"
print(f"MAP SIZE {H}x{W}, n_good={len(gm)}, n_defect={len(dm)}")

masks = [np.zeros((H, W), dtype=np.float32) for _ in gm]
for n in dn:
    p = f"{root}/test_mask/{n}"
    assert os.path.exists(p), f"missing mask for {n}: {p}"
    masks.append(prep_mask(p, (H, W)))

preds_arr = np.stack(gm + dm).astype(np.float32)
target = np.stack(masks)
print("MASK FRACTION defect pixels:", float(target.mean()))

img_auroc = float(roc_auc_score([0] * len(gs) + [1] * len(ds), gs + ds))
pix_auroc = float(roc_auc_score(target.ravel().astype(int), preds_arr.ravel()))
print(f"IMAGE AUROC (pred_score): {img_auroc:.4f}")
print(f"PIXEL AUROC (raw maps): {pix_auroc:.4f}")

aupro = None
try:
    from anomalib.metrics.aupro import _AUPRO
    metric = _AUPRO(fpr_limit=0.3)
    metric.update(torch.from_numpy(preds_arr), torch.from_numpy(target))
    aupro = float(metric.compute())
    print(f"AUPRO (fpr_limit=0.3): {aupro:.4f}")
except Exception as e:
    print("AUPRO FAILED:", repr(e))

os.makedirs(f"{home}/weavesight/results", exist_ok=True)
with open(f"{home}/weavesight/results/ayesha_ablation_pixel_{args.model}_{args.category}_5000.json", "w") as f:
    json.dump({"model": args.model, "category": args.category, "map_size": [H, W],
               "preprocessor_resize": str(resize_size), "preprocessor_crop": str(crop_size),
               "image_auroc": img_auroc, "pixel_auroc_raw": pix_auroc, "aupro_fpr0.3": aupro,
               "mask_alignment": "masks resized and center-cropped like the model's images (nearest)"}, f, indent=2)
