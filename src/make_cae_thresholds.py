import json, os
home = os.path.expanduser("~")
res = f"{home}/weavesight/results"
cats = ["carpet", "grid", "leather", "tile", "wood"] + [f"aitex_{i:02d}" for i in range(7)]
out = {}
for c in cats:
    r, p = f"{res}/manahil_rawthreshold_cae_{c}.json", f"{res}/manahil_pixel_cae_{c}.json"
    if not (os.path.exists(r) and os.path.exists(p)):
        print("skip (files missing):", c); continue
    r, p = json.load(open(r)), json.load(open(p))
    out[f"cae/{c}"] = {
        "image_threshold": r["rules"]["P99_main"]["threshold"],
        "pixel_threshold": p["pixel_threshold_p99.9_val"],
        "n_val_images": len(r["val_scores"]),
        "image_rule": "99th percentile of raw val_good image scores",
        "pixel_rule": "99.9th percentile of all raw val_good pixel values"}
    print("ok:", c)
dst = os.path.expanduser("~/weavesight/app/thresholds_cae.json")
json.dump(out, open(dst, "w"), indent=2)
print("wrote", dst)
