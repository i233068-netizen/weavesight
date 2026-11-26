import csv, json, os
R = os.path.expanduser("~/weavesight/results")
cats = ["carpet", "grid", "leather", "tile", "wood"] + [f"aitex_0{i}" for i in range(7)]
models = [("cae", "manahil"), ("dinomaly", "ayesha"), ("patchcore", "ayesha")]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


rows = []
for c in cats:
    for m, who in models:
        t = load(f"{R}/{who}_rawthreshold_{m}_{c}.json")
        p = load(f"{R}/{who}_pixel_{m}_{c}.json")
        row = {"category": c, "model": m, "image_auroc": None, "pixel_auroc": None, "aupro": None,
               "p99_normal_flagged": None, "p99_defects_caught": None}
        if t:
            row["image_auroc"] = t["image_auroc_raw"]
            row["p99_normal_flagged"] = t["rules"]["P99_main"]["fpr_test_good"]
            row["p99_defects_caught"] = t["rules"]["P99_main"]["recall_test_defect"]
        if p:
            row["pixel_auroc"] = p["pixel_auroc_raw"]
            row["aupro"] = p["aupro_fpr0.3"]
        rows.append(row)
out = f"{R}/manahil_merged_table.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
fmt = lambda v: "  -   " if v is None else f"{v:.4f}"
print("category   model       imgAUROC pixAUROC AUPRO   P99_fpr P99_recall")
for r in rows:
    print(f"{r['category']:<10} {r['model']:<10} {fmt(r['image_auroc'])}  {fmt(r['pixel_auroc'])}  {fmt(r['aupro'])}  {fmt(r['p99_normal_flagged'])}  {fmt(r['p99_defects_caught'])}")
print("saved", out)
